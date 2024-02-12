"""
Code for aligning 3D pose to a fly template.
The best practice for getting good alignment is to have an accurate 3D pose and
a template whose key points are matching the tracked key points closely.

This class expects a 3D file in the following format:
>>> pose_data_dict = {
        "<side (R,L)><segment (F, M, H)>_leg": NDArray[N_frames,5,3],
        "<side (R,L)>_head": NDArray[N_frames,2,3],
        "Neck": NDArray[N_frames,1,3],
    }

Usage might differ based on the needs. For now, there are three cases that you
can use the class with:

Case 1: we have 3D pose obtained, and we would like to align it but first
we need to convert the pose data into a dictionary format
NOTE: if the 3D pose is not in the format described above, then you need to:
* Convert your 3D pose file manually to the required format
* Or, if you obtain the 3D pose from anipose (see the method `convert_from_anipose_to_dict`),
simply set `convert` to True.

>>> data_path = Path("../data/anipose_220525_aJO_Fly001_001/pose-3d")
>>> align = AlignPose.from_file_path(
>>>     main_dir=data_path,
>>>     file_name="pose3d.h5",
>>>     legs_list=["RF","LF"],
>>>     convert_dict=True,
>>>     pts2align=PTS2ALIGN,
>>>     include_claw=False,
>>>     nmf_template=NMF_TEMPLATE,
>>> )
>>> aligned_pos = align.align_pose(export_path=data_path)

Case 2: we have a pose data in the required data structure, we just want to load and align it

>>> data_path = Path("../data/anipose_220525_aJO_Fly001_001/pose-3d")
>>> align = AlignPose.from_file_path(
>>>     main_dir=data_path,
>>>     file_name="converted_pose_dict.pkl",
>>>     legs_list=["RF","LF"],
>>>     convert_dict=False,
>>>     pts2align=PTS2ALIGN,
>>>     include_claw=False,
>>>     nmf_template=NMF_TEMPLATE,
>>> )
>>> aligned_pos = align.align_pose(export_path=data_path)

Case 3: we have a pose data in the required format loaded already, we want to feed it
to the class and align the pose. This assumes that the pose data is already aligned
in the right format. If not, use the static method `convert_from_anipose`.

>>> data_path = Path("../data/anipose_220525_aJO_Fly001_001/pose-3d")
>>> f_path = data_path / "converted_pose_dict.pkl"
>>> with open(f_path, "rb") as f:
>>>     pose_data = pickle.load(f)
>>> align = AlignPose(
>>>     pose_data_dict=pose_data,
>>>     legs_list=["RF","LF"],
>>>     include_claw=False,
>>>     nmf_template=NMF_TEMPLATE,
>>> )
>>> aligned_pos = align.align_pose(export_path=data_path)

"""
from pathlib import Path
from typing import Dict, List, Union, Optional, Literal, Callable
import pickle
import logging

import numpy as np
from nptyping import NDArray

from seqikpy.data import PTS2ALIGN, NMF_TEMPLATE
from seqikpy.utils import save_file, calculate_nmf_size

logging.basicConfig(
    format=" %(asctime)s - %(levelname)s- %(message)s",
    handlers=[logging.StreamHandler()]
)


def _get_mean_quantile(vector, quantile_diff=0.05):
    """ Returns the mean of upper and lower quantiles. """
    return 0.5 * (
        np.quantile(vector, q=0.5 - quantile_diff) + np.quantile(vector, q=0.5 + quantile_diff)
    )


def _leg_length_model(nmf_size: dict, leg_name: str, claw_is_ee: bool):
    """ Sums up the segments of the model leg size."""
    if claw_is_ee:
        return nmf_size[leg_name]

    return nmf_size[leg_name] - nmf_size[f"{leg_name}_Tarsus"]


def _get_distance_btw_vecs(vector1, vector2):
    """ Calculates the distance between two vectors. """
    return np.linalg.norm(vector1 - vector2, axis=1)


def convert_from_anipose_to_dict(
    pose_3d: Dict[str, NDArray],
    pts2align: Dict[str, List[str]]
) -> Dict[str, NDArray]:
    """Loads anipose 3D pose data into a dictionary.
    See data.py for a mapping from keypoint name to segment name.

    Parameters
    ----------
    pose_3d : Dict[str, NDArray]
        3D pose data from anipose.
        It should have the following format:
        >>> pose_3d = {
            "{keypoint_name}_x" : NDArray[N_frames,],
            "{keypoint_name}_y" : NDArray[N_frames,],
            "{keypoint_name}_z" : NDArray[N_frames,],
        }
    pts2align : Dict[str, List[str]]
        Segment names and corresponding key point names to be aligned,
        check data.py for an example, by default None

    Returns
    -------
    Dict[str, NDArray]
        Pose data dictionary of the following format:
        >>> pose_data_dict = {
            "RF_leg": NDArray[N_frames,N_key_points,3],
            "LF_leg": NDArray[N_frames,N_key_points,3],
            "R_head": NDArray[N_frames,N_key_points,3],
            "L_head": NDArray[N_frames,N_key_points,3],
            "Neck": NDArray[N_frames,N_key_points,3],
        }
    """

    points_3d_dict = {}

    for segment in pts2align:
        segment_kps = pts2align[segment]
        temp_array = np.empty((pose_3d[f"{segment_kps[0]}_x"].shape[0], len(segment_kps), 3))
        for i, kp_name in enumerate(segment_kps):
            temp_array[:, i, 0] = pose_3d[f"{kp_name}_x"]
            temp_array[:, i, 1] = pose_3d[f"{kp_name}_y"]
            temp_array[:, i, 2] = pose_3d[f"{kp_name}_z"]

        points_3d_dict[segment] = temp_array.copy()

    return points_3d_dict


def convert_from_df3d_to_dict(
    pose_3d: Dict[str, NDArray],
    pts2align: Dict[str, List[str]]
) -> Dict[str, NDArray]:
    # TODO
    pass


class AlignPose:
    """Aligns the 3D poses.

    Parameters
    ----------
    pose_data_dict : Dict[str, NDArray]
        3D pose put in a dictionary that has the following structure defined
        by PTS2ALIGN (see data.py for more details):

        Example format:
            pose_data_dict = {
                "RF_leg": NDArray[N_frames,N_key_points,3],
                "LF_leg": NDArray[N_frames,N_key_points,3],
                "R_head": NDArray[N_frames,N_key_points,3],
                "L_head": NDArray[N_frames,N_key_points,3],
                "Neck": NDArray[N_frames,N_key_points,3],
            }
    legs_list : List[str]
        A list containing the leg names to operate on.
        Should follow the convention <R or L><F or M or H>
        e.g., ["RF", "LF", "RM", "LM", "RH", "LH"]
    include_claw : bool, optional
        If True, claw is included in the scaling process, by default False
    nmf_template : Dict[str, NDArray], optional
        A dictionary containing the positions of fly model body segments.
        Check ./data.py for the default dictionary, by default None
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"

    For the class usage examples, please refer to `example_alignment.py`
    """

    def __init__(
        self,
        pose_data_dict: Dict[str, NDArray],
        legs_list: List[str],
        include_claw: Optional[bool] = False,
        nmf_template: Optional[Dict[str, NDArray]] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        self.pose_data_dict = pose_data_dict
        self.include_claw = include_claw
        self.nmf_template = NMF_TEMPLATE if nmf_template is None else nmf_template
        # Calculate the size of the limbs from the template
        self.nmf_size = calculate_nmf_size(self.nmf_template, legs_list)

        # Get the logger of the module
        self.logger = logging.getLogger(self.__class__.__name__)
        numeric_level = getattr(logging, log_level.upper(), None)
        self.logger.setLevel(numeric_level)

    @classmethod
    def from_file_path(
        cls, main_dir: Union[str, Path],
        file_name: Optional[str] = "pose3d.*",
        convert_func: Optional[Callable] = None,
        pts2align: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        """
        Class method to load pose3d data and convert it into a proper
        structure.

        Parameters
        ----------
        main_dir : Union[str, Path]
            Path where the Anipose triangulation results are saved.
            By default, the result file is caled pose3d.h5
            However, if the name is different, <pose_result_path> should
            be modified accordingly.
            Example: "../Fly001/001_Beh/behData/pose-3d"
        file_name : str, optional
            File name, by default "pose3d.*"
        convert_func : Callable, optional
            Function to convert the loaded pose into the required format,
            set by the user if None, no conversion is performed,
            by default None
        pts2align : Dict[str, List[str]], optional
            Body part names and corresponding key points names to be aligned,
            check data.py for an example, by default None

        Returns
        -------
        AlignPose
            Instance of the AlignPose class.

        Raises
        ------
        FileNotFoundError
            If file with a name that contains `file_name` does not exist in `main_dir`.
        """
        paths = list(Path(main_dir).rglob(file_name))
        if len(paths) > 0:
            with open(paths[-1].as_posix(), "rb") as f:
                pose_3d = pickle.load(f)
        else:
            raise FileNotFoundError(f"{file_name} does not exits in {main_dir}")

        if convert_func is not None:
            pts2align = PTS2ALIGN if pts2align is None else pts2align
            converted_dict = convert_func(pose_3d, pts2align)
            return cls(converted_dict, **kwargs)

        return cls(pose_3d, **kwargs)

    def align_pose(
        self,
        export_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, NDArray]:
        """Aligns the leg and head key point positions.

        Parameters
        ----------
        export_path : Union[str, Path], optional
            The path where the aligned pose data will be saved, if specified.

        Returns
        -------
        Dict[str, NDArray]
            A dictionary containing the aligned pose data.
        """
        aligned_pose = {}
        for segment, segment_array in self.pose_data_dict.items():
            if "leg" in segment:
                aligned_pose[segment] = self.align_leg(
                    leg_array=segment_array, leg_name=segment[:2]
                )
            elif "head" in segment:
                aligned_pose[segment] = self.align_head(
                    head_array=segment_array, side=segment[0]
                )
            else:
                self.logger.debug("%s is not aligned", segment)
                continue
        # Take the neck as in the template as the other points are already aligned
        if "Neck" in self.nmf_template:
            aligned_pose["Neck"] = self.nmf_template["Neck"]

        if export_path is not None:
            export_full_path = export_path / "pose3d_aligned.pkl"
            save_file(out_fname=export_full_path, data=aligned_pose)
            self.logger.info("Aligned pose is saved at %s", export_path)

        return aligned_pose

    @property
    def thorax_mid_pts(self) -> NDArray:
        """ Gets the middle point of right and left wing hinges. """
        assert "Thorax" in self.pose_data_dict, "To align the head, you need to have a `Thorax` key point"
        thorax_pts = self.pose_data_dict["Thorax"]
        return 0.5 * (thorax_pts[:, 0, :] + thorax_pts[:, -1, :])

    @staticmethod
    def get_fixed_pos(points_3d: NDArray) -> NDArray:
        """ Gets the fixed pose of a steady key point determined by the quantiles. """
        fixed_pos = [
            _get_mean_quantile(points_3d[:, 0]),
            _get_mean_quantile(points_3d[:, 1]),
            _get_mean_quantile(points_3d[:, 2]),
        ]
        return np.array(fixed_pos)

    def get_mean_length(self, segment_array: NDArray, segment_is_leg: bool) -> Dict[str, float]:
        """ Computes the mean length of a body segment. """
        lengths = np.linalg.norm(np.diff(segment_array, axis=1), axis=2)

        if segment_is_leg:
            segments = ["coxa", "femur", "tibia", "tarsus"]
        else:
            segments = ["antenna"]

        length_mean = {}
        for i, s in enumerate(segments):
            length_mean[s] = _get_mean_quantile(lengths[:, i])

        return length_mean

    def find_scale_leg(self, leg_name: str, mean_length: Dict) -> float:
        """ Computes the ratio between the model size and the real fly size. """
        nmf_size = _leg_length_model(self.nmf_size, leg_name, self.include_claw)
        fly_leg_size = mean_length["coxa"] + mean_length["femur"] + mean_length["tibia"]
        fly_leg_size += mean_length["tarsus"] if self.include_claw else 0

        return nmf_size / fly_leg_size

    def find_stationary_indices(
        self, array: NDArray, threshold: Optional[float] = 5e-5
    ) -> NDArray:
        """ Find the indices in an array where the function value does not move significantly."""
        indices_stat = np.where((np.diff(np.diff(array)) < threshold))
        assert (
            indices_stat
        ), f"Threshold ({threshold}) is too low to find stationary points, please increase it."

        return indices_stat[0]

    def align_leg(
        self,
        leg_array: NDArray,
        leg_name: Literal["RF", "LF", "RM", "LM", "RH", "LH"]
    ) -> NDArray:
        """Scales and translates the leg key point locations based on the model size and configuration.

        This method takes a 3D array of leg key point positions and scales and translates them to align with a
        predefined model size and joint configuration. It accounts for the relative positions
        of key points and ensures that the scaled leg key points match the model size.

        Parameters
        ----------
        leg_array : NDArray
            A 3D array containing the leg key point positions.
        leg_name : str
            A string indicating the name of the leg (e.g., "RF", "LF", ...) for alignment.

        Returns
        -------
        NDArray
            A new 3D array containing the scaled and aligned leg key point positions.

        Notes
        -----
        * This method is used to align leg key point positions with a model of a fly"s leg.
        * It calculates the scale factor and multiplies the first 4 or 5 segments with the scale factor.
        """
        aligned_array = np.empty_like(leg_array)
        fixed_coxa = AlignPose.get_fixed_pos(leg_array[:, 0, :])

        mean_length = self.get_mean_length(leg_array, segment_is_leg=True)
        scale_factor = self.find_scale_leg(leg_name, mean_length)
        self.logger.info("Scale factor for %s leg: %s", leg_name, scale_factor)

        for i in range(0, 5):
            if i == 0:
                # Translate the 3D pose coxa to the template coxa position
                aligned_array[:, i, :] = (
                    np.zeros_like(leg_array[:, i, :])
                    + self.nmf_template[f"{leg_name}_Coxa"]
                )
            else:
                # Scale the length of the leg and
                # move the leg to the predefined coxa pos
                pos_aligned = (leg_array[:, i, :] - fixed_coxa).reshape(
                    -1, 3
                ) * scale_factor + self.nmf_template[f"{leg_name}_Coxa"]

                aligned_array[:, i, :] = pos_aligned

        return aligned_array.copy()

    def align_head(self, head_array: NDArray, side: str) -> NDArray:
        """Scales and translates the head key point locations based on the model size and configuration.

        This method takes a 3D array of head key point positions and scales and translates
        them to align with a predefined model size and configuration, such as a fly"s head.
        It accounts for the relative positions of key points and ensures that the scaled
        head key points match the model size.

        Parameters
        ----------
        head_array : NDArray
            A 3D array containing the head key point positions.
        side : str
            A string indicating the side of the head (e.g., "R" or "L") for alignment.

        Returns
        -------
        NDArray
            A new 3D array containing the scaled and aligned head key point positions.

        Raises
        ------
        KeyError
            If the "nmf_template" dictionary does not contain the required key names,
            "Antenna_mid_thorax" or "Antenna".

        Notes
        -----
        * This method is used to align head key point positions with a model of a fly"s head.
        * It calculates the scale factor and translations necessary to match the model"s
         head size and position.
        """
        antbase2thoraxmid_real = _get_distance_btw_vecs(
            head_array[:, 0, :], self.thorax_mid_pts
        )
        ant_size = self.get_mean_length(head_array, segment_is_leg=False)["antenna"]

        if self.nmf_size["Antenna_mid_thorax"] and self.nmf_size["Antenna"]:
            antbase2thoraxmid_tmp = self.nmf_size["Antenna_mid_thorax"]
            ant_tmp = self.nmf_size["Antenna"]
        else:
            raise KeyError(
                """Nmf template dictionary does not contain
                a key name <Antenna_mid_thorax> or <Antenna>
                Please check the dictionary you provided."""
            )

        stationary_indices = self.find_stationary_indices(antbase2thoraxmid_real)
        antenna_origin_fixed = AlignPose.get_fixed_pos(
            head_array[stationary_indices, 0, :]
        )
        scale_base_ant = antbase2thoraxmid_tmp / _get_mean_quantile(
            antbase2thoraxmid_real[stationary_indices]
        )
        scale_tip_ant = ant_tmp / _get_mean_quantile(ant_size)
        self.logger.info(
            "Scale factor antenna base %s: %s, ant itself: %s", side, scale_base_ant, scale_tip_ant
        )

        aligned_array = np.empty_like(head_array)
        aligned_array[:, 0, :] = (
            head_array[:, 0, :] - antenna_origin_fixed
        ) * scale_base_ant + self.nmf_template[f"{side}_Antenna_base"]
        aligned_array[:, 1, :] = (
            head_array[:, 1, :] - antenna_origin_fixed
        ) * scale_tip_ant + self.nmf_template[f"{side}_Antenna_base"]

        return aligned_array.copy()
