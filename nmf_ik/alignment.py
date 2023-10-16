"""
Code for aligning 3d pose to nmf template.

Example usage:

>>> from pathlib import Path
>>> from nmf_ik.alignment import AlignPose

>>> DATA_PATH = Path('../data/anipose/normal_case/pose-3d')

>>> align = AlignPose.from_file_path(DATA_PATH, convert_dict=True)
>>> aligned_pose = align.align_pose(export_path=DATA_PATH)

NOTES:
------
Despite being extendable, this method is written with a special focus on forelegs and head.
"""
from pathlib import Path
from typing import Dict, List, Union, Optional
import pickle
import logging
from nptyping import NDArray
import numpy as np

from nmf_ik.data import PTS2ALIGN, NMF_TEMPLATE
from nmf_ik.utils import save_file, calculate_nmf_size

# Change the logging level here
logging.basicConfig(
    format=" %(asctime)s - %(levelname)s- %(message)s"
)
# Get the logger of the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_mean_quantile(vector, quantile_diff=0.05):
    """ Returns the mean of upper and lower quantiles. """
    return 0.5 * (
        np.quantile(vector, q=0.5 - quantile_diff) + np.quantile(vector, q=0.5 + quantile_diff)
    )


def _leg_length_model(nmf_size: dict, leg_name: str, claw_is_ee: bool):
    """ Sums up the segments of the model leg size."""
    if claw_is_ee:
        return nmf_size[leg_name]

    return nmf_size[leg_name] - nmf_size[f'{leg_name}_Tarsus']


def _get_distance_btw_vecs(vector1, vector2):
    """ Calculates the distance between two vectors. """
    return np.linalg.norm(vector1 - vector2, axis=1)


class AlignPose:
    """Aligns the 3D poses.

    Parameters
    ----------
    pose_data_dict : Dict[str, NDArray]
        3D pose put in a dictionary that has the following structure defined
        by PTS2ALIGN (see data.py for more details):

        pose_data_dict = {
            'RF_leg': NDArray[N_frames,N_key_points,3],
            'LF_leg': NDArray[N_frames,N_key_points,3],
            'R_head': NDArray[N_frames,N_key_points,3],
            'L_head': NDArray[N_frames,N_key_points,3],
            'Neck': NDArray[N_frames,N_key_points,3],
        }
    include_claw : bool, optional
        True if claw is included in the scaling process, by default False
    nmf_template : Dict[str, NDArray], optional
        Dictionary containing the positions of fly model body segments.
        Check ./data.py for the default dictionary, by default None

    For the class usage examples, please refer to `example_alignment.py`
    """

    def __init__(
        self,
        pose_data_dict: Dict[str, NDArray],
        include_claw: Optional[bool] = False,
        nmf_template: Optional[Dict[str, NDArray]] = None,
    ):
        self.pose_data_dict = pose_data_dict
        self.include_claw = include_claw
        self.nmf_template = NMF_TEMPLATE if nmf_template is None else nmf_template
        self.nmf_size = calculate_nmf_size(self.nmf_template)

    @classmethod
    def from_file_path(
        cls, main_dir: Union[str, Path],
        file_name: Optional[str] = "pose3d.*",
        convert_dict: Optional[bool] = True,
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
            Example: '../Fly001/001_Beh/behData/pose-3d'
        file_name : str, optional
            Key words for the file name, by default "pose3d.*"
        convert_dict : bool, optional
            If true converts the loaded file into a dictionary, by default True
        pts2align : Dict[str, List[str]], optional
            Region names and corresponding key points names to be aligned,
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
        main_dir = Path(main_dir) if not isinstance(main_dir, Path) else main_dir
        paths = list(main_dir.rglob(file_name))
        if len(paths) > 0:
            with open(paths[-1].as_posix(), "rb") as f:
                pose_3d = pickle.load(f)
        else:
            raise FileNotFoundError(f"pose3d does not exits in {main_dir}")

        if convert_dict:
            pts2align = PTS2ALIGN if pts2align is None else pts2align
            return cls(AlignPose.convert_from_anipose_to_dict(pose_3d, pts2align), **kwargs)

        return cls(pose_3d, **kwargs)

    @staticmethod
    def convert_from_anipose_to_dict(
            pose_3d: NDArray, pts2align: Dict[str, List[str]]) -> Dict[str, NDArray]:
        """ Loads anipose data into a dictionary.
            Segment names are under data.py
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

    def align_pose(self, export_path: Optional[Union[str, Path]] = None) -> Dict[str, NDArray]:
        """ Aligns the leg and head key point positions.
            Saves the results of save_pose_file is True.
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
                continue
        # Take the neck as in the template as the other points are already aligned
        if 'Neck' in self.nmf_template:
            aligned_pose["Neck"] = self.nmf_template["Neck"]

        if export_path is not None:
            export_full_path = export_path / "pose3d_aligned.pkl"
            save_file(out_fname=export_full_path, data=aligned_pose)
            logger.info("Aligned pose is saved at %s", export_path)

        return aligned_pose

    @property
    def thorax_mid_pts(self) -> NDArray:
        """ Middle point of right and left wing hinges. """
        thorax_pts = self.pose_data_dict["Thorax"]
        return 0.5 * (thorax_pts[:, 0, :] + thorax_pts[:, -1, :])

    @staticmethod
    def get_fixed_pos(points_3d: NDArray) -> NDArray:
        """ Fixed pose of a steady key point determined by the quantiles. """
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
        self, array1: NDArray, threshold: Optional[float] = 5e-5
    ) -> NDArray:
        """ Find the indices in an array where the function value does not move much."""
        indices_stat = np.where((np.diff(np.diff(array1)) < threshold))
        assert (
            indices_stat
        ), f"Threshold ({threshold}) is too low to find stationary points, please increase it."

        return indices_stat[0]

    def align_leg(self, leg_array: NDArray, leg_name: str) -> NDArray:
        """ Scales and translated the leg key point locations based on the model size/config."""
        aligned_array = np.empty_like(leg_array)
        fixed_coxa = AlignPose.get_fixed_pos(leg_array[:, 0, :])

        mean_length = self.get_mean_length(leg_array, segment_is_leg=True)
        scale_factor = self.find_scale_leg(leg_name, mean_length)
        logger.info("Scale factor for %s leg: %s", leg_name, scale_factor)

        for i in range(0, 5):
            if i == 0:
                aligned_array[:, i, :] = (
                    np.zeros_like(leg_array[:, i, :])
                    + self.nmf_template[f"{leg_name}_Coxa"]
                )
            else:
                pos_aligned = (leg_array[:, i, :] - fixed_coxa).reshape(
                    -1, 3
                ) * scale_factor + self.nmf_template[f"{leg_name}_Coxa"]

                aligned_array[:, i, :] = pos_aligned

        return aligned_array.copy()

    def align_head(self, head_array: NDArray, side: str) -> NDArray:
        """ Scales and translated the head key point locations based on the model size/config."""
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
        logger.info(
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
