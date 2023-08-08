# pylint: disable = C0103, W0311
""" Script to calculate inverse kinematics from aligned pose based on ikpy.

Example usage:
>>> import pickle
>>> from pathlib import Path
>>> from nmf_ik.leg_inverse_kinematics import LegInverseKinematics
>>> from nmf_ik.data import BOUNDS, INITIAL_ANGLES, NMF_TEMPLATE


>>> DATA_PATH = Path('../data/anipose/normal_case/pose-3d')
>>> f_path = DATA_PATH / "aligned_pose3d.h5"
>>> with open(f_path, "rb") as f:
>>>     data = pickle.load(f)

>>> seq_ik = LegInverseKinematics(
    aligned_pos=aligned_pos,
    nmf_template=NMF_TEMPLATE,
    nmf_size=NMF_SIZE,
    bounds=BOUNDS,
    initial_angles=INITIAL_ANGLES
)
>>> leg_joint_angles, forward_kinematics = seq_ik.run_ik_and_fk(export_path=DATA_PATH)

Note that there are two kinds of kinematic_chain classes for now.
One with the original configuration where there is no trochanter, the other one is
with the offset introduced by trochanter. If you want to manually change the inherited
class, just change the import in this file.

"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Union, List
import logging
from nptyping import NDArray
import warnings

import numpy as np

from tqdm import trange

from ikpy.chain import Chain

from nmf_ik.utils import save_file
from nmf_ik.kinematic_chain import KinematicChain
from nmf_ik.data import BOUNDS, INITIAL_ANGLES

# Suppress warnings
warnings.filterwarnings("ignore")

# Change the logging level here
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s- %(message)s")


class LegInverseKinematics(KinematicChain):
    """LegInverseKinematics class.

    Parameters
    ----------
    aligned_pos : Dict[str, NDArray]
        Aligned pose from the AlignPose class.
        If not provide the pkl using the classmethod load_from_pkl.
    nmf_template : Dict[str, NDArray], optional
        Dictionary that contains the location of body segments.
    bounds : Dict[str, NDArray], optional
        Dictionary that contains the bounds of joint degrees-of-freedom.
    initial_angles : Dict[str, NDArray], optional
        Initial angles of DOFs, by default None
    """

    def __init__(
            self, aligned_pos: Dict[str, NDArray],
            nmf_size: Dict[str, float] = None,
            bounds: Dict[str, NDArray] = None,
            initial_angles: Dict[str, NDArray] = None,
    ):
        super().__init__(nmf_size, bounds)
        self.aligned_pos = aligned_pos
        # TODO: Think about the below, for now commented
        # self.mean_segment_size = get_mean_length_of_segments(self.aligned_pos)
        # self.leg_length = get_leg_length(self.mean_segment_size)
        self.bounds_dof = BOUNDS if bounds is None else bounds
        self.initial_angles = INITIAL_ANGLES if initial_angles is None else initial_angles
        self.joint_angles_empty = {}

    @classmethod
    def load_data_from_pkl(cls, data_path: Union[str, Path], file_name: str = "*aligned.pkl", **kwargs):
        """Loads aligned pose dictionary."""

        data_path = Path(data_path) if not isinstance(data_path, Path) else data_path

        with open(list(data_path.glob(file_name))[0].as_posix(), "rb") as f:
            aligned_pos = pickle.load(f)
        return cls(aligned_pos, **kwargs)

    @staticmethod
    def calculate_ik(
        kinematic_chain: Chain, target_pos: NDArray, initial_angles: NDArray = None
    ) -> NDArray:
        """Calculates the joint angles in the leg chain."""
        # don't take the last and first ones ¨
        return kinematic_chain.inverse_kinematics(
            target_position=target_pos, initial_position=initial_angles
        )

    @staticmethod
    def calculate_fk(kinematic_chain: Chain, joint_angles: NDArray) -> NDArray:
        """Calculates the forward kinematics from the joint dof angles."""
        fk = kinematic_chain.forward_kinematics(
            joint_angles,
            full_kinematics=True
        )
        end_effector_positions = np.zeros((len(kinematic_chain.links), 3))
        for link in range(len(kinematic_chain.links)):
            end_effector_positions[link, :] = fk[link][:3, 3]
        return end_effector_positions

    @staticmethod
    def get_scale_factor(vector: NDArray, length: float) -> float:
        """ Gets scale the ratio between two vectors. """
        vector_diff = np.linalg.norm(np.diff(vector, axis=0), axis=1)
        norm_sum = np.sum(vector_diff)
        return length / norm_sum

    def calculate_ik_for_trial(
        self,
        end_effector_pos: NDArray,
        origin: NDArray,
        initial_angles: NDArray,
        stage: int = None,
        segment_name: str = None,
        kinematic_chain: Chain = None,
        femur_orientation: float = 0
    ) -> NDArray:
        """For a given trial pose data, calculates the inverse kinematics
        by using a sequential inverse kinematics method.

        Parameters
        ----------
        end_effector_pos : NDArray
            3D array containing the position of the end eff pos
        origin : NDArray
            Origin of the kinematic chain, i.e. Thorax Coxa
        initial_angles : NDArray
            Initial angles for the optimization
        stage : int, optional
            Stage of the sequential kinematic chain, by default None
        segment_name : str, optional
            Leg side, i.e. LF or RF, by default None
        kinematic_chain : Chain, optional
            Kinematic chain of the respective leg, by default None

        Returns
        -------
        NDArray
            Forward kinematics. Returns a non-empty array at the 4th stage.
        """
        # TODO: check scaling
        frames_no = end_effector_pos.shape[0]
        end_effector_pos_diff = end_effector_pos - origin

        # Initialize the arrays
        joint_angles = np.empty((frames_no, len(initial_angles)))
        forward_kinematics = np.empty((frames_no, len(initial_angles), 3))

        femur_orientation = 1 * femur_orientation if 'RF' in segment_name else -1 * femur_orientation

        if origin.size == 3:
            origin = np.tile(origin, (frames_no, 1))

        if stage == 1:
            kinematic_chain = self.create_leg_chain(
                stage=stage, leg_name=segment_name,
                femur_orientation=femur_orientation
            )

        # Start the IK process
        for t in trange(frames_no, disable=False):
            if stage in [2, 3, 4]:
                kinematic_chain = self.create_leg_chain(
                    stage=stage, leg_name=segment_name,
                    angles=self.joint_angles_empty, t=t,
                    femur_orientation=femur_orientation
                )

            initial_angles = initial_angles if t == 0 else joint_angles[t - 1, :]
            joint_angles[t, :] = LegInverseKinematics.calculate_ik(
                kinematic_chain, end_effector_pos_diff[t, :], initial_angles
            )

            if stage == 4:
                forward_kinematics[t, :] = (
                    LegInverseKinematics.calculate_fk(kinematic_chain, joint_angles[t, :]) + origin[t, :]
                )

        if stage == 1:
            self.joint_angles_empty[f'Angle_{segment_name}_ThC_yaw'] = joint_angles[:, 1]
            self.joint_angles_empty[f'Angle_{segment_name}_ThC_pitch'] = joint_angles[:, 2]
            logging.debug('Stage 1 is completed!')
        elif stage == 2:
            self.joint_angles_empty[f'Angle_{segment_name}_ThC_roll'] = joint_angles[:, 1]
            self.joint_angles_empty[f'Angle_{segment_name}_CTr_pitch'] = joint_angles[:, -2]
            logging.debug('Stage 2 is completed!')
        elif stage == 3:
            self.joint_angles_empty[f'Angle_{segment_name}_CTr_roll'] = joint_angles[:, -3]
            self.joint_angles_empty[f'Angle_{segment_name}_FTi_pitch'] = joint_angles[:, -2]
            logging.debug('Stage 3 is completed!')
        elif stage == 4:
            self.joint_angles_empty[f'Angle_{segment_name}_TiTa_pitch'] = joint_angles[:, -2]
            logging.debug('Stage 4 is completed!')

        return forward_kinematics

    def run_ik_and_fk(
        self, export_path: Union[Path, str] = None, stages: List[int] = [1,2,3,4],  femur_orientation: float = 0
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """ Runs inverse and forward kinematics for leg joints.

        Parameters
        ----------
        export_path : Union[Path, str], optional
            Path where the results will be saved, by default None

        Returns
        Tuple[Dict[str,NDArray], Dict[str,NDArray]]
            Two dictionaries containing joint angles and forward
            kinematics, respectively.
        """
        assert 1 in stages, "stages should start with 1 and strictly be incrementary"
        forward_kinematics_dict = {}
        joint_angles_dict = {}

        logging.info("Calculating joint angles and forward kinematics...")
        for segment, segment_array in self.aligned_pos.items():
            if "leg" in segment.lower():
                leg_name = segment[:2]
                origin = segment_array[:, 0, :]

                for stage in stages:
                    end_effector_pos = segment_array[:, stage, :]
                    init_ang = self.initial_angles[segment[:2]][f'stage_{stage}']

                    forward_kinematics_dict[segment] = self.calculate_ik_for_trial(
                        end_effector_pos=end_effector_pos,
                        origin=origin,
                        initial_angles=init_ang,
                        stage=stage,
                        segment_name=leg_name,
                        kinematic_chain=None,
                        femur_orientation=femur_orientation,
                    )

            else:
                continue

        joint_angles_dict = self.joint_angles_empty.copy()

        export_path = Path(export_path) if not isinstance(export_path, Path) else export_path

        if export_path is not None:
            save_file(export_path / "forward_kinematics.pkl", forward_kinematics_dict)
            save_file(export_path / "leg_joint_angles.pkl", joint_angles_dict)
            # save_file(export_path.replace('aligned_pose','joint_angles'), joint_angles_dict)
            logging.info("Files have been saved at %s", export_path)

        return joint_angles_dict, forward_kinematics_dict
