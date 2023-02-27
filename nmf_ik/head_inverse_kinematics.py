# pylint: disable=C0301, C0103
""" Implementation of a class to calculate head inverse kinematics.

Example usage:
>>> import pickle
>>> from pathlib import Path
>>> from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
>>> from nmf_ik.data import NMF_TEMPLATE

>>> DATA_PATH = Path('../data/anipose/normal_case/pose-3d')
>>> f_path = DATA_PATH / "aligned_pose3d.h5"
>>> with open(f_path, "rb") as f:
>>>     data = pickle.load(f)

>>> class_hk = HeadInverseKinematics(
        aligned_pos = data,
        nmf_template=NMF_TEMPLATE,
        angles_to_calculate=[
            'Angle_head_roll',
            'Angle_head_pitch',
            'Angle_head_yaw',
            'Angle_antenna_pitch_L',
            'Angle_antenna_pitch_R',
            'Angle_antenna_yaw_L',
            'Angle_antenna_yaw_R'
            ]
    )
>>> joint_angles = class_hk.compute_head_angles(export_path = DATA_PATH)

IMPORTANT NOTES:
----------------
Note that the aligned pose, by default, should contain keys as: ['R_head', 'L_head', 'Neck', 'R_leg', 'L_leg'].
Each of these keys contain an array in the shape (N, key_points, 3) which corresponds to the frame number,
number of key points (5 for leg, 2 for head (antenna base and edge), 1 for neck), and x,y,z.
This configration is internally checked as well.
"""
from pathlib import Path
import logging
from typing import Dict, List, Union
from nptyping import NDArray

import numpy as np

from nmf_ik.utils import save_file

X_AXIS = np.array([1, 0, 0])
Y_AXIS = np.array([0, 1, 0])
Z_AXIS = np.array([0, 0, 1])


# Change the logging level here
logging.basicConfig(
    level=logging.INFO,
    format=' %(asctime)s - %(levelname)s- %(message)s')


class HeadInverseKinematics:
    """Calculates the head DOFs (3) and antennae DOFs (2)

    Parameters
    ----------
    aligned_pos : Dict[str, NDArray]
        Aligned pose dictionary.
        In principle, it should have body parts (R_head) as keys,
        and arrays (N,2 for the head) as values.
        Check the sample data for more detailed example.
    nmf_template : Dict[str, NDArray]
        Dictionary containing the positions of fly model body segments.
        Check ./data.py for the default dictionary.
    angles_to_calculate : List[str], optional
        DOFs to calculate, by default None
    """

    def __init__(
            self, aligned_pos: Dict[str, NDArray],
            nmf_template: Dict[str, NDArray],
            angles_to_calculate: List[str] = None,
    ) -> None:
        self.aligned_pos = aligned_pos
        self.nmf_template = nmf_template
        if angles_to_calculate is None:
            self.angles_to_calculate = [
                'Angle_head_roll',
                'Angle_head_pitch',
                'Angle_head_yaw',
                'Angle_antenna_pitch_L',
                'Angle_antenna_pitch_R',
                'Angle_antenna_yaw_L',
                'Angle_antenna_yaw_R']
        else:
            self.angles_to_calculate = angles_to_calculate

    def compute_head_angles(self, export_path: Union[str, Path] = None) -> Dict[str, NDArray]:
        """Calculates all desired joint angles and saves them if export path is provided."""
        head_angles = {}
        for angle in self.angles_to_calculate:
            if 'head_roll' in angle:
                head_angles[angle] = self.compute_head_roll()
            elif 'head_pitch' in angle:
                head_angles[angle] = self.compute_head_pitch()
            elif 'head_yaw' in angle:
                head_angles[angle] = self.compute_head_yaw()
            elif 'antenna_pitch' in angle:
                head_angles[angle] = self.compute_antenna_pitch(side=angle[-1])
            elif 'antenna_yaw' in angle:
                head_angles[angle] = self.compute_antenna_yaw(side=angle[-1])
            else:
                logging.warning('%s cannot be calculated! Check the name.', angle)
                continue

        if export_path is not None:

            export_path = Path(export_path) if not isinstance(export_path, Path) else export_path

            save_file(export_path / 'head_joint_angles.pkl', head_angles)
            logging.info('Head joint angles are saved at %s!', export_path)

        return head_angles

    def head_vector(self, side):
        """ Vector ((N,3) array) from one antenna base to neck."""
        return self.aligned_pos["Neck"] - self.aligned_pos[f"{side}_head"][:, 0, :]

    @property
    def head_vector_mid(self):
        """ Vector ((N,3) array) from antenna mid base to neck."""
        return (self.aligned_pos["R_head"][:, 0, :] + self.aligned_pos["L_head"]
                [:, 0, :]) * 0.5 - self.aligned_pos["Neck"]

    @property
    def head_vector_horizontal(self):
        """ Vector ((N,3) array) from right antenna base to left antenna base."""
        return self.aligned_pos["L_head"][:, 0, :] - self.aligned_pos["R_head"][:, 0, :]

    def ant_vector(self, side):
        """ Vector ((N,3) array) from antenna base to antenna edge."""
        return self.aligned_pos[f"{side}_head"][:, 1, :] - self.aligned_pos[f"{side}_head"][:, 0, :]

    @staticmethod
    def angle_between_segments(v1: NDArray, v2: NDArray, rot_axis: NDArray) -> float:
        """ Calculates the angle between two vectors based on the cosinus formula.
        It reverses the sign of the angle if determinant of the matrix having
        two vectors and the rotation axis is negative.

        The returned angle is in radians.
        """
        try:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except BaseException:
            cos_angle = 0

        angle = np.arccos(cos_angle)
        det = np.linalg.det([rot_axis, v1, v2])
        angle_corr = -angle if det < 0 else angle

        return angle_corr

    def compute_head_pitch(self):
        """ Calculates the head pitch angle (rad) from head mid vector
        projected onto sagittal plane to the anteroposterior plane.
        Furthermore, it sums the angle with the resting joint angle of the head pitch.

        Higher head pitch means head is lowered more.
        """
        head_vector = self.head_vector_mid.copy()
        assert head_vector.shape[1] == 3, f'Head vector ({head_vector.shape}) does not have the right shape (N,3).'
        # take the projection on the sagittal plane
        head_vector[:, 1] = 0
        anteroposterior_axis = self._get_plane([1, 0, 0], head_vector.shape[0])
        angle = np.zeros(head_vector.shape[0])
        for t in range(angle.shape[0]):
            angle[t] = HeadInverseKinematics.angle_between_segments(
                v1=anteroposterior_axis[t, :], v2=head_vector[t, :], rot_axis=Y_AXIS
            )

        return angle + self.rest_head_pitch

    def compute_head_roll(self):
        """ Calculates the head roll angle (rad) from horizontal axis
        to head horizontal vector projected onto transverse plane.

        Positive head roll -> rotation to the right in fly coords
        Negative head roll -> rotation to the left in the fly coords
        """
        head_vector = self.head_vector_horizontal.copy()
        assert head_vector.shape[1] == 3, f'Head vector ({head_vector.shape}) does not have the right shape (N,3).'
        # take the projection on the dorsoventral plane
        head_vector[:, 0] = 0
        horizontal_axis = self._get_plane([0, 1, 0], head_vector.shape[0])

        angle = np.zeros(head_vector.shape[0])
        for t in range(angle.shape[0]):
            angle[t] = HeadInverseKinematics.angle_between_segments(
                v1=horizontal_axis[t, :], v2=head_vector[t, :], rot_axis=X_AXIS
            )

        return angle

    def compute_head_yaw(self):
        """ Calculates the head yaw angle (rad) from horizontal axis
        to head horizontal vector projected onto frontal plane.

        Positive head yaw -> #TODO
        Negative head yaw -> #TODO
        """
        head_vector = self.head_vector_horizontal.copy()
        assert head_vector.shape[1] == 3, f'Head vector ({head_vector.shape}) does not have the right shape (N,3).'
        # take the projection on the horizontal plane
        head_vector[:, 2] = 0
        horizontal_axis = self._get_plane([0, 1, 0], head_vector.shape[0])

        angle = np.zeros(head_vector.shape[0])
        for t in range(angle.shape[0]):
            angle[t] = HeadInverseKinematics.angle_between_segments(
                v1=horizontal_axis[t, :], v2=head_vector[t, :], rot_axis=Z_AXIS
            )

        return angle

    def compute_antenna_pitch(self, side: str) -> NDArray:
        """ Calculates the head pitch angle (rad) from head vector
        projected onto sagittal plane to antenna vector (from base ant to edge).
        Furthermore, it subtracts the angle with the resting joint angle of the antenna pitch.

        Higher antenna pitch means antenna is lifted upward more.
        """
        side = side.upper()
        if side not in {"R", "L"}:
            raise ValueError("Side should be either R or L")

        antenna_vector = self.ant_vector(side).copy()
        assert antenna_vector.shape[1] == 3, f'Ant vector ({antenna_vector.shape}) does not have the right shape (N,3).'
        antenna_vector[:, 1] = 0

        head_vector = self.head_vector(side).copy()
        assert head_vector.shape[1] == 3, f'Head vector ({head_vector.shape}) does not have the right shape (N,3).'
        head_vector[:, 1] = 0

        angle = np.zeros(head_vector.shape[0])
        for t in range(angle.shape[0]):
            angle[t] = HeadInverseKinematics.angle_between_segments(
                v1=head_vector[t, :], v2=antenna_vector[t, :], rot_axis=Y_AXIS
            )

        return angle - self.rest_antenna_pitch

    def compute_antenna_yaw(self, side):
        """ Calculates the antenna yaw angle (rad) from the lateral head vector
        projected onto transverse plane to antenna vector (from base ant to edge)
        projected, again, on the transverse plane.

        Higher antenna yaw means antennae get closer to the midline,
        deviating from the resting position.
        """
        side = side.upper()
        if side not in {"R", "L"}:
            raise ValueError("Side should be either R or L")

        antenna_vector = self.ant_vector(side).copy()
        antenna_vector[:, 0] = 0

        head_vector = self.head_vector_horizontal.copy()
        head_vector[:, 0] = 0

        angle = np.zeros(head_vector.shape[0])
        for t in range(angle.shape[0]):
            angle[t] = HeadInverseKinematics.angle_between_segments(
                v1=antenna_vector[t, :], v2=head_vector[t, :], rot_axis=X_AXIS
            )
        if side == 'R':
            return np.pi - angle

        return angle

    @property
    def rest_antenna_pitch(self) -> float:
        """ Antenna pitch angle at zero pose in the fly biomechanical model."""
        head_vector = self.nmf_template["Neck"] - self.nmf_template["R_Antenna_base"]
        head_vector[1] = 0
        antenna_vector = self.nmf_template["R_Antenna_edge"] - self.nmf_template["R_Antenna_base"]
        antenna_vector[1] = 0
        return HeadInverseKinematics.angle_between_segments(head_vector, antenna_vector, Y_AXIS)

    @property
    def rest_head_pitch(self) -> float:
        """ Head pitch angle at zero pose in the fly biomechanical model."""
        head_vector = (
            self.nmf_template["R_Antenna_base"] + self.nmf_template["L_Antenna_base"]) * 0.5 - self.nmf_template["Neck"]
        head_vector[1] = 0

        return HeadInverseKinematics.angle_between_segments(head_vector, X_AXIS, Y_AXIS)

    def _get_plane(self, row, n_row):
        """ Construct an array by repeating row n_row many times."""
        return np.tile(row, (n_row, 1))
