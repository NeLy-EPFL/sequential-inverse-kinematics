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
from typing import Dict, Union
from nptyping import NDArray

import numpy as np
from scipy.spatial.transform import Rotation as R

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
    """

    def __init__(
            self, aligned_pos: Dict[str, NDArray],
            nmf_template: Dict[str, NDArray],
    ) -> None:
        self.aligned_pos = aligned_pos
        self.nmf_template = nmf_template

    def compute_head_angles(self, export_path: Union[str, Path] = None) -> Dict[str, NDArray]:
        """Calculates all desired joint angles and saves them if export path is provided."""
        head_angles = {}

        head_angles['Angle_head_roll'] = self.compute_head_roll()
        head_angles['Angle_head_pitch'] = self.compute_head_pitch()
        head_angles['Angle_head_yaw'] = self.compute_head_yaw()
        for side in ['L', 'R']:
            head_angles[f'Angle_antenna_yaw_{side}'] = self.compute_antenna_yaw(
                side=side, head_roll=head_angles['Angle_head_roll'])
            head_angles[f'Angle_antenna_pitch_{side}'] = self.compute_antenna_pitch(
                side=side, head_roll=head_angles['Angle_head_roll'])

        if export_path is not None:
            export_path = Path(export_path) if not isinstance(export_path, Path) else export_path

            save_file(export_path / 'head_joint_angles.pkl', head_angles)
            logging.info('Head joint angles are saved at %s!', export_path)

        return head_angles

    def head_vector(self, side) -> NDArray:
        """ Vector ((N,3) array) from one antenna base to neck."""
        return self.aligned_pos["Neck"] - self.aligned_pos[f"{side}_head"][:, 0, :]

    @property
    def head_vector_mid(self) -> NDArray:
        """ Vector ((N,3) array) from antenna mid base to neck."""
        return (self.aligned_pos["R_head"][:, 0, :] + self.aligned_pos["L_head"]
                [:, 0, :]) * 0.5 - self.aligned_pos["Neck"]

    @property
    def head_vector_horizontal(self) -> NDArray:
        """ Vector ((N,3) array) from right antenna base to left antenna base."""
        return self.aligned_pos["L_head"][:, 0, :] - self.aligned_pos["R_head"][:, 0, :]

    def ant_vector(self, side: str) -> NDArray:
        """ Vector ((N,3) array) from antenna base to antenna edge."""
        return self.aligned_pos[f"{side}_head"][:, 1, :] - self.aligned_pos[f"{side}_head"][:, 0, :]

    @staticmethod
    def angle_between_segments(v1: NDArray, v2: NDArray) -> float:
        """ Calculates the angle between two vectors based on the cosinus formula.
        It reverses the sign of the angle if determinant of the matrix having
        two vectors and the rotation axis is negative.

        The returned angle is in radians.
        """
        # reshape to (N,3)
        v1 = v1.reshape(-1, 3)
        v2 = v2.reshape(-1, 3)

        v1_norm = v1 / np.linalg.norm(v1, axis=1)[:, None]
        v2_norm = v2 / np.linalg.norm(v2, axis=1)[:, None]

        return np.arccos(np.einsum("ij,ij->i", v1_norm, v2_norm))

    def compute_head_pitch(self) -> NDArray:
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

        angle = HeadInverseKinematics.angle_between_segments(
            v1=anteroposterior_axis, v2=head_vector
        )

        return angle + self.rest_head_pitch

    def compute_head_roll(self) -> NDArray:
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

        angle = HeadInverseKinematics.angle_between_segments(
            v1=horizontal_axis, v2=head_vector
        )

        return angle

    def compute_head_yaw(self) -> NDArray:
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

        angle = HeadInverseKinematics.angle_between_segments(
            v1=horizontal_axis, v2=head_vector
        )

        return angle

    def compute_antenna_pitch(self, side: str, head_roll: NDArray) -> NDArray:
        """ Calculates the head pitch angle (rad) from head vector
        projected onto sagittal plane to antenna vector (from base ant to edge).
        Furthermore, it subtracts the angle with the resting joint angle of the antenna pitch.

        Higher antenna pitch means antenna is lifted upward more.
        """
        side = side.upper()
        if side not in {"R", "L"}:
            raise ValueError("Side should be either R or L")

        v_derotate = np.vectorize(self.derotate_vector, signature='(m),(m,n)->(m,n)')

        antenna_vector = self.ant_vector(side).copy()
        assert antenna_vector.shape[1] == 3, f'Ant vector ({antenna_vector.shape}) does not have the right shape (N,3).'
        # Derotate the antenna vector to eliminate head roll based errors
        antenna_vector = v_derotate(head_roll, antenna_vector)
        antenna_vector[:, 1] = 0

        head_vector = self.head_vector(side).copy()
        assert head_vector.shape[1] == 3, f'Head vector ({head_vector.shape}) does not have the right shape (N,3).'

        head_vector = v_derotate(head_roll, head_vector)
        head_vector[:, 1] = 0

        angle = HeadInverseKinematics.angle_between_segments(
            v1=head_vector, v2=antenna_vector
        )

        return angle - self.rest_antenna_pitch

    def compute_antenna_yaw(self, side: str, head_roll: NDArray) -> NDArray:
        """ Calculates the antenna yaw angle (rad) from the lateral head vector
        projected onto transverse plane to antenna vector (from base ant to edge)
        projected, again, on the transverse plane.

        Higher antenna yaw means antennae get closer to the midline,
        deviating from the resting position.
        """
        side = side.upper()
        if side not in {"R", "L"}:
            raise ValueError("Side should be either R or L")

        v_derotate = np.vectorize(self.derotate_vector, signature='(m),(m,n)->(m,n)')

        antenna_vector = self.ant_vector(side).copy()
        antenna_vector = v_derotate(head_roll, antenna_vector)
        antenna_vector[:, 0] = 0

        head_vector = self.head_vector_horizontal.copy()
        head_vector = v_derotate(head_roll, head_vector)
        head_vector[:, 0] = 0

        angle = HeadInverseKinematics.angle_between_segments(
            v1=antenna_vector, v2=head_vector
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
        return HeadInverseKinematics.angle_between_segments(head_vector, antenna_vector)

    @property
    def rest_head_pitch(self) -> float:
        """ Head pitch angle at zero pose in the fly biomechanical model."""
        head_vector = (
            self.nmf_template["R_Antenna_base"] +
            self.nmf_template["L_Antenna_base"]) * 0.5 - self.nmf_template["Neck"]
        head_vector[1] = 0

        return HeadInverseKinematics.angle_between_segments(head_vector, X_AXIS)

    def derotate_vector(self, head_roll_angle: float, vector_to_derotate: NDArray) -> NDArray:
        """Rotates a vector by the amount of `head_roll_angle` along the x axis."""
        # counter-clockwise rotation in its coordinate system
        rotation = R.from_euler('x', -head_roll_angle, degrees=False)
        return rotation.apply(vector_to_derotate)

    def _get_plane(self, row: NDArray, n_row: int) -> NDArray:
        """ Construct an array by repeating row n_row many times."""
        return np.tile(row, (n_row, 1))
