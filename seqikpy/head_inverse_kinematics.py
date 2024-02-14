""" Implementation of a class to calculate head inverse kinematics.

Example usage:
>>> import pickle
>>> from pathlib import Path
>>> from seqikpy.head_inverse_kinematics import HeadInverseKinematics
>>> from seqikpy.data import NMF_TEMPLATE

>>> DATA_PATH = Path("../data/anipose/normal_case/pose-3d")
>>> f_path = DATA_PATH / "aligned_pose3d.h5"
>>> with open(f_path, "rb") as f:
>>>     data = pickle.load(f)

>>> class_hk = HeadInverseKinematics(
        aligned_pos = data,
        body_template=NMF_TEMPLATE,
    )
>>> joint_angles = class_hk.compute_head_angles(export_path = DATA_PATH)

IMPORTANT NOTES:
----------------
* The aligned pose must include keys: ["R_head", "L_head", "Neck"] for head angles calculation.

* Each key corresponds to an array shaped as (N, key_points, 3), where N is the frame count, key_points is the number of key points per part (2 for the head, including antenna base and edge, 1 for the neck), and 3 represents the x, y, z dimensions.

* The key points vary based on the 3D data. If antennal joint angles are not required, "L_head" and "R_head" may contain only a single key point. In such cases, set calculate_ant_angle to False in compute_head_angles. The head segments ("L_head", "R_head") can include any key point such as the head bristles for calculating head roll, pitch, and yaw. The examples utilize the antennae base for these calculations.
"""
from collections import namedtuple
from pathlib import Path
import logging
from typing import Dict, Union, Literal, Optional
from nptyping import NDArray

import numpy as np
from scipy.spatial.transform import Rotation as R

from seqikpy.utils import save_file

# Axes as a named tuple to ensure immutability
AxesTuple = namedtuple("AxesTuple", "X_AXIS Y_AXIS Z_AXIS")
Axes = AxesTuple(
    X_AXIS=np.array([1, 0, 0]),
    Y_AXIS=np.array([0, 1, 0]),
    Z_AXIS=np.array([0, 0, 1])
)


logging.basicConfig(
    format=" %(asctime)s - %(levelname)s- %(message)s",
    handlers=[logging.StreamHandler()]
)


class HeadInverseKinematics:
    """Calculates the head DOFs (3) and antennae DOFs (2)

    Parameters
    ----------
    aligned_pos : Dict[str, NDArray]
        Aligned pose dictionary.
        In principle, it should have body parts (R_head) as keys,
        and arrays (N,2 for the head) as values.
        Check the sample data for more detailed example.
    body_template : Dict[str, NDArray]
        Dictionary containing the positions of fly model body segments.
        Check data.py for the default dictionary.
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"
    """

    def __init__(
        self, aligned_pos: Dict[str, NDArray],
        body_template: Dict[str, NDArray],
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        self.aligned_pos = aligned_pos
        self.body_template = body_template

        # Check self.aligned_pos keys
        if not all(key in self.aligned_pos for key in ["R_head", "L_head", "Neck"]):
            raise ValueError(
                """self.aligned_pos must have R_head, L_head, Neck as keys,
                at least one of them is missing in the current data"""
            )

        # Set the constants values to avoid repetitive computations
        self.head_vector_mid = self.get_head_vector_mid()
        self.head_vector_horizontal = self.get_head_vector_horizontal()
        assert self.head_vector_mid.shape[1] == 3 and self.head_vector_horizontal.shape[1] == 3, f"""
                One of head vectors
                (mid: {self.head_vector_mid.shape}, horizontal: {self.head_vector_horizontal.shape})
                does not have the right shape (N,3).
                """

        # Biomechanical model `zero pose` joint angles
        self.rest_head_pitch = self.get_rest_head_pitch()
        self.rest_antenna_pitch = self.get_rest_antenna_pitch()

        # Get the logger of the module
        self.logger = logging.getLogger(self.__class__.__name__)
        numeric_level = getattr(logging, log_level.upper(), None)
        self.logger.setLevel(numeric_level)

    def compute_head_angles(
        self,
        export_path: Union[str, Path] = None,
        compute_ant_angles: Optional[bool] = True,
    ) -> Dict[str, NDArray]:
        """Calculates the head & antennal joint angles

        Parameters
        ----------
        export_path : Union[str, Path], optional
            Export path of the joint angles, by default None
        compute_ant_angles : Optional[bool], optional
            If True, computes the head roll,pitch and yaw, by default True

        Returns
        -------
        Dict[str, NDArray]
            Dicitonary containing the head joint angles, saved
            in export path if provided
        """
        head_angles = {}

        head_angles["Angle_head_roll"] = self.compute_head_roll()
        head_angles["Angle_head_pitch"] = self.compute_head_pitch()
        head_angles["Angle_head_yaw"] = self.compute_head_yaw()

        if compute_ant_angles:
            for side in ["L", "R"]:
                head_angles[f"Angle_antenna_yaw_{side}"] = self.compute_antenna_yaw(
                    side=side, head_roll=head_angles["Angle_head_roll"]
                )
                head_angles[f"Angle_antenna_pitch_{side}"] = self.compute_antenna_pitch(
                    side=side, head_roll=head_angles["Angle_head_roll"]
                )

        if export_path is not None:
            save_file(Path(export_path) / "head_joint_angles.pkl", head_angles)
            self.logger.info("Head joint angles are saved at %s!", export_path)

        return head_angles

    def get_head_vector(self, side: Literal["R", "L"]) -> NDArray:
        """Vector ((N,3) array) from <side> antenna base (or any head key point) to neck."""
        return self.aligned_pos["Neck"][:,0,:] - self.aligned_pos[f"{side}_head"][:, 0, :]

    def get_head_vector_mid(self) -> NDArray:
        """ Vector ((N,3) array) from mid antenna base (or any head key point) to neck."""
        return (self.aligned_pos["R_head"][:, 0, :] + self.aligned_pos["L_head"]
                [:, 0, :]) * 0.5 - self.aligned_pos["Neck"][:,0,:]

    def get_head_vector_horizontal(self) -> NDArray:
        """ Vector ((N,3) array) from right antenna base (or any head key point)
        to left antenna base (or any head key point).
        """
        return self.aligned_pos["L_head"][:, 0, :] - self.aligned_pos["R_head"][:, 0, :]

    def get_ant_vector(self, side: Literal["R", "L"]) -> NDArray:
        """ Vector ((N,3) array) from antenna base to antenna edge."""
        return self.aligned_pos[f"{side}_head"][:, 1, :] - self.aligned_pos[f"{side}_head"][:, 0, :]

    @staticmethod
    def angle_between_segments(v1: NDArray, v2: NDArray, rot_axis: NDArray) -> float:
        """ Calculates the angle between two vectors based on the cos formula.
        It reverses the sign of the angle if determinant of the matrix having
        two vectors and the rotation axis is negative.

        The returned angle is in radians.
        """
        # reshape to (N,3)
        v1 = v1.reshape(-1, 3)
        v2 = v2.reshape(-1, 3)

        v1_norm = v1 / np.linalg.norm(v1, axis=1)[:, None]
        v2_norm = v2 / np.linalg.norm(v2, axis=1)[:, None]

        mask = np.empty((v1_norm.shape[0],))

        for row in range(v1_norm.shape[0]):
            mask[row] = 1 if np.linalg.det([rot_axis, v1[row, :], v2[row, :]]) > 0 else -1

        return np.arccos(np.einsum("ij,ij->i", v1_norm, v2_norm)) * mask

    def compute_head_pitch(self) -> NDArray:
        """ Calculates the head pitch angle (rad) from head mid vector
        projected onto sagittal plane to the anteroposterior plane.
        Furthermore, it sums the angle with the resting joint angle of the head pitch.

        Higher head pitch means head is lowered more.
        """
        head_vector = self.head_vector_mid.copy()

        # take the projection on the sagittal plane
        head_vector[:, 1] = 0
        anteroposterior_axis = self.get_plane(Axes.X_AXIS, head_vector.shape[0])

        angle = HeadInverseKinematics.angle_between_segments(
            v1=anteroposterior_axis, v2=head_vector, rot_axis=Axes.Y_AXIS
        )

        return angle + self.rest_head_pitch

    def compute_head_roll(self) -> NDArray:
        """ Calculates the head roll angle (rad) from horizontal axis
        to head horizontal vector projected onto transverse plane.

        Positive head roll -> rotation to the right in fly coords
        Negative head roll -> rotation to the left in the fly coords
        """
        head_vector = self.head_vector_horizontal.copy()

        # take the projection on the dorsoventral plane
        head_vector[:, 0] = 0
        horizontal_axis = self.get_plane(Axes.Y_AXIS, head_vector.shape[0])

        angle = HeadInverseKinematics.angle_between_segments(
            v1=horizontal_axis, v2=head_vector, rot_axis=Axes.X_AXIS
        )

        return angle

    def compute_head_yaw(self) -> NDArray:
        """ Calculates the head yaw angle (rad) from horizontal axis
        to head horizontal vector projected onto frontal plane.

        Positive head yaw -> head yaw to the left
        Negative head yaw -> head yaw to the right
        """
        head_vector = self.head_vector_horizontal.copy()

        # take the projection on the horizontal plane
        head_vector[:, 2] = 0
        horizontal_axis = self.get_plane(Axes.Y_AXIS, head_vector.shape[0])

        angle = HeadInverseKinematics.angle_between_segments(
            v1=horizontal_axis, v2=head_vector, rot_axis=Axes.Z_AXIS
        )

        return angle

    def compute_antenna_pitch(self, side: Literal["R", "L"], head_roll: NDArray) -> NDArray:
        """ Calculates the head pitch angle (rad) from head vector
        projected onto sagittal plane to antenna vector (from base ant to edge).
        Furthermore, it subtracts the angle with the resting joint angle of the antenna pitch.

        Higher antenna pitch means antenna is lifted upward more.
        """
        side = side.upper()
        if side not in {"R", "L"}:
            raise ValueError("Side should be either R or L")

        v_derotate = np.vectorize(self.derotate_vector, signature="(m),(m,n)->(m,n)")

        antenna_vector = self.get_ant_vector(side).copy()
        assert antenna_vector.shape[1] == 3, f"""
            Ant vector ({antenna_vector.shape}) does not have the right shape (N,3).
        """
        # Derotate the antenna vector to eliminate the singularity errors
        # coming from head roll
        antenna_vector = v_derotate(head_roll, antenna_vector)
        antenna_vector[:, 1] = 0

        head_vector = self.get_head_vector(side).copy()
        assert head_vector.shape[1] == 3, f"""
            Head vector ({head_vector.shape}) does not have the right shape (N,3).
        """

        head_vector = v_derotate(head_roll, head_vector)
        head_vector[:, 1] = 0

        angle = HeadInverseKinematics.angle_between_segments(
            v1=head_vector, v2=antenna_vector, rot_axis=Axes.Y_AXIS
        )

        return angle - self.rest_antenna_pitch

    def compute_antenna_yaw(self, side: Literal["R", "L"], head_roll: NDArray) -> NDArray:
        """ Calculates the antenna yaw angle (rad) from the lateral head vector
        projected onto transverse plane to antenna vector (from base ant to edge)
        projected, again, on the transverse plane.

        Higher antenna yaw means antennae get closer to the midline,
        deviating from the resting position.
        """
        side = side.upper()
        if side not in {"R", "L"}:
            raise ValueError("Side should be either R or L")

        v_derotate = np.vectorize(self.derotate_vector, signature="(m),(m,n)->(m,n)")

        antenna_vector = self.get_ant_vector(side).copy()
        antenna_vector = v_derotate(head_roll, antenna_vector)
        antenna_vector[:, 0] = 0

        head_vector = self.head_vector_horizontal.copy()
        head_vector = v_derotate(head_roll, head_vector)
        head_vector[:, 0] = 0

        angle = HeadInverseKinematics.angle_between_segments(
            v1=antenna_vector, v2=head_vector, rot_axis=Axes.X_AXIS
        )

        if side == "R":
            return np.pi - angle

        return angle

    def get_rest_antenna_pitch(self) -> float:
        """ Antenna pitch angle at zero pose in the fly biomechanical model."""
        head_vector = self.body_template["Neck"] - self.body_template["R_Antenna_base"]
        # project onto x-z plane
        head_vector[1] = 0
        # We consider only one side as the model is symmetrical
        antenna_vector = self.body_template["R_Antenna_edge"] - self.body_template["R_Antenna_base"]
        # project onto x-z plane
        antenna_vector[1] = 0
        return HeadInverseKinematics.angle_between_segments(head_vector, antenna_vector, Axes.Y_AXIS)

    def get_rest_head_pitch(self) -> float:
        """ Head pitch angle at zero pose in the fly biomechanical model."""
        head_vector = (
            self.body_template["R_Antenna_base"] +
            self.body_template["L_Antenna_base"]
        ) * 0.5 - self.body_template["Neck"]
        # project onto x-z plane
        head_vector[1] = 0

        return HeadInverseKinematics.angle_between_segments(head_vector, Axes.X_AXIS, Axes.Y_AXIS)

    def derotate_vector(self, head_roll_angle: float, vector_to_derotate: NDArray) -> NDArray:
        """Rotates a vector by the inverse amount of `head_roll_angle` along the x axis."""
        # counter-clockwise rotation in its coordinate system
        rotation = R.from_euler("x", -head_roll_angle, degrees=False)
        return rotation.apply(vector_to_derotate)

    def get_plane(self, row: NDArray, n_row: int) -> NDArray:
        """ Construct an array by repeating row n_row many times."""
        return np.tile(row, (n_row, 1))
