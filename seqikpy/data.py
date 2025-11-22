"""Data, constants, and paths."""

import numpy as np
import warnings
from dataclasses import dataclass
from copy import deepcopy as _deepcopy


@dataclass
class BodyConfig:
    # Size of the template body segments
    segment_sizes: dict[str, float]

    # Key points to align, to be provided in the alignment.Align class
    points_to_align: dict[str, list[str]]

    # Key points that are used in alignment
    skeleton: list[str]

    # Pose of each body landmark in the NeuroMechFly v0.0.6 model
    # Note that each leg segment represents the joint in the proximal part
    # For example, RF_Coxa means Thorax-Coxa joint
    template: dict[str, np.ndarray]

    # Initial joint angles for each leg and stage
    initial_angles_rad: dict[str, dict[str, np.ndarray]] | None = None

    # Lower bound of a DOF should be strictly lower than the initial angle.
    # Upper bound of a DOF should be strictly bigger than the initial angle.
    dof_bounds_rad: dict[str, tuple[float, float]] | None = None

    def get_copy_of_initial_angles_in_deg(self):
        initial_angles_deg = {}
        for leg_name, stages in self.initial_angles_rad.items():
            initial_angles_deg[leg_name] = {}
            for stage, angles_rad in stages.items():
                initial_angles_deg[leg_name][stage] = np.rad2deg(angles_rad)
        return initial_angles_deg

    def get_copy_of_dof_bounds_in_deg(self):
        if self.dof_bounds_rad is None:
            return None
        dof_bounds_deg = {}
        for dof, (lower_rad, upper_rad) in self.dof_bounds_rad.items():
            dof_bounds_deg[dof] = (np.rad2deg(lower_rad), np.rad2deg(upper_rad))
        return dof_bounds_deg

    def set_initial_angles_in_deg(
        self, initial_angles_deg: dict[str, dict[str, np.ndarray]]
    ):
        initial_angles_rad = {}
        for leg_name, stages in initial_angles_deg.items():
            initial_angles_rad[leg_name] = {}
            for stage, angles_deg in stages.items():
                initial_angles_rad[leg_name][stage] = np.deg2rad(angles_deg)
        self.initial_angles_rad = initial_angles_rad

    def set_dof_bounds_in_deg(self, dof_bounds_deg: dict[str, tuple[float, float]]):
        dof_bounds_rad = {}
        for dof, (lower_deg, upper_deg) in dof_bounds_deg.items():
            dof_bounds_rad[dof] = (np.deg2rad(lower_deg), np.deg2rad(upper_deg))
        self.dof_bounds_rad = dof_bounds_rad

    def deepcopy(self):
        return _deepcopy(self)


# Define default BodyConfig for NeuroMechFly
_NMF_INITIAL_ANGLES_RAD = {
    "RF": {
        # Base ThC yaw pitch CTr pitch
        "stage_1": np.array([0.0, 0.45, -0.07, -2.14]),
        # Base ThC yaw pitch roll CTr pitch CTr roll
        "stage_2": np.array([0.0, 0.45, -0.07, -0.32, -2.14, 1.4]),
        # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch
        "stage_3": np.array([0.0, 0.45, -0.07, -0.32, -2.14, -1.25, 1.48, 0.0]),
        # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch TiTa pitch
        "stage_4": np.array([0.0, 0.45, -0.07, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
    },
    # Same order for the contralateral leg
    "LF": {
        "stage_1": np.array([0.0, -0.45, -0.07, -2.14]),
        "stage_2": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
}

_NMF_BOUNDS_DEG = {
    "RF_ThC_roll": (-130, 50),
    "RF_ThC_yaw": (-50, 50),
    "RF_ThC_pitch": (-40, 60),
    "RF_CTr_pitch": (-180, 0),
    "RF_CTr_roll": (-150, 0),
    "RF_FTi_pitch": (0, 170),
    "RF_TiTa_pitch": (-150, 0),
    "LF_ThC_roll": (-50, 130),
    "LF_ThC_yaw": (-50, 50),
    "LF_ThC_pitch": (-40, 60),
    "LF_CTr_pitch": (-180, 0),
    "LF_CTr_roll": (0, 150),
    "LF_FTi_pitch": (0, 170),
    "LF_TiTa_pitch": (-150, 0),
}


_NMF_SIZE = {
    "RF_Coxa": 0.40,
    "RM_Coxa": 0.182,
    "RH_Coxa": 0.199,
    "LF_Coxa": 0.40,
    "LM_Coxa": 0.182,
    "LH_Coxa": 0.199,
    "RF_Femur": 0.69,
    "RM_Femur": 0.7829999999999999,
    "RH_Femur": 0.8360000000000001,
    "LF_Femur": 0.69,
    "LM_Femur": 0.7829999999999999,
    "LH_Femur": 0.8360000000000001,
    "RF_Tibia": 0.54,
    "RM_Tibia": 0.668,
    "RH_Tibia": 0.6849999999999998,
    "LF_Tibia": 0.54,
    "LM_Tibia": 0.668,
    "LH_Tibia": 0.6849999999999998,
    "RF_Tarsus": 0.63,
    "RM_Tarsus": 0.6949999999999998,
    "RH_Tarsus": 0.7950000000000002,
    "LF_Tarsus": 0.63,
    "LM_Tarsus": 0.6949999999999998,
    "LH_Tarsus": 0.7950000000000002,
    "RF": 2.26,
    "RM": 2.328,
    "RH": 2.515,
    "LF": 2.26,
    "LM": 2.328,
    "LH": 2.515,
    "Antenna": 0.2745906043549196,
    "Antenna_mid_thorax": 0.9355746896961248,
}

_NMF_PTS2ALIGN = {
    "R_head": ["base_anten_R", "tip_anten_R"],
    "RF_leg": [
        "thorax_coxa_R",
        "coxa_femur_R",
        "femur_tibia_R",
        "tibia_tarsus_R",
        "claw_R",
    ],
    "Thorax": ["thorax_wing_R", "thorax_midpoint_tether", "thorax_wing_L"],
    "L_head": ["base_anten_L", "tip_anten_L"],
    "LF_leg": [
        "thorax_coxa_L",
        "coxa_femur_L",
        "femur_tibia_L",
        "tibia_tarsus_L",
        "claw_L",
    ],
}


_NMF_SKELETON = [
    "base_anten_R",
    "tip_anten_R",
    "thorax_coxa_R",
    "coxa_femur_R",
    "femur_tibia_R",
    "tibia_tarsus_R",
    "claw_R",
    "thorax_wing_R",
    "thorax_midpoint_tether",
    "thorax_wing_L",
    "base_anten_L",
    "tip_anten_L",
    "thorax_coxa_L",
    "coxa_femur_L",
    "femur_tibia_L",
    "tibia_tarsus_L",
    "claw_L",
]


_NMF_TEMPLATE = {
    "RF_Coxa": np.array([0.33, -0.17, 1.07]),
    "RF_Femur": np.array([0.33, -0.17, 0.67]),
    "RF_Tibia": np.array([0.33, -0.17, -0.02]),
    "RF_Tarsus": np.array([0.33, -0.17, -0.56]),
    "RF_Claw": np.array([0.33, -0.17, -1.19]),
    "LF_Coxa": np.array([0.33, 0.17, 1.07]),
    "LF_Femur": np.array([0.33, 0.17, 0.67]),
    "LF_Tibia": np.array([0.33, 0.17, -0.02]),
    "LF_Tarsus": np.array([0.33, 0.17, -0.56]),
    "LF_Claw": np.array([0.33, 0.17, -1.19]),
    "R_Antenna_base": np.array([1.01, -0.10, 1.41]),
    "L_Antenna_base": np.array([1.01, 0.10, 1.41]),
    "R_Antenna_edge": np.array([1.06, -0.10, 1.14]),
    "L_Antenna_edge": np.array([1.06, 0.10, 1.14]),
    # "Labellum": np.array([0.75, 0.0, 0.81]),
    "R_post_vertical": np.array([0.7, -0.2, 1.59]),
    "L_post_vertical": np.array([0.7, 0.2, 1.59]),
    # "R_ant_orb": np.array([0.88, -0.18, 1.49]),
    # "L_ant_orb": np.array([0.88, 0.18, 1.49]),
    "R_wing": np.array([0.08, -0.4, 1.43]),
    "L_wing": np.array([0.08, 0.4, 1.43]),
    "Neck": np.array([0.53, 0.0, 1.3]),
    "Thorax_mid": np.array([0.08, 0.0, 1.43]),
    "L_dorsal_hum": np.array([0.41, 0.37, 1.32]),
    # "L_ant_notopleural": np.array([0.28, 0.39, 1.39]),
    "R_dorsal_hum": np.array([0.41, -0.37, 1.32]),
    # "R_ant_notopleural": np.array([0.30, -0.39, 1.39]),
}


neuromechfly_body_config = BodyConfig(
    initial_angles_rad=_NMF_INITIAL_ANGLES_RAD,
    segment_sizes=_NMF_SIZE,
    points_to_align=_NMF_PTS2ALIGN,
    skeleton=_NMF_SKELETON,
    template=_NMF_TEMPLATE,
)
neuromechfly_body_config.set_dof_bounds_in_deg(_NMF_BOUNDS_DEG)


# Define constants for backward compatibility but with deprecation warnings
class DeprecatedConstant:
    """Wrapper to issue deprecation warnings for constants."""

    def __init__(self, value, name):
        self._value = value
        self._name = name

    def __get__(self, instance, owner):
        message = (
            f"The constant `seqikpy.data.{self._name}` is deprecated and will be "
            "removed in a future version. Use the `seqikpy.data.BodyConfig` dataclass "
            "instead. You can import the default configuration "
            "`seqikpy.data.neuromechfly_body_config` and access its attributes. "
            "For example, instead of `NMF_TEMPLATE`, use "
            "`neuromechfly_body_config.template`."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return self._value

    def __set__(self, instance, value):
        raise AttributeError(
            f"`seqikpy.data.{self._name}` is a constant and cannot be modified."
        )


# Deprecated constants
INITIAL_ANGLES = DeprecatedConstant(_NMF_INITIAL_ANGLES_RAD, "INITIAL_ANGLES")
_nmf_bounds_rad = {
    key: (np.deg2rad(low_deg), np.deg2rad(up_deg))
    for key, (low_deg, up_deg) in _NMF_BOUNDS_DEG.items()
}
BOUNDS = DeprecatedConstant(_nmf_bounds_rad, "BOUNDS")
NMF_SIZE = DeprecatedConstant(_NMF_SIZE, "NMF_SIZE")
PTS2ALIGN = DeprecatedConstant(_NMF_PTS2ALIGN, "PTS2ALIGN")
SKELETON = DeprecatedConstant(_NMF_SKELETON, "SKELETON")
NMF_TEMPLATE = DeprecatedConstant(_NMF_TEMPLATE, "NMF_TEMPLATE")
