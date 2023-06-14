""" Data, constants, and paths. """
import numpy as np

INITIAL_ANGLES = {
    "RF": {
        "stage_1": np.array([0.0, 0.45, 0.37, -2.14]),
        "stage_2": np.array([0.0, -0.32, 0.45, 0.37, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.32, 0.45, 0.37, -2.14, -1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.32, 0.45, 0.37, -2.14, -1.25, 1.48, 0.0, 0.0]),
    },

    "LF": {
        "stage_1": np.array([0.0, -0.45, 0.37, -2.14]),
        "stage_2": np.array([0.0, 0.32, -0.45, 0.37, -2.14, 1.4]),
        "stage_3": np.array([0.0, 0.32, -0.45, 0.37, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, 0.32, -0.45, 0.37, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
    "head": np.array([0, -0.17, 0]),  #  none, roll, pitch, yaw
}

BOUNDS = {
    "RF_ThC_roll": (np.deg2rad(-130), np.deg2rad(-5)),
    "RF_ThC_yaw": (np.deg2rad(-10), np.deg2rad(60)),
    "RF_ThC_pitch": (np.deg2rad(-40), np.deg2rad(60)),
    "RF_CTr_pitch": (np.deg2rad(-180), np.deg2rad(-20)),
    "RF_CTr_roll": (np.deg2rad(-150), np.deg2rad(20)),
    "RF_FTi_pitch": (np.deg2rad(0), np.deg2rad(170)),
    "RF_TiTa_pitch": (np.deg2rad(-100), np.deg2rad(0)),
    "LF_ThC_roll": (np.deg2rad(5), np.deg2rad(130)),
    "LF_ThC_yaw": (np.deg2rad(-60), np.deg2rad(10)),
    "LF_ThC_pitch": (np.deg2rad(-40), np.deg2rad(60)),
    "LF_CTr_pitch": (np.deg2rad(-180), np.deg2rad(-20)),
    "LF_CTr_roll": (np.deg2rad(20), np.deg2rad(150)),
    "LF_FTi_pitch": (np.deg2rad(0), np.deg2rad(170)),
    "LF_TiTa_pitch": (np.deg2rad(-100), np.deg2rad(0)),
    "Head_roll": (np.deg2rad(-90), np.deg2rad(90)),
    "Head_pitch": (np.deg2rad(-40), np.deg2rad(70)),
    "Head_yaw": (np.deg2rad(-10), np.deg2rad(10)),
}

NMF_SIZE = {
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
    "LF": 2.26,  #  leg is updated
    "LM": 2.328,
    "LH": 2.515,
    "Antenna": 0.2745906043549196,  #  updated
    "Antenna_mid_thorax": 0.9355746896961248,  # updated
}

PTS2ALIGN = {
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


def get_pts2align(path: str):
    """ Deletes the keys from the PTS2ALIGN dictionary."""
    pts_temp = PTS2ALIGN.copy()
    if "_RF" in path:
        del pts_temp["RF_leg"]
    elif "_LF" in path:
        del pts_temp["LF_leg"]
    elif "_RLF" in path or "_LRF" in path:
        del pts_temp["LF_leg"]
        del pts_temp["RF_leg"]

    return pts_temp


SKELETON = [
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

NMF_TEMPLATE = {
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


"""
"L_dorsal_hum": np.array([0.404, 0.366, 1.299])
"L_ant_notopleural": np.array([0.2623, 0.401, 1.349])
"R_dorsal_hum": np.array([0.404, -0.366, 1.299])
"R_ant_notopleural": np.array([0.2623, -0.401, 1.349])

"""

# TODO: double check this based on the resting pose.
_NMF_TEMPLATE_OLD = {
    "RF_Coxa": np.array([0.33, -0.18, 1.03]),
    "RF_Femur": np.array([0.33, -0.18, 0.64]),
    "RF_Tibia": np.array([0.33, -0.18, -0.06]),
    "RF_Tarsus": np.array([0.33, -0.18, -0.58]),
    "RF_Claw": np.array([0.35, -0.27, -1.12]),
    "LF_Coxa": np.array([0.33, 0.18, 1.03]),
    "LF_Femur": np.array([0.33, 0.18, 0.64]),
    "LF_Tibia": np.array([0.33, 0.18, -0.06]),
    "LF_Tarsus": np.array([0.33, 0.18, -0.58]),
    "LF_Claw": np.array([0.33, 0.18, -1.12]),
    "R_Antenna_base": np.array([0.98, -0.1, 1.40]),
    "L_Antenna_base": np.array([0.98, 0.1, 1.40]),
    "R_Antenna_edge": np.array([1.01, -0.08, 1.12]),
    "L_Antenna_edge": np.array([1.01, 0.08, 1.12]),
    "Labellum": np.array([0.75, 0.0, 0.81]),
    "R_post_vertical": np.array([0.7, -0.2, 1.59]),
    "L_post_vertical": np.array([0.7, 0.2, 1.59]),
    "R_ant_orb": np.array([0.88, -0.18, 1.49]),
    "L_ant_orb": np.array([0.88, 0.18, 1.49]),
    "R_wing": np.array([0.01, -0.4, 1.31]),
    "L_wing": np.array([0.01, 0.4, 1.31]),
    "Neck": np.array([0.52, 0.0, 1.3]),
    "Thorax_mid": np.array([0.01, 0.0, 1.31]),
    "L_dorsal_hum": np.array([0.404, 0.366, 1.299]),
    "L_ant_notopleural": np.array([0.2623, 0.401, 1.349]),
    "R_dorsal_hum": np.array([0.404, -0.366, 1.299]),
    "R_ant_notopleural": np.array([0.2623, -0.401, 1.349]),
}
