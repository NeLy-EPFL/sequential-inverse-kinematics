"""
    Example usage of leg inverse kinematics module.
    It speeds up the process by running the pipeline in parallel.
    Experiment on a Macbook Pro 2.3 GHz Quad-Core Intel Core i7, running IK on 6 legs:
    Sequential IK took 1.7729304512341817 mins [serial]
    Sequential IK took 0.58028298219045 mins [parallel]

"""
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from seqikpy.kinematic_chain import KinematicChainSeq, KinematicChainGeneric
from seqikpy.leg_inverse_kinematics import LegInvKinSeq, LegInvKinGeneric
from seqikpy.data import NMF_SIZE
from seqikpy.utils import load_file, calculate_body_size

TEMPLATE_NMF_LOCOMOTION = {
    "RF_Coxa": np.array([0.35, -0.27, 0.400]),
    "RF_Femur": np.array([0.35, -0.27, -0.025]),
    "RF_Tibia": np.array([0.35, -0.27, -0.731]),
    "RF_Tarsus": np.array([0.35, -0.27, -1.249]),
    "RF_Claw": np.array([0.35, -0.27, -1.912]),
    "LF_Coxa": np.array([0.35, 0.27, 0.400]),
    "LF_Femur": np.array([0.35, 0.27, -0.025]),
    "LF_Tibia": np.array([0.35, 0.27, -0.731]),
    "LF_Tarsus": np.array([0.35, 0.27, -1.249]),
    "LF_Claw": np.array([0.35, 0.27, -1.912]),
    "RM_Coxa": np.array([0, -0.125, 0]),
    "RM_Femur": np.array([0, -0.125, -0.182]),
    "RM_Tibia": np.array([0, -0.125, -0.965]),
    "RM_Tarsus": np.array([0, -0.125, -1.633]),
    "RM_Claw": np.array([0, -0.125, -2.328]),
    "LM_Coxa": np.array([0, 0.125, 0]),
    "LM_Femur": np.array([0, 0.125, -0.182]),
    "LM_Tibia": np.array([0, 0.125, -0.965]),
    "LM_Tarsus": np.array([0, 0.125, -1.633]),
    "LM_Claw": np.array([0, 0.125, -2.328]),
    "RH_Coxa": np.array([-0.215, -0.087, -0.073]),
    "RH_Femur": np.array([-0.215, -0.087, -0.272]),
    "RH_Tibia": np.array([-0.215, -0.087, -1.108]),
    "RH_Tarsus": np.array([-0.215, -0.087, -1.793]),
    "RH_Claw": np.array([-0.215, -0.087, -2.588]),
    "LH_Coxa": np.array([-0.215, 0.087, -0.073]),
    "LH_Femur": np.array([-0.215, 0.087, -0.272]),
    "LH_Tibia": np.array([-0.215, 0.087, -1.108]),
    "LH_Tarsus": np.array([-0.215, 0.087, -1.793]),
    "LH_Claw": np.array([-0.215, 0.087, -2.588]),
}

INITIAL_ANGLES_LOCOMOTION = {
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
    "LF": {
        "stage_1": np.array([0.0, -0.45, -0.07, -2.14]),
        "stage_2": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.45, -0.07, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
    "RM": {
        "stage_1": np.array([0.0, 0.45, 0.37, -2.14]),
        "stage_2": np.array([0.0, 0.45, 0.37, -0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, 0.45, 0.37, -0.32, -2.14, -1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, 0.45, 0.37, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
    },
    "LM": {
        "stage_1": np.array([0.0, -0.45, 0.37, -2.14]),
        "stage_2": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.45, 0.37, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
    "RH": {
        "stage_1": np.array([0.0, 0.45, 0.07, -2.14]),
        "stage_2": np.array([0.0, 0.45, 0.07, -0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, 0.45, 0.07, -0.32, -2.14, -1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, 0.45, 0.07, -0.32, -2.14, -1.25, 1.48, 0.0, 0.0]),
    },
    "LH": {
        "stage_1": np.array([0.0, -0.45, 0.07, -2.14]),
        "stage_2": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.4]),
        "stage_3": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.25, 1.48, 0.0]),
        "stage_4": np.array([0.0, -0.45, 0.07, 0.32, -2.14, 1.25, 1.48, 0.0, 0.0]),
    },
}

BOUNDS_LOCOMOTION = {
    "RF_ThC_yaw": (-3.141592653589793, 3.141592653589793),
    "RF_ThC_pitch": (np.deg2rad(-90), np.deg2rad(90)),
    "RF_ThC_roll": (-3.141592653589793, 3.141592653589793),
    "RF_CTr_pitch": (-3.141592653589793, 3.141592653589793),
    "RF_FTi_pitch": (-3.141592653589793, 3.141592653589793),
    "RF_CTr_roll": (-3.141592653589793, 3.141592653589793),
    "RF_TiTa_pitch": (-3.141592653589793, np.deg2rad(0)),
    "RM_ThC_yaw": (np.deg2rad(-50), np.deg2rad(50)),
    "RM_ThC_pitch": (-3.141592653589793, 3.141592653589793),
    "RM_ThC_roll": (-3.141592653589793, 0),
    "RM_CTr_pitch": (-3.141592653589793, 3.141592653589793),
    "RM_FTi_pitch": (-3.141592653589793, 3.141592653589793),
    "RM_CTr_roll": (-3.141592653589793, 3.141592653589793),
    "RM_TiTa_pitch": (-3.141592653589793, np.deg2rad(0)),
    "RH_ThC_yaw": (np.deg2rad(-50), np.deg2rad(50)),
    "RH_ThC_pitch": (np.deg2rad(-50), np.deg2rad(50)),
    "RH_ThC_roll": (-3.141592653589793, 0),
    "RH_CTr_pitch": (np.deg2rad(-180), np.deg2rad(0)),
    "RH_FTi_pitch": (-3.141592653589793, 3.141592653589793),
    "RH_CTr_roll": (-3.141592653589793, 3.141592653589793),
    "RH_TiTa_pitch": (-3.141592653589793, np.deg2rad(0)),
    "LF_ThC_yaw": (-3.141592653589793, 3.141592653589793),
    "LF_ThC_pitch": (np.deg2rad(-90), np.deg2rad(90)),
    "LF_ThC_roll": (-3.141592653589793, 3.141592653589793),
    "LF_CTr_pitch": (-3.141592653589793, 3.141592653589793),
    "LF_FTi_pitch": (-3.141592653589793, 3.141592653589793),
    "LF_CTr_roll": (-3.141592653589793, 3.141592653589793),
    "LF_TiTa_pitch": (-3.141592653589793, np.deg2rad(0)),
    "LM_ThC_yaw": (np.deg2rad(-50), np.deg2rad(50)),
    "LM_ThC_pitch": (-3.141592653589793, 3.141592653589793),
    "LM_ThC_roll": (0, 3.141592653589793),
    "LM_CTr_pitch": (-3.141592653589793, 3.141592653589793),
    "LM_FTi_pitch": (-3.141592653589793, 3.141592653589793),
    "LM_CTr_roll": (-3.141592653589793, 3.141592653589793),
    "LM_TiTa_pitch": (-3.141592653589793, np.deg2rad(0)),
    "LH_ThC_yaw": (np.deg2rad(-50), np.deg2rad(50)),
    "LH_ThC_pitch": (np.deg2rad(-50), np.deg2rad(50)),
    "LH_ThC_roll": (0, 3.141592653589793),
    "LH_CTr_pitch": (np.deg2rad(-180), np.deg2rad(0)),
    "LH_FTi_pitch": (-3.141592653589793, 3.141592653589793),
    "LH_CTr_roll": (-3.141592653589793, 3.141592653589793),
    "LH_TiTa_pitch": (-3.141592653589793, np.deg2rad(0)),
}


def worker_wrapper(aligned_pos, single_leg):
    """ Wrapper for the function to run single leg IK. """
    kin_chain = KinematicChainSeq(
        bounds_dof=BOUNDS_LOCOMOTION,
        body_size=calculate_body_size(
            TEMPLATE_NMF_LOCOMOTION,
            [single_leg]
        ),
        legs_list=[single_leg],
    )
    class_seq_ik = LegInvKinSeq(
        aligned_pos=aligned_pos,
        kinematic_chain_class=kin_chain,
        initial_angles=INITIAL_ANGLES_LOCOMOTION,
        log_level="ERROR"
    )
    leg_joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(hide_progress_bar=True)
    return leg_joint_angles, forward_kinematics


if __name__ == "__main__":
    # Aligned pose
    data_path = Path("../data/df3d_pose_result__210902_PR_Fly1")
    pose_data = load_file(
        data_path / "pose3d_aligned.pkl"
    )

    legs_to_align = ["RF", "RM", "RH", "LF", "LM", "LH"]

    # start = time.time()

    # for leg in legs_to_align:
    #     worker_wrapper(pose_data, leg)
    # end = time.time()
    # total_time = (end - start) / 60.0

    # print(f'Sequential IK took {total_time} mins [serial]')

    start = time.time()
    # Dictionary to hold concatenated results
    all_legs_joint_angles = {}
    all_legs_for_kins = []

    pool = Pool(processes=6)
    results = pool.starmap(worker_wrapper, [(pose_data, leg) for leg in legs_to_align])
    pool.close()
    pool.join()

    for ik, fk in results:
        all_legs_joint_angles.update(ik)
        all_legs_for_kins.append(fk)

    end = time.time()
    total_time = (end - start) / 60.0

    print(f'Sequential IK took {total_time} mins [parallel]')

    # Compare the joint angles
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for key in all_legs_joint_angles:
        plt.plot(all_legs_joint_angles[key], label=key[6:], lw=2)

    plt.xlabel('Frames (AU)')
    plt.ylabel('Angles (rad)')
    plt.title('Leg joint angles from SeqIK')
    plt.legend()
    plt.grid(True)
    plt.show()
