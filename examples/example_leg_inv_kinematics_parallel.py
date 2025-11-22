"""
Example usage of leg inverse kinematics module.
It speeds up the process by running the pipeline in parallel.
Experiment on a Macbook Pro 2.3 GHz Quad-Core Intel Core i7, running IK on 6 legs:
Sequential IK took 1.7729304512341817 mins [serial]
Sequential IK took 0.58028298219045 mins [parallel]

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pathlib import Path

from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq
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


if __name__ == "__main__":
    import seqikpy

    legs = [f"{side}{pos}" for side in "RL" for pos in ["F", "M", "H"]]

    # Load aligned pose
    data_dir = (
        Path(seqikpy.__path__[0]).parent / "data/df3d_pose_result__210902_PR_Fly1"
    )
    aligned_pose_data_original = load_file(data_dir / "pose3d_aligned.pkl")

    # Artificially repeat data to increase sequence length for benchmarking
    n_repeats = 8
    aligned_pose_data = {}
    for key, arr in aligned_pose_data_original.items():
        arr_backward = arr[::-1, ...]
        aligned_pose_data[key] = np.concatenate(
            [arr, arr_backward] * (n_repeats // 2) + [arr] * (n_repeats % 2), axis=0
        )

    # Define kinematic chains
    kin_chain = KinematicChainSeq(
        bounds_dof=BOUNDS_LOCOMOTION,
        body_size=calculate_body_size(TEMPLATE_NMF_LOCOMOTION, legs),
        legs_list=legs,
    )

    # Define leg IK instance
    class_seq_ik = LegInvKinSeq(
        aligned_pos=aligned_pose_data,
        kinematic_chain_class=kin_chain,
        initial_angles=INITIAL_ANGLES_LOCOMOTION,
        log_level="INFO",
    )

    # Run inverse and forward kinematics
    # Serial processing
    print("Running in series...")
    start = time()
    results_serial = class_seq_ik.run_ik_and_fk(n_workers=1)
    walltime_serial = time() - start
    print(f"Sequential IK took {walltime_serial} secs")

    # Parallel over legs only
    print("Running in parallel over legs only...")
    start = time()
    results_parallel_legs = class_seq_ik.run_ik_and_fk(
        n_workers=-1, parallel_over_time=False
    )
    walltime_parallel_legs = time() - start
    print(f"Parallel IK over legs took {walltime_parallel_legs} secs")

    # Parallel over legs and over time
    print("Running in parallel over legs and over time...")
    start = time()
    results_parallel_legs_and_time = class_seq_ik.run_ik_and_fk(
        n_workers=-1, parallel_over_time=True
    )
    walltime_parallel_legs_and_time = time() - start
    print(f"Parallel IK over legs and time took {walltime_parallel_legs_and_time} secs")

    # Save results
    output_path = data_dir / "parallel_inv_and_fwd_kinematics_benchmark.pkl"
    print(f"Saving results to {output_path}...")
    with open(output_path, "wb") as f:
        data = {
            "serial": {
                "walltime_secs": walltime_serial,
                "leg_joint_angles": results_serial[0],
                "forward_kinematics": results_serial[1],
            },
            "parallel_legs": {
                "walltime_secs": walltime_parallel_legs,
                "leg_joint_angles": results_parallel_legs[0],
                "forward_kinematics": results_parallel_legs[1],
            },
            "parallel_legs_and_time": {
                "walltime_secs": walltime_parallel_legs_and_time,
                "leg_joint_angles": results_parallel_legs_and_time[0],
                "forward_kinematics": results_parallel_legs_and_time[1],
            },
        }
        pickle.dump(data, f)

    # Load and inspect saved results
    with open(data_dir / "parallel_inv_and_fwd_kinematics_benchmark.pkl", "rb") as f:
        data = pickle.load(f)
        print(data.keys())

    example_dof = "Angle_RF_ThC_pitch"
    seq_serial = data["serial"]["leg_joint_angles"][example_dof]
    seq_parallel_legs = data["parallel_legs"]["leg_joint_angles"][example_dof]
    seq_parallel_time = data["parallel_legs_and_time"]["leg_joint_angles"][example_dof]

    plt.plot(
        np.rad2deg(seq_serial),
        linestyle="-",
        color="black",
        label="Serial",
    )
    plt.plot(
        np.rad2deg(seq_parallel_legs),
        linestyle=":",
        color="tab:blue",
        label="Parallel over legs",
    )
    plt.plot(
        np.rad2deg(seq_parallel_time),
        linestyle="--",
        color="tab:red",
        label="Parallel over legs and time",
    )
    plt.xlabel("Frame index")
    plt.ylabel(f"{example_dof} (deg)")
    plt.legend()
    plt.title("Leg Inverse Kinematics Result Comparison")
    plt.show()

    diff_serial_parallel_legs_all = []
    diff_serial_parallel_time_all = []
    for key in data["serial"]["leg_joint_angles"].keys():
        seq_serial = data["serial"]["leg_joint_angles"][key]
        seq_parallel_legs = data["parallel_legs"]["leg_joint_angles"][key]
        seq_parallel_time = data["parallel_legs_and_time"]["leg_joint_angles"][key]

        diff_serial_parallel_legs = np.abs(seq_serial - seq_parallel_legs)
        diff_serial_parallel_time = np.abs(seq_serial - seq_parallel_time)

        range_ = np.nanpercentile(seq_serial, 99) - np.nanpercentile(seq_serial, 1)
        diff_serial_parallel_legs_all.append(diff_serial_parallel_legs / range_)
        diff_serial_parallel_time_all.append(diff_serial_parallel_time / range_)

    diff_serial_parallel_legs_all = np.stack(diff_serial_parallel_legs_all)
    diff_serial_parallel_time_all = np.stack(diff_serial_parallel_time_all)

    print(
        "Max % absolute difference between serial and parallel over legs: ",
        np.nanmax(diff_serial_parallel_legs_all),
    )
    print(
        "Max % absolute difference between serial and parallel over legs and time: ",
        np.nanmax(diff_serial_parallel_time_all),
    )
