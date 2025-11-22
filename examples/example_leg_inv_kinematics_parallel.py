import matplotlib
matplotlib.use("Agg")

import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pathlib import Path

import seqikpy
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
    # This script runs a weak scaling test (sequence length is proportional to number
    # of workers. Change the multiplier below to global adjust sequence lengths.
    task_size_multiplier = 1
    hide_progress_bar = True

    # Load aligned pose
    data_dir = (
        Path(seqikpy.__path__[0]).parent / "data/df3d_pose_result__210902_PR_Fly1"
    )
    aligned_pose_data_original = load_file(data_dir / "pose3d_aligned.pkl")

    def _generate_input_sequence(n_repeats):
        """Artificially repeat data to increase sequence length for benchmarking"""
        repeated_pose_data = {}
        for key, arr in aligned_pose_data_original.items():
            arr_back = arr[::-1, ...]
            repeated_pose_data[key] = np.concatenate(
                [arr, arr_back] * (n_repeats // 2) + [arr] * (n_repeats % 2), axis=0
            )
        return repeated_pose_data

    def _run_invik_pipeline(n_workers, **kwargs):
        n_repeats = n_workers * task_size_multiplier
        aligned_pose_data = _generate_input_sequence(n_repeats)
        seq_length = aligned_pose_data[list(aligned_pose_data.keys())[0]].shape[0]
        legs = [f"{side}{pos}" for side in "RL" for pos in ["F", "M", "H"]]
        
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
        
        # Solve inverse and forward kinematics
        start_time = time()
        joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(
            n_workers=n_workers, hide_progress_bar=True, **kwargs
        )
        wall_time = time() - start_time
        
        return {
            "wall_time": wall_time,
            "seq_length": seq_length,
            "n_workers": n_workers,
            "joint_angles": joint_angles,
            "forward_kinematics": forward_kinematics
        }

    # Run weak scaling test (task size is proportional to number of workers)
    # n_cpu_cores = joblib.cpu_count(only_physical_cores=True)
    n_cpu_cores = 36  # joblib.cpu_count is unreliable on clusters - hardcoding it
    assert n_cpu_cores >= 12, (
        "At least 12-ish CPU cores required for this scaling test to make sense"
    )

    print("Running in series...")
    res_serial = _run_invik_pipeline(n_workers=1)
    print(f"Serial processing done in {res_serial['wall_time']} secs")

    print("Running in parallel (over legs only)...")
    res_par_legs = _run_invik_pipeline(n_workers=6, parallel_over_time=False)
    print(f"Parallel-over-legs processing done in {res_par_legs['wall_time']} secs")
    
    print("Running in parallel (over legs and time)...")
    res_par_time = _run_invik_pipeline(n_workers=n_cpu_cores)
    print(f"Parallel-over-time processing done in {res_par_time['wall_time']} secs")

    # Save results
    output_path = data_dir / "parallel_inv_and_fwd_kinematics_benchmark.pkl"
    print(f"Saving results to {output_path}...")
    with open(output_path, "wb") as f:
        data = {
            "serial": res_serial,
            "parallel_legs": res_par_legs,
            "parallel_legs_and_time": res_par_time,
        }
        pickle.dump(data, f)

    # Load and inspect saved results
    with open(data_dir / "parallel_inv_and_fwd_kinematics_benchmark.pkl", "rb") as f:
        data = pickle.load(f)

    example_dof = "Angle_RF_ThC_pitch"
    seq_serial = data["serial"]["joint_angles"][example_dof]
    seq_parallel_legs = data["parallel_legs"]["joint_angles"][example_dof]
    seq_parallel_time = data["parallel_legs_and_time"]["joint_angles"][example_dof]

    plt.plot(
        np.rad2deg(seq_serial),
        linestyle="-",
        color="black",
        label="Serial",
    )
    plt.plot(
        np.rad2deg(seq_parallel_legs[:seq_serial.shape[0]]),
        linestyle=":",
        color="tab:blue",
        label="Parallel over legs",
    )
    plt.plot(
        np.rad2deg(seq_parallel_time[:seq_serial.shape[0]]),
        linestyle="--",
        color="tab:red",
        label="Parallel over legs and time",
    )
    plt.xlabel("Frame index")
    plt.ylabel(f"{example_dof} (deg)")
    plt.legend()
    plt.title("Leg Inverse Kinematics Result Comparison")
    plt.savefig(data_dir / "output_comparison.png")

    # Result should be identical if only parallelizing over legs
    for dof_key, serial_output in data["serial"]["joint_angles"].items():
        par_legs_output = data["parallel_legs"]["joint_angles"][dof_key]
        assert np.allclose(serial_output, par_legs_output[:serial_output.shape[0]])
    print(
        "Results are identical between serial processing and parallel processing "
        "over legs only - OK"
    )

    # Result should be close if parallelizing over legs and time
    par_legs_all = []
    diff_all = []
    for dof_key, par_legs_output in data["parallel_legs"]["joint_angles"].items():
        par_time_output = data["parallel_legs_and_time"]["joint_angles"][dof_key]
        diff = np.abs(par_legs_output - par_time_output[:par_legs_output.shape[0]])
        diff_all.append(diff)
        par_legs_all.append(par_legs_output)
    par_legs_all = np.concatenate(par_legs_all)
    range_ = np.nanpercentile(par_legs_all, 95) - np.nanpercentile(par_legs_all, 5)
    diff_all_norm = np.concatenate(diff_all) / range_
    diff_all_norm = diff_all_norm[~np.isnan(diff_all_norm)]
    max_diff_norm = diff_all_norm.max()
    nonzero_mask = ~np.isclose(diff_all_norm, 0)
    nonzero_diff_frac = nonzero_mask.sum() / nonzero_mask.size
    nonzero_diff_mean = np.mean(diff_all_norm[nonzero_mask])
    print(
        f"Parallelizing over legs and over time:\n"
        f"  Max normalized difference: {max_diff_norm}\n"
        f"  Fraction of values with non-zero difference: {nonzero_diff_frac}\n"
        f"  Mean difference among different frames: {nonzero_diff_mean}"
    )
    assert max_diff_norm < 1e-3
    print("Max difference in results is smaller enough - OK")
    assert nonzero_diff_mean < 1e-4
    print("Mean difference in results among nonzero frames is smaller enough - OK")
    

    # Calculate speedup
    steps_per_sec_by_mode = {}
    for mode, res in data.items():
        seq_length = res["seq_length"]
        wall_time = res["wall_time"]
        steps_per_sec_by_mode[mode] = seq_length / wall_time

    serial_steps_per_sec = steps_per_sec_by_mode["serial"]
    for mode, steps_per_sec in steps_per_sec_by_mode.items():
        speedup = steps_per_sec / serial_steps_per_sec
        n_workers = data[mode]["n_workers"]
        print(f"{mode}: {speedup:.2f}x speedup with {n_workers} processes")

