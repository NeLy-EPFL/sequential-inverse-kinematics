""" Plots the raw 3D pose and forward kinematics in the same axis.
    Example usage:
    >>> python make_video.py --data_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d'
"""
import pickle
import logging
from pathlib import Path
from datetime import date
import time
import numpy as np
import argparse
from nmf_ik.visualization import animate_3d_points

# Change the logging level here
logging.basicConfig(level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s")


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline to visualize camera positions",
    )
    parser.add_argument(
        "-dp",
        "--data_path",
        type=str,
        default=None,
        help="Path of pose files",
    )
    return parser.parse_args()


if __name__ == "__main__":

    start = time.time()
    today = date.today()

    fps = 100
    args = parse_args()
    DATA_PATH = Path(args.data_path)
    print(DATA_PATH)
    # DATA_PATH = Path('../data/anipose/220525_aJO_Fly001_001/new-template')
    anipose_data = DATA_PATH / "pose3d_aligned.pkl"
    forward_kinematics = DATA_PATH / "forward_kinematics.pkl"
    out_dir = DATA_PATH / f"inverse_kinematics_results_{today}_front.mp4"

    with open(anipose_data, "rb") as f:
        aligned_pose = pickle.load(f)
    with open(forward_kinematics, "rb") as f:
        forward_kin = pickle.load(f)

    points_aligned_all = np.concatenate(
        (
            aligned_pose["RF_leg"],
            aligned_pose["LF_leg"],
            aligned_pose["R_head"],
            aligned_pose["L_head"],
            np.tile(aligned_pose["Neck"], (aligned_pose["RF_leg"].shape[0], 1)).reshape(-1, 1, 3),
        ),
        axis=1,
    )

    points_fk = np.concatenate(
        (
            forward_kin["RF_leg"],
            forward_kin["LF_leg"],
            # forward_kin["R_head"],
            # forward_kin["L_head"],
            # np.tile(aligned_pose["Neck"],(aligned_pose["RF_leg"].shape[0],1)).reshape(-1,1,3),
        ),
        axis=1,
    )

    KEY_POINTS_DICT = {
        "RF": (np.arange(0, 5), "solid"),
        "R Ant": (np.arange(10, 12), "o"),
        "Neck": (np.arange(14, 15), "x"),
        "L Ant": (np.arange(12, 14), "o"),
        "LF": (np.arange(5, 10), "solid"),
    }

    KEY_POINTS_DICT2 = {
        "RF": (np.arange(0, 9), ":"),
        # "R Ant": (np.arange(10, 12), "."),
        # "Neck": (np.arange(14,15), "x"),
        # "L Ant": (np.arange(12, 14), "."),
        "LF": (np.arange(9, 18), ":"),
    }

    animate_3d_points(
        points_aligned_all,
        KEY_POINTS_DICT,
        points3d_second=points_fk,
        key_points_second=KEY_POINTS_DICT2,
        export_path=out_dir,
        frame_no=6000,
        elev=0,
        azim=0,
    )

    end = time.time()
    total_time = (end - start) / 60.

    print(f"Total time taken to execute the code: {total_time} min")
