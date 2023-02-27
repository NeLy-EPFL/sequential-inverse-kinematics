"""
    Runs the entire pipeline from pose alignment to joint angles on a path given by the user.

    Example:
    >>> python example_entire_pipeline.py -p /mnt/nas/GO/7cam/220413_aJO-GAL4xUAS-CsChr/Fly001
"""
import pickle
import logging
from pathlib import Path
import time
import argparse

from nmf_ik.alignment import AlignPose
from nmf_ik.leg_inverse_kinematics import LegInverseKinematics
from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
from nmf_ik.data import BOUNDS, INITIAL_ANGLES, NMF_TEMPLATE, PTS2ALIGN
from nmf_ik.utils import save_file

# Change the logging level here
logging.basicConfig(level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s")


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline to visualize camera positions",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=None,
        help="Path of pose files",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    path_name = args.path

    path_name += "/" if not path_name.endswith("/") else ""

    paths = Path(path_name).rglob("pose-3d")

    # DATA_PATH = Path('../data/anipose/220525_aJO_Fly001_001/new-template')
    # f_path = DATA_PATH / "pose3d.h5"

    for DATA_PATH in paths:

        logging.info("Running code in %s", DATA_PATH)

        f_path = DATA_PATH / "pose3d.h5"
        with open(f_path, "rb") as f:
            data = pickle.load(f)

        start = time.time()

        align = AlignPose(DATA_PATH, pts2align=PTS2ALIGN, nmf_template=NMF_TEMPLATE)
        aligned_pos = align.align_pose(
            save_pose_file=True,
        )

        class_hk = HeadInverseKinematics(
            aligned_pos=aligned_pos,
            nmf_template=NMF_TEMPLATE,
            angles_to_calculate=[
                "Angle_head_roll",
                "Angle_head_pitch",
                "Angle_head_yaw",
                "Angle_antenna_pitch_L",
                "Angle_antenna_pitch_R",
                "Angle_antenna_yaw_L",
                "Angle_antenna_yaw_R",
            ],
        )
        head_joint_angles = class_hk.compute_head_angles(export_path=DATA_PATH)

        class_seq_ik = LegInverseKinematics(
            aligned_pos=aligned_pos, bounds=BOUNDS, initial_angles=INITIAL_ANGLES
        )
        leg_joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(export_path=DATA_PATH)

        full_body_ik = {**head_joint_angles, **leg_joint_angles}

        save_file(DATA_PATH / "body_joint_angles.pkl", full_body_ik)

        end = time.time()
        total_time = (end - start) / 60.0

        print(f"Total time taken to execute the code: {total_time} mins")
