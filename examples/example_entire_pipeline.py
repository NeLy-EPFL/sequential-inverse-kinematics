"""
    Runs the entire pipeline from pose alignment to joint angles on a path given by the user.

    Example:
    >>> python example_entire_pipeline.py -p /mnt/nas/GO/7cam/220413_aJO-GAL4xUAS-CsChr/Fly001
    >>> python example_entire_pipeline.py -p ../data/anipose_220525_aJO_Fly001_001
"""
import logging
from pathlib import Path
import time
import argparse

from nmf_ik.alignment import AlignPose
from nmf_ik.kinematic_chain import KinematicChain
from nmf_ik.leg_inverse_kinematics import LegInverseKinematics
from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
from nmf_ik.data import BOUNDS, INITIAL_ANGLES, NMF_TEMPLATE
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

        start = time.time()

        align = AlignPose.from_file_path(
            main_dir=DATA_PATH,
            file_name="pose3d.h5",
            convert_dict=True,
            include_claw=False,
            nmf_template=NMF_TEMPLATE,
        )

        aligned_pos = align.align_pose(export_path=DATA_PATH)

        class_hk = HeadInverseKinematics(
            aligned_pos=aligned_pos,
            nmf_template=NMF_TEMPLATE,
        )
        head_joint_angles = class_hk.compute_head_angles(export_path=DATA_PATH)

        class_seq_ik = LegInverseKinematics(
            aligned_pos=aligned_pos,
            kinematic_chain_class=KinematicChain(
                bounds_dof=BOUNDS,
                nmf_size=None,
            ),
            initial_angles=INITIAL_ANGLES
        )
        leg_joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(export_path=DATA_PATH)

        full_body_ik = {**head_joint_angles, **leg_joint_angles}

        save_file(DATA_PATH / "body_joint_angles.pkl", full_body_ik)

        end = time.time()
        total_time = (end - start) / 60.0

        print(f"Total time taken to execute the code: {total_time} mins")

        # plot the joint angles
        PLOT = True
        if PLOT:
            import matplotlib.pyplot as plt

            for ja_name, ja_value in full_body_ik.items():
                plt.plot(ja_value, label=ja_name, lw=2)
            plt.legend()
            plt.show()
