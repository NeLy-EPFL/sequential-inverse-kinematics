"""
    Runs the entire pipeline from pose alignment to joint angles on a path given by the user.

    Example usage:
    >>> python example_entire_pipeline.py -p ../data/anipose_220525_aJO_Fly001_001 --plot
"""
import logging
from pathlib import Path
import time
import argparse
import numpy as np

from seqikpy.alignment import AlignPose, convert_from_anipose_to_dict
from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq
from seqikpy.head_inverse_kinematics import HeadInverseKinematics
from seqikpy.data import BOUNDS, INITIAL_ANGLES, NMF_TEMPLATE, PTS2ALIGN
from seqikpy.utils import save_file

logging.basicConfig(
    format=" %(asctime)s - %(levelname)s- %(message)s",
    handlers=[logging.StreamHandler()]
)


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
        help="Path of 3D pose data.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the joint angles.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    path_name = args.path

    path_name += "/" if not path_name.endswith("/") else ""

    paths = Path(path_name).rglob("pose-3d")

    for data_path in paths:

        logging.info("Running code in %s", data_path)

        start = time.time()
        # Align the 3D pose
        align = AlignPose.from_file_path(
            main_dir=data_path,
            file_name="pose3d.h5",
            legs_list=["RF", "LF"],
            convert_func=convert_from_anipose_to_dict,
            pts2align=PTS2ALIGN,
            include_claw=False,
            nmf_template=NMF_TEMPLATE,
            log_level="INFO"
        )

        aligned_pos = align.align_pose(export_path=data_path)
        # Compute the head joint angles
        class_hk = HeadInverseKinematics(
            aligned_pos=aligned_pos,
            nmf_template=NMF_TEMPLATE,
        )
        head_joint_angles = class_hk.compute_head_angles(
            export_path=data_path,
            compute_ant_angles=True
        )
        # Calculate the leg joint angles using the sequential IK
        class_seq_ik = LegInvKinSeq(
            aligned_pos=aligned_pos,
            kinematic_chain_class=KinematicChainSeq(
                bounds_dof=BOUNDS,
                legs_list=["RF", "LF"],
                nmf_size=None,
            ),
            initial_angles=INITIAL_ANGLES
        )
        leg_joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(
            export_path=data_path,
            hide_progress_bar=False
        )

        full_body_ik = {**head_joint_angles, **leg_joint_angles}

        save_file(data_path / "body_joint_angles.pkl", full_body_ik)

        end = time.time()
        total_time = (end - start) / 60.0

        print(f"Total time taken to execute the code: {total_time} mins")

        # Plot the joint angles
        if args.plot:
            import matplotlib.pyplot as plt

            time_step = 1e-2
            time = time = np.arange(0, full_body_ik['Angle_head_roll'].shape[0], 1) * time_step

            for ja_name, ja_value in full_body_ik.items():
                plt.plot(ja_value, label=ja_name, lw=2)

            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Joint angles (rad)")
            plt.title("Head and foreleg joint angles")
            plt.show()
