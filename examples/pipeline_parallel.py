"""
    Parallel running of the pipeline.
"""
import logging
from pathlib import Path
import time
import multiprocessing

from nmf_ik.alignment import AlignPose
from nmf_ik.leg_inverse_kinematics import LegInverseKinematics
from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
from nmf_ik.data import BOUNDS, INITIAL_ANGLES, NMF_TEMPLATE, PTS2ALIGN
from nmf_ik.utils import save_file

# Change the logging level here
logging.basicConfig(level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s")

NO_CORES = multiprocessing.cpu_count()


def worker_wrapper(arg):
    """ Provide kwargs during multiprocessing. """
    return run_pipeline(arg)


def run_pipeline(path):

    logging.info("Running code in %s", path)

    if "_RF" in path:
        logging.info("Right leg is amputated!")
        del PTS2ALIGN["RF_leg"]
    elif "_LF" in path:
        logging.info("Left leg is amputated!")
        del PTS2ALIGN["LF_leg"]
    elif "_RLF" in path or "_LRF" in path:
        logging.info("Both legs are amputated!")
        del PTS2ALIGN["LF_leg"]
        del PTS2ALIGN["RF_leg"]

    align = AlignPose.from_file_path(
        main_dir=path,
        file_name="pose3d.h5",
        convert_dict=True,
        pts2align=PTS2ALIGN,
        include_claw=False,
        nmf_template=NMF_TEMPLATE,
    )

    aligned_pos = align.align_pose(export_path=path)

    class_hk = HeadInverseKinematics(
        aligned_pos=aligned_pos,
        nmf_template=NMF_TEMPLATE,
    )
    head_joint_angles = class_hk.compute_head_angles(export_path=path)

    if (Path(path) / "body_joint_angles.pkl").is_file():
        logging.info("Joint angles exist!!")
        return

    if 'RLF' not in path:
        logging.info("Running leg IK")

        class_seq_ik = LegInverseKinematics(
            aligned_pos=aligned_pos, bounds=BOUNDS, initial_angles=INITIAL_ANGLES
        )
        leg_joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(export_path=path)

        full_body_ik = {**head_joint_angles, **leg_joint_angles}

        save_file(Path(path) / "body_joint_angles.pkl", full_body_ik)
    else:
        logging.info("Skipping leg IK as it is a legless fly")


if __name__ == "__main__":

    # main_dir = '/mnt/nas2/GO/7cam/221221_aJO-GAL4xUAS-CsChr/Fly001'
    main_dirs = [
        '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly002',
    ]

    paths_to_run_ik = []

    for main_dir in main_dirs:
        paths_to_run_ik += list(Path(main_dir).rglob('pose-3d'))

    then = time.time()

    # Max number of cores
    with multiprocessing.Pool(processes=NO_CORES) as pool:
        pool.map(
            worker_wrapper,
            [
                (
                    str(path)
                )
                for path in paths_to_run_ik
            ]
        )

    now = time.time()
    print("Time taken: ", now - then)
