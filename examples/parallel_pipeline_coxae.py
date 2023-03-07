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

    if (Path(path) / "body_joint_angles.pkl").is_file():
        logging.info("Joint angles exist, deleting!!")
        (Path(path) / "body_joint_angles.pkl").unlink()

    align = AlignPose(path, nmf_template=NMF_TEMPLATE, pts2align=PTS2ALIGN)
    aligned_pos = align.align_pose(
        save_pose_file=True,
    )

    class_hk = HeadInverseKinematics(
        aligned_pos=aligned_pos,
        nmf_template=NMF_TEMPLATE,
    )
    head_joint_angles = class_hk.compute_head_angles(export_path=path)

    if 'RLF' in path:
        logging.info("Running leg IK for only one stage")
        stages = [1]
    else:
        logging.info("Running leg IK for all stages")
        stages = range(1, 5)

    class_seq_ik = LegInverseKinematics(
        aligned_pos=aligned_pos,
        nmf_template=NMF_TEMPLATE,
        bounds=BOUNDS,
        initial_angles=INITIAL_ANGLES
    )
    leg_joint_angles, forward_kinematics = class_seq_ik.run_ik_and_fk(export_path=path, stages=stages)

    full_body_ik = {**head_joint_angles, **leg_joint_angles}

    save_file(Path(path) / "body_joint_angles.pkl", full_body_ik)


if __name__ == "__main__":

    # main_dir = '/mnt/nas2/GO/7cam/221221_aJO-GAL4xUAS-CsChr/Fly001'
    main_dirs = [
        '/mnt/nas2/GO/7cam/221219_aJO-GAL4xUAS-CsChr/Fly001',
        '/mnt/nas2/GO/7cam/221219_aJO-GAL4xUAS-CsChr/Fly002',
        '/mnt/nas2/GO/7cam/221219_aJO-GAL4xUAS-CsChr/Fly003',
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
