"""
    Parallel running of the pipeline.
"""
import logging
from pathlib import Path
import time
import multiprocessing

import nmf_ik
from nmf_ik.alignment import AlignPose
from nmf_ik.leg_inverse_kinematics import LegInverseKinematics
from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
from nmf_ik.data import BOUNDS, INITIAL_ANGLES, NMF_TEMPLATE, get_pts2align
from nmf_ik.utils import save_file

# Change the logging level here
logging.basicConfig(level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s")

NO_CORES = multiprocessing.cpu_count()


def worker_wrapper(arg):
    """ Provide kwargs during multiprocessing. """
    return run_pipeline(arg)


def run_pipeline(path):

    logging.info("Running code in %s", path)

    PTS2ALIGN = get_pts2align(str(path))

    align = AlignPose(path, nmf_template=NMF_TEMPLATE, pts2align=PTS2ALIGN)
    aligned_pos = align.align_pose(
        save_pose_file=True,
    )

    class_hk = HeadInverseKinematics(
        aligned_pos=aligned_pos,
        nmf_template=NMF_TEMPLATE,
    )
    head_joint_angles = class_hk.compute_head_angles(export_path=path)

    # if (Path(path) / "body_joint_angles.pkl").is_file():
    #     logging.info("Joint angles exist!!")
    #     return

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
    print(nmf_ik.__path__)

    # main_dir = '/mnt/nas2/GO/7cam/221221_aJO-GAL4xUAS-CsChr/Fly001'
    main_dirs = [
        '/mnt/nas2/GO/7cam/220713_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/220713_aJO-GAL4xUAS-CsChr/Fly002',
        # '/mnt/nas2/GO/7cam/220713_aJO-GAL4xUAS-CsChr/Fly003',
        # '/mnt/nas2/GO/7cam/220713_aJO-GAL4xUAS-CsChr/Fly004',
        # '/mnt/nas2/GO/7cam/220713_aJO-GAL4xUAS-CsChr/Fly005',
        '/mnt/nas2/GO/7cam/220714_aJO-GAL4xUAS-CsChr',
        '/mnt/nas2/GO/7cam/220807_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/220807_aJO-GAL4xUAS-CsChr/Fly002',
        # '/mnt/nas2/GO/7cam/220807_aJO-GAL4xUAS-CsChr/Fly003',
        '/mnt/nas2/GO/7cam/220808_aJO-GAL4xUAS-CsChr',
        '/mnt/nas2/GO/7cam/220809_aJO-GAL4xUAS-CsChr',
        '/mnt/nas2/GO/7cam/220810_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/220809_aJO-GAL4xUAS-CsChr/Fly002',
        # '/mnt/nas2/GO/7cam/220809_aJO-GAL4xUAS-CsChr/Fly003',
        # '/mnt/nas2/GO/7cam/221219_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/221221_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/221222_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/221223_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/221226_aJO-GAL4xUAS-CsChr',
        # '/mnt/nas2/GO/7cam/221228_aJO-GAL4xUAS-CsChr',
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
