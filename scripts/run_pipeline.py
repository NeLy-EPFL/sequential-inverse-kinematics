""" Run the entire pipeline automatically. """
from pathlib import Path
from typing import Dict
from nptyping import NDArray
import numpy as np
import pandas as pd

from nmf_ik.alignment import AlignPose
from nmf_ik.leg_inverse_kinematics import LegInverseKinematics
from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
from nmf_ik.data import BOUNDS, INITIAL_ANGLES, NMF_SIZE, NMF_TEMPLATE


def run_pipeline(aligned_pos, f_path: Path, **kwargs) -> Dict[str, NDArray]:
    """ Runs the entire alignment and IK pipeline on a file. """
    is_head_ik = kwargs.get('is_head_ik', True)
    is_leg_ik = kwargs.get('is_leg_ik', True)

    head_joint_angles = {}
    leg_joint_angles = {}

    # align = AlignPose(f_path)
    # aligned_pos = align.align_pose(save_pose_file=True)

    if is_leg_ik:
        class_seq_ik = LegInverseKinematics(
            aligned_pos=aligned_pos,
            nmf_template=NMF_TEMPLATE,
            nmf_size=NMF_SIZE,
            bounds=BOUNDS,
            initial_angles=INITIAL_ANGLES
        )
        leg_joint_angles, _ = class_seq_ik.run_ik_and_fk(export_path=f_path)

    if is_head_ik:
        class_hk = HeadInverseKinematics(
            aligned_pos=aligned_pos,
            nmf_template=NMF_TEMPLATE,
            angles_to_calculate=[
                'Angle_head_roll',
                'Angle_head_pitch',
                'Angle_head_yaw',
                'Angle_antenna_pitch_L',
                'Angle_antenna_pitch_R',
                'Angle_antenna_yaw_L',
                'Angle_antenna_yaw_R'
            ]
        )
        head_joint_angles = class_hk.compute_head_angles(export_path=f_path)

    if is_leg_ik or is_head_ik:
        full_body_ik = {**head_joint_angles, **leg_joint_angles}

        return full_body_ik

    return aligned_pos


def convert_pp2anipose(pp_dict):
    converted_dict = {
        'RF_leg': np.concatenate(
            (np.tile(
                pp_dict["RF_leg"]['Coxa']['fixed_pos_aligned'],
                (pp_dict["RF_leg"]['Coxa']['raw_pos_aligned'].shape[0],
                 1)).reshape(-1, 1, 3),
             np.array(pp_dict["RF_leg"]['Femur']['raw_pos_aligned']).reshape(-1, 1, 3),
             np.array(pp_dict["RF_leg"]['Tibia']['raw_pos_aligned']).reshape(-1, 1, 3),
             np.array(pp_dict["RF_leg"]['Tarsus']['raw_pos_aligned']).reshape(-1, 1, 3),
             np.array(pp_dict["RF_leg"]['Claw']['raw_pos_aligned']).reshape(-1, 1, 3),),
            axis=1),
        'LF_leg': np.concatenate(
            (np.tile(
                pp_dict["LF_leg"]['Coxa']['fixed_pos_aligned'],
                (pp_dict["RF_leg"]['Coxa']['raw_pos_aligned'].shape[0],
                 1)).reshape(-1, 1, 3),
             np.array(pp_dict["LF_leg"]['Femur']['raw_pos_aligned']).reshape(-1, 1, 3),
             np.array(pp_dict["LF_leg"]['Tibia']['raw_pos_aligned']).reshape(-1, 1, 3),
             np.array(pp_dict["LF_leg"]['Tarsus']['raw_pos_aligned']).reshape(-1, 1, 3),
             np.array(pp_dict["LF_leg"]['Claw']['raw_pos_aligned']).reshape(-1, 1, 3),),
            axis=1), }

    return converted_dict


def run_txt_file(path_txt: Path):
    """ Runs the entire pipeline on the dirs from a txt. """
    path_list = []
    for line in open(path_txt):
        p_name = Path(line.rstrip())
        parts = p_name.parts
        if len(parts) < 8:
            raise IndexError(
                """
                Directory should have at least 8 parts.
                Example: ('/','mnt','nas','GO','7cam','220504_aJO-GAL4xUAS-CsChr', 'Fly001', '002_Beh')
                """
            )
        # If child directory is given, we take the parent directory.
        if parts[0] == "/":
            new_parts = parts[:8]
        else:
            raise ValueError(f"Directory {p_name} is faulty!\nPlease, check it again.")

        new_path = Path(*new_parts) / 'behData/pose-3d'
        print(new_path)
        path_list.append(new_path)

    for p_name in np.unique(path_list):
        if not p_name.is_dir():
            print(f'{p_name} is not a directory yet, skipping...')
            continue
        joint_angles = run_pipeline(p_name, is_head_ik=False, is_leg_ik=False)


if __name__ == '__main__':
    # path_txt_f = Path('bout_file_gizem.txt')
    # run_txt_file(path_txt_f)

    main_dir = Path('/Volumes/data2/VAS/data_matching_bouts')
    file_names = list(main_dir.glob('aligned_pose*.pkl'))
    not_processed = []
    for fname in file_names:
        print(fname)
        pp_dict = pd.read_pickle(fname)
        try:
            converted_dict = convert_pp2anipose(pp_dict)
        except BaseException:
            not_processed.append(fname)
            continue
        run_pipeline(converted_dict, fname.as_posix(), is_leg_ik=True, is_head_ik=False)
        # from IPython import embed; embed()
    print(not_processed)
