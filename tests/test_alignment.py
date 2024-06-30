""" Test alignment tools """
import pytest
from pathlib import Path
import numpy as np

import seqikpy
from seqikpy.alignment import AlignPose, convert_from_anipose_to_dict

PKG_PATH = Path(seqikpy.__path__[0])

@pytest.fixture
def main_folder():
    return PKG_PATH / '../data/anipose_220525_aJO_Fly001_001/pose-3d'


@pytest.fixture
def main_folder_exc():
    return '../data/anipose/normal_case/'

def test_pose_file_exc(main_folder_exc):
    with pytest.raises(FileNotFoundError):
        align = AlignPose.from_file_path(
            main_dir=main_folder_exc,
            legs_list=["RF", "LF"],
        )


def test_converted_dict(main_folder):
    align = AlignPose.from_file_path(
        main_dir=main_folder,
        legs_list=["RF", "LF"],
        convert_func=convert_from_anipose_to_dict,
    )
    kp_names = {'RF_leg', 'LF_leg', 'L_head', 'R_head', 'Thorax'}

    assert kp_names == set(align.pose_data_dict.keys())
    assert align.pose_data_dict['RF_leg'].shape == (6000, 5, 3)
    assert align.pose_data_dict['LF_leg'].shape == (6000, 5, 3)
    assert align.pose_data_dict['Thorax'].shape == (6000, 3, 3)
    assert align.pose_data_dict['R_head'].shape == (6000, 2, 3)
    assert align.pose_data_dict['L_head'].shape == (6000, 2, 3)


@pytest.mark.parametrize('leg_name', ['RF_leg', 'LF_leg'])
def test_scale_factor(main_folder, leg_name):
    align = AlignPose.from_file_path(
        main_dir=main_folder,
        legs_list=["RF", "LF"],
        convert_func=convert_from_anipose_to_dict,
    )
    mean_length = align.get_mean_length(align.pose_data_dict[leg_name], True)
    # check mean length
    correct_mean_length = {
        'RF_leg':
            {
                'coxa': 0.33463473686922274,
                'femur': 0.6652300069548103,
                'tibia': 0.5083878473974696,
                'tarsus': 0.5542674489903121
            },
        'LF_leg':
            {
                'coxa': 0.3042863816867363,
                'femur': 0.6616840367058072,
                'tibia': 0.5101094810133566,
                'tarsus': 0.5423598868556229
            }
    }
    assert mean_length == correct_mean_length[leg_name]
    # check scale factor
    scale_factor = align.find_scale_leg(leg_name[:2], mean_length)
    scale = {
        'RF_leg': 1.0807208351486381,
        'LF_leg': 1.1042762662482228
    }
    assert np.isclose(scale_factor, scale[leg_name])



@pytest.mark.parametrize('segment_name', ['R_head', 'L_head'])
def test_scale_factor(main_folder, segment_name):
    align = AlignPose.from_file_path(
        main_dir=main_folder,
        legs_list=["RF", "LF"],
        convert_func=convert_from_anipose_to_dict,
    )
    aligned_head_kin = align.align_head(align.pose_data_dict[segment_name], segment_name[0])

    aligned_head = np.load('antenna.npy')
    ground_truth = aligned_head[:,:2,:] if segment_name[0] == 'R' else aligned_head[:,2:,:]

    assert np.allclose(aligned_head_kin, ground_truth)


def test_export(main_folder):
    align = AlignPose.from_file_path(
        main_dir=main_folder,
        legs_list=["RF", "LF"],
        convert_func=convert_from_anipose_to_dict,
    )
    tmp_path = Path("./")
    aligned_pos = align.align_pose(export_path=tmp_path)
    # check if exists
    assert (tmp_path / "pose3d_aligned.pkl").exists()
    # remove
    (tmp_path / "pose3d_aligned.pkl").unlink()
