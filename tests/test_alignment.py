import pytest
from pathlib import Path
import numpy as np

from seqikpy.alignment import AlignPose

PKG_PATH = Path(seqikpy.__path__[0])

@pytest.fixture
def main_folder():
    return PKG_PATH / 'data/anipose_pose-220807_aJO-GAL4xUAS-CsChr_Fly002_002_Beh'


@pytest.fixture
def main_folder_exc():
    return '../data/anipose/normal_case/'


def test_pose_file(main_folder):
    align = AlignPose(main_folder)
    assert align.pose_result_path == PKG_PATH / 'data/anipose_pose-220807_aJO-GAL4xUAS-CsChr_Fly002_002_Beh/pose_3d.h5'


def test_pose_file_exc(main_folder_exc):
    with pytest.raises(FileNotFoundError):
        align = AlignPose(main_folder_exc)


def test_converted_dict(main_folder):
    align = AlignPose(main_folder)
    kp_names = {'RF_leg', 'LF_leg', 'L_head', 'R_head', 'Thorax'}

    assert kp_names == set(align.pose_data_dict.keys())
    assert align.pose_data_dict['RF_leg'].shape == (6000, 5, 3)
    assert align.pose_data_dict['LF_leg'].shape == (6000, 5, 3)
    assert align.pose_data_dict['Thorax'].shape == (6000, 3, 3)
    assert align.pose_data_dict['R_head'].shape == (6000, 2, 3)
    assert align.pose_data_dict['L_head'].shape == (6000, 2, 3)


def test_trial_length(main_folder):
    align = AlignPose(main_folder)
    assert 6000 == align.trial_length


@pytest.mark.parametrize('leg_name', ['RF_leg', 'LF_leg'])
def test_mean_length(main_folder, leg_name):
    align = AlignPose(main_folder)
    mean_length = align.get_mean_length(align.pose_data_dict[leg_name], True)

    correct_mean_length = {
        'RF_leg':
            {
                'coxa': 0.3346347368692227,
                'femur': 0.6652300069548103,
                'tibia': 0.5083878473974697,
                'tarsus': 0.554267448990312
            },
        'LF_leg':
            {
                'coxa': 0.3042863816867363,
                'femur': 0.6616840367058072,
                'tibia': 0.5101094810133566,
                'tarsus': 0.5423598868556228
            }
    }

    assert mean_length == correct_mean_length[leg_name]


@pytest.mark.parametrize('leg_name', ['RF_leg', 'LF_leg'])
def test_scale_factor(main_folder, leg_name):
    align = AlignPose(main_folder)
    mean_length = align.get_mean_length(align.pose_data_dict[leg_name], True)
    scale_factor = align.find_scale_leg(leg_name[:2], mean_length)
    scale = {
        'RF_leg': 1.0674604568032562,
        'LF_leg': 1.0907268642083676
    }
    assert np.isclose(scale_factor, scale[leg_name])



@pytest.mark.parametrize('segment_name', ['R_head', 'L_head'])
def test_scale_factor(main_folder, segment_name):
    align = AlignPose(main_folder)
    aligned_head_class = align.align_head(align.pose_data_dict[segment_name], segment_name[0])

    aligned_head = np.load('antenna.npy')
    array = aligned_head[:,:2,:] if segment_name == 'R' else aligned_head[:,2:,:]

    # assert np.allclose(aligned_head_class[:,0,:], array[:,0,:], atol=1e-1)
    assert np.allclose(aligned_head_class[:,1,:], array[:,1,:], rtol=1e-1, atol=1e-1)
