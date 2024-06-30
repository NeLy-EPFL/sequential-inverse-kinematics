""" Test visualization tools """
import pytest

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from seqikpy.visualization import (
    load_grid_plot_data,
    get_plot_config,
    plot_grid,
    video_frames_generator)


def test_plot_config():
    # Load the data - right left legs are amptuated until coxa
    data_path = Path("./Fly001/001_RLF_coxa/behData/pose-3d")
    # plot config determines which body parts to plot
    exp_type, plot_config = get_plot_config(data_path)
    assert exp_type == "coxa"
    assert plot_config['plot_head'] == True
    assert plot_config['plot_right_leg'] == True
    assert plot_config['plot_left_leg'] == True
    # Load the data - right left legs are completely amputated
    data_path = Path("./Fly001/001_RLF/behData/pose-3d")
    exp_type, plot_config = get_plot_config(data_path)
    assert exp_type == "RLF"
    assert plot_config['plot_head'] == True
    assert plot_config['plot_right_leg'] == False
    assert plot_config['plot_left_leg'] == False
    # Load the data - right leg amputated
    data_path = Path("./Fly001/001_RF/behData/pose-3d")
    exp_type, plot_config = get_plot_config(data_path)
    assert exp_type == "RF"
    assert plot_config['plot_head'] == True
    assert plot_config['plot_right_leg'] == False
    assert plot_config['plot_left_leg'] == True
    # Load the data - intact
    data_path = Path("./Fly001/001_Beh/behData/pose-3d")
    exp_type, plot_config = get_plot_config(data_path)
    assert exp_type == "Beh"
    assert plot_config['plot_head'] == True
    assert plot_config['plot_right_leg'] == True
    assert plot_config['plot_left_leg'] == True
    # Check error with a wrong path name
    data_path = Path("./Fly001/001_Beh/pose-3d")
    with pytest.raises(Exception):
        exp_type, plot_config = get_plot_config(data_path)


def test_grid_plot():
    # Set up the constant variables
    leg_joint_angle_names = [
        "ThC_yaw",
        "ThC_pitch",
        "ThC_roll",
        "CTr_pitch",
        "CTr_roll",
        "FTi_pitch",
        "TiTa_pitch",
    ]

    head_angles_to_plot = [
        "Angle_head_roll",
        "Angle_head_pitch",
        "Angle_head_yaw",
        "Angle_antenna_pitch_L",
        "Angle_antenna_pitch_R",
    ]

    # Load the data
    data_path = Path("../data/anipose_220807_Fly002_002")
    video_path_front = data_path / "camera_3.mp4"
    video_path_side = data_path / "camera_5.mp4"

    # plot config determines which body parts to plot
    # as our path is not standard, we need to specify plot config
    plot_config = {
        'plot_head': True,
        'plot_right_leg': True,
        'plot_left_leg': True,
    }

    # loads the joint angles and the aligned pos
    joint_angles, aligned_pose = load_grid_plot_data(data_path)

    # as neck pose is one dimensional, make it consistent with the other key points
    aligned_pose["Neck"] = np.tile(
        aligned_pose["Neck"],
        (aligned_pose["RF_leg"].shape[0], 1)
    ).reshape(-1, 1, 3)

    # Start, end of the plotting data
    t_start = 0
    t_end = 200

    # t: snapshot to show
    t = t_start + 100
    fps = 100

    # Stimulation applied? If so, when
    stim_lines = [50]

    # Ignore this, we put it because we cropped the data in the previous tutorial
    crop_time = 400

    fly_frames_front = video_frames_generator(
        video_path_front,
        t_start + crop_time,
        t_end + crop_time,
        stim_lines)
    fly_frames_side = video_frames_generator(
        video_path_side,
        t_start + crop_time,
        t_end + crop_time,
        stim_lines)

    fig = plot_grid(
        img_front=list(fly_frames_front)[t - t_start],
        img_side=list(fly_frames_side)[t - t_start],
        aligned_pose=aligned_pose,
        joint_angles=joint_angles,
        leg_angles_to_plot=leg_joint_angle_names,
        head_angles_to_plot=head_angles_to_plot,
        key_points_to_trail={'LF_leg': [3], 'RF_leg': [3]},
        marker_trail="x",
        t=t,
        t_start=t_start,
        t_end=t_end,
        fps=fps,
        trail=30,
        t_interval=100,
        stim_lines=stim_lines,
        export_path=f'generate_frame_{t}.png',
        **plot_config
    )
    # Load the ground truth and generated images
    img_ground_truth = plt.imread('test_frame_100.png')
    img_test = plt.imread(f'generate_frame_{t}.png')
    # Convert colors to compare in a more robust way
    img_ground_truth = np.where(img_ground_truth > 0, 1, 0)
    img_test = np.where(img_test > 0, 1, 0)

    assert np.allclose(img_ground_truth, img_test)
