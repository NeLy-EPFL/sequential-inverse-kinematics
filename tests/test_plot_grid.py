""" Plots the layout that will be a frame in the video. """

import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nmf_ik.visualization import (get_frames_from_video_ffmpeg,
                                  load_grid_plot_data,
                                  plot_grid,
                                  video_frames_generator)

if __name__ == '__main__':

    DATA_PATH = Path(
        '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d'
    )

    VIDEO_PATH_FRONT = Path(
        '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/videos/camera_3.mp4')

    VIDEO_PATH_SIDE = Path(
        '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/videos/camera_5.mp4')

    joint_angles, aligned_pose = load_grid_plot_data(DATA_PATH)

    points_aligned_all = np.concatenate(
        (
            aligned_pose["RF_leg"],
            aligned_pose["LF_leg"],
            aligned_pose["R_head"],
            aligned_pose["L_head"],
            np.tile(aligned_pose["Neck"], (aligned_pose["RF_leg"].shape[0], 1)).reshape(
                -1, 1, 3
            ),
        ),
        axis=1,
    )

    KEY_POINTS_DICT = {
        "Head roll": ([10,12], '.'),
        "Neck": (np.arange(14, 15), "x"),
        "RF": (np.arange(0, 5), "solid"),
        "R Ant": (np.arange(10, 12), "o"),
        "LF": (np.arange(5, 10), "solid"),
        "L Ant": (np.arange(12, 14), "o"),
    }

    KEY_POINTS_TRAIL = {
        "RF": (np.arange(3, 4), "x"),
        "LF": (np.arange(8, 9), "x"),
    }

    leg_joint_angles = [
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

    # (DATA_PATH / 'grid_video').mkdir()

    t_start = 300
    t_end = 600

    t = t_start + 290
    fps = 100
    stim_lines=[350,500]

    fly_frames_front = video_frames_generator(VIDEO_PATH_FRONT, t_start, t_end, stim_lines)
    fly_frames_side = video_frames_generator(VIDEO_PATH_SIDE, t_start, t_end, stim_lines)

    fig = plot_grid(
        img_front=list(fly_frames_front)[t - t_start],
        img_side=list(fly_frames_side)[t - t_start],
        aligned_pose=points_aligned_all,
        joint_angles=joint_angles,
        leg_angles_to_plot=leg_joint_angles,
        head_angles_to_plot=head_angles_to_plot,
        key_points_3d=KEY_POINTS_DICT,
        key_points_3d_trail=KEY_POINTS_TRAIL,
        t=t,
        t_start=t_start,
        t_end=t_end,
        fps=fps,
        trail=30,
        t_interval=100,
        stim_lines=stim_lines,
        export_path=DATA_PATH / f'frame_{t}_alpha1.2_beta_0.png',
    )

    plt.show()