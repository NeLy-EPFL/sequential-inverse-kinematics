""" Makes a grid video of 3D pose estimation, joint angles, and the fly recording.

Example usage:
>>> python make_grid_video.py --data_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d' --video_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/videos' --time_start 200 -export_path /Volumes/data2/GO/grid_videos --time_end 600

"""
import argparse
from pathlib import Path
import numpy as np

import utils_video
from nmf_ik.visualization import (video_frames_generator,
                                  plot_grid_generator,
                                  load_grid_plot_data)
from nmf_ik.utils import get_fps_from_video


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline to visualize camera positions",
    )
    parser.add_argument(
        "-data",
        "--data_path",
        type=str,
        default=None,
        help="Path of pose files",
    )
    parser.add_argument(
        "-video",
        "--video_path",
        type=str,
        default=None,
        help="Path of video files",
    )
    parser.add_argument(
        "-export",
        "--export_path",
        type=str,
        default=None,
        help="Path where the grid video to be saved",
    )
    parser.add_argument(
        "-ts",
        "--time_start",
        type=int,
        default=0,
        help="Start time of the grid video",
    )
    parser.add_argument(
        "-te",
        "--time_end",
        type=int,
        default=0,
        help="End time of the grid video",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DATA_PATH = Path(
        args.data_path
    )

    VIDEO_PATH_FRONT = Path(
        args.video_path) / 'camera_3.mp4'

    VIDEO_PATH_SIDE = Path(
        args.video_path) / 'camera_5.mp4'

    t_start = args.time_start
    t_end = args.time_end
    fps = get_fps_from_video(VIDEO_PATH_FRONT)

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

    angles = [
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

    stim_lines = [250, 550]

    fly_frames_front = video_frames_generator(VIDEO_PATH_FRONT, t_start, t_end, stim_lines)
    fly_frames_side = video_frames_generator(VIDEO_PATH_SIDE, t_start, t_end, stim_lines)

    generator = plot_grid_generator(
        fly_frames_front=fly_frames_front,
        fly_frames_side=fly_frames_side,
        aligned_pose=points_aligned_all,
        joint_angles=joint_angles,
        leg_angles_to_plot=angles,
        head_angles_to_plot=head_angles_to_plot,
        key_points_3d=KEY_POINTS_DICT,
        key_points_3d_trail=KEY_POINTS_TRAIL,
        t_start=t_start,
        t_end=t_end,
        t_interval=100,
        fps=fps,
        trail=30,
        stim_lines=stim_lines,
        export_path=None
    )

    export_path = str(DATA_PATH / 'grid_video.mp4') if args.export_path is None \
        else args.export_path

    utils_video.make_video(
        export_path,
        generator,
        fps=fps,
        n_frames=t_end - t_start)
