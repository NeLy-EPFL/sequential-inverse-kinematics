""" Makes a grid video of 3D pose estimation, joint angles, and the fly recording.

Example usage:
>>> python make_grid_video.py --data_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d' --video_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/videos/camera_3.mp4' --time_start 200 --time_end 600 --frame_rate 100

"""
import argparse
from pathlib import Path
import numpy as np

import utils_video
from nmf_ik.visualization import (video_frames_generator,
                                  plot_grid_generator,
                                  load_grid_plot_data)


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
        "-fps",
        "--frame_rate",
        type=int,
        default=100,
        help="Frame rate of the video",
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

    t_start = args.time_start
    t_end = args.time_end
    fps = args.frame_rate

    DATA_PATH = Path(
        args.data_path
    )

    VIDEO_PATH = Path(
        args.video_path)

    # Frames generator
    fly_frames = video_frames_generator(VIDEO_PATH, t_start, t_end)

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
        "RF": (np.arange(0, 5), "solid"),
        "R Ant": (np.arange(10, 12), "o"),
        "Neck": (np.arange(14, 15), "x"),
        "L Ant": (np.arange(12, 14), "o"),
        "LF": (np.arange(5, 10), "solid"),
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

    generator = plot_grid_generator(
        fly_frames=fly_frames,
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
        stim_lines=[250, 550],
        export_path=None
    )

    export_path = str(DATA_PATH / 'grid_video.mp4') if args.export_path is None \
        else args.export_path

    utils_video.make_video(
        export_path,
        generator,
        fps=fps,
        n_frames=t_end - t_start)
