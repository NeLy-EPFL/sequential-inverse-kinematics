""" Makes a grid video of 3D pose estimation, joint angles, and the fly recording."""
import shutil
import argparse
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nmf_ik.visualization import (get_frames_from_video,
                                  get_frames_from_video_ffmpeg,
                                  load_grid_plot_data,
                                  plot_grid)


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

    DATA_PATH = Path(
        args.data_path
    )

    VIDEO_PATH = Path(
        args.video_path)

    FRAMES_PATH = Path(str(VIDEO_PATH).replace('.mp4', '_frames'))

    if not FRAMES_PATH.is_dir():
        get_frames_from_video(VIDEO_PATH)

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

    leg_angles_to_plot = [f"Angle_LF_{ja}" for ja in angles]
    head_angles_to_plot = [
        "Angle_head_roll",
        "Angle_head_pitch",
        "Angle_head_yaw",
        "Angle_antenna_pitch_L",
        "Angle_antenna_pitch_R",
    ]

    (DATA_PATH / 'grid_video').mkdir()

    t_start = args.time_start
    t_end = args.time_end
    fps = args.fps

    for t in range(t_start, t_end):
        fig = plot_grid(
            img_path=FRAMES_PATH,
            aligned_pose=points_aligned_all,
            joint_angles=joint_angles,
            leg_angles_to_plot=leg_angles_to_plot,
            head_angles_to_plot=head_angles_to_plot,
            key_points_3d=KEY_POINTS_DICT,
            key_points_3d_trail=KEY_POINTS_TRAIL,
            t=t,
            t_start=t_start,
            t_end=t_end,
            fps=fps,
            trail=30,
            export_path=DATA_PATH / 'grid_video' / f'frame_{t}.png',
        )

        plt.clf()
        # fig.savefig(DATA_PATH / 'grid_video' / f'frame_{t}.png')

    export_path = str(DATA_PATH / 'grid_video.mp4') if args.export_path is None \
        else args.export_path

    cmd = ['ffmpeg', '-r', str(fps), '-pattern_type', 'glob',
           '-i', str(DATA_PATH / 'grid_video' / '*.png'),
           '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
           '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
           '-y', export_path]

    subprocess.run(cmd, check=True)
    shutil.rmtree(DATA_PATH / 'grid_video')

    # plt.show()
