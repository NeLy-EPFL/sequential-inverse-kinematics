"""
    Makes a grid video of 3D pose estimation, joint angles, and the fly recording on a given trial.
    The data is plotted from 0.5 seconds before to after the stimulation. This script makes a video for each stimulation.
    The video is saved in the same directory as the pose data if export_path is not provided.

    Example usage:
    >>> python make_grid_video.py --data_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d' --video_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/videos/camera_3.mp4' --export_path '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d'

"""
import argparse
from pathlib import Path
import numpy as np

import utils_video
from nmf_ik.visualization import (video_frames_generator,
                                  plot_grid_generator,
                                  load_grid_plot_data)
from nmf_ik.utils import (load_stim_data, get_stim_intervals, get_stim_array, get_fps_from_video)

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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    DATA_PATH = Path(
        args.data_path
    )

    VIDEO_PATH = Path(
        args.video_path)

    fps = get_fps_from_video(VIDEO_PATH)

    stim_data = load_stim_data(DATA_PATH.parents[0] / 'StimulusData')
    stim_intervals = get_stim_intervals(get_stim_array(stim_data, fps))

    stim_beg_end = [(stim_intervals[i], stim_intervals[i+1] + 1) for i in range(0, len(stim_intervals), 2)]

    # from IPython import embed; embed()

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

    video_name = Path(*DATA_PATH.parts[5:-2]) # 221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh'

    for stim_ind, (stim_start, stim_end) in enumerate(stim_beg_end):

        print('Stimulation number: ', stim_ind)

        t_start = stim_start - int(fps/2)
        t_end = stim_end + int(fps/2)

        # Frames generator
        fly_frames = video_frames_generator(VIDEO_PATH, t_start, t_end)

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
            t_interval=fps,
            fps=fps,
            trail=int(fps/4),
            stim_lines=(stim_start, stim_end),
            export_path=None
        )

        out_video_name = str(video_name).replace('/','_') + f'_stim_{stim_ind}_grid_video.mp4'

        export_path = str(DATA_PATH / out_video_name) if args.export_path is None \
            else str(args.export_path / out_video_name)

        utils_video.make_video(
            export_path,
            generator,
            fps=fps,
            n_frames=t_end - t_start)

        generator = 0
        print('Video saved at ', export_path)
