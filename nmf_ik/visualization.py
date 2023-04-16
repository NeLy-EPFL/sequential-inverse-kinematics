""" Functions for plotting and animation. """
import pickle
import logging
import time
import subprocess
from pathlib import Path
from datetime import date
from typing import Tuple, List, Dict

import cv2
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mpl_toolkits.mplot3d.axes3d as p3

# Change the logging level here
logging.basicConfig(
    level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s"
)


def video_frames_generator(video_path: Path, start_frame: int, end_frame: int):
    """ Returns the frames as a generator in a given interval.
    Modifies the brightness and contrast of the images.

    Parameters
    ----------
    video_path : Path
        Video path.
    start_frame : int
        Starting frame.
    end_frame : int
        End frame.

    Yields
    ------
    Frame
        Generator containing all the frames in the specified interval.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the contrast and brightness values
    alpha = 1.15  # Contrast control
    beta = -5  # Brightness control

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        if not ret:
            break
        yield adjusted_frame

    cap.release()


def get_frames_from_video_ffmpeg(path):
    """ Saves frames of a video using FFMPEG.

    Parameters
    ----------
    path : Path
        Video path.
        This path appended with a `_frames` folder will be used to save the frames.
    """
    write_path = path.parents[0] / str(path.name).replace('.mp4', '_frames')
    write_path.mkdir()
    cmd = ['ffmpeg', '-i', str(path), '-r', '1', str(write_path / 'frame_%d.jpg')]
    subprocess.run(cmd, check=True)


def load_grid_plot_data(data_path: Path) -> [Dict, Dict]:
    """ Loads the set of data necessary for plotting the grid.

    Parameters
    ----------
    data_path : Path
        Data path where the pose3d and inverse kinematics are saved.

    Returns
    -------
    Tuple
        Returns joint angles (head and leg) and aligned pose as a tuple.
    """
    if (data_path / "body_joint_angles.pkl").is_file():
        joint_angles = pd.read_pickle(data_path / "body_joint_angles.pkl")
    else:
        head_joint_angles = pd.read_pickle(data_path / "head_joint_angles.pkl")
        leg_joint_angles = pd.read_pickle(data_path / "leg_joint_angles.pkl")
        joint_angles = {**head_joint_angles, **leg_joint_angles}
    aligned_pose = pd.read_pickle(data_path / "pose3d_aligned.pkl")

    return joint_angles, aligned_pose


def animate_3d_points(
    points3d: Dict[str, np.ndarray],
    key_points: Dict[str, Tuple[np.ndarray, str]],
    export_path: Path,
    points3d_second: Dict[str, np.ndarray] = None,
    key_points_second: Dict[str, Tuple[np.ndarray, str]] = None,
    fps: int = 100,
    frame_no: int = 1000,
    format_video: str = "mp4",
    elev: int = 10,
    azim: int = 90,
):
    """ Makes an animation of 3D pose.
    This code is intended for animating the raw 3D pose and
    IK based 3D pose.

    Parameters
    ----------
    points3d : Dict[str,np.ndarray]
        Dictionary containing the 3D pose,
        usually this is the raw 3D pose.
    key_points : Dict[str, Tuple]
        Dictionary mapping key points names to their indices
        and line styles.
        Example:
            KEY_POINTS_DICT = {
                "RF": (np.arange(0, 5), "solid"),
                "R Ant": (np.arange(10, 12), "o"),
                "Neck": (np.arange(14, 15), "x"),
                "L Ant": (np.arange(12, 14), "o"),
                "LF": (np.arange(5, 10), "solid"),
            }
    export_path : Path
        Path where the animation will be saved.
    points3d_second : Dict[str,np.ndarray], optional
        Dictionary containing the 3D pose
        usually this is the forward kinematics, by default None
    key_points_second : Dict[str, Tuple, optional
        Same as `key_points` for `points3d_second`, by default None
    fps : int, optional
        Frames per second, by default 100
    frame_no : int, optional
        Frame number until which the animation will be recorded,
        by default 1000
    format_video : str, optional
        Format of the video, by default "mp4"
    elev : int, optional
        Elevation of 3D axis, by default 10
    azim : int, optional
        Azimuth of 3D axis, by default 90
    """
    # Dark background
    plt.rcParams.update({
        "axes.facecolor": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})

    fig = plt.figure()
    ax3d = p3.Axes3D(fig)
    ax3d.view_init(azim=azim, elev=elev)
    # First remove fill
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = True

    # Now set color to white (or whatever is "invisible")
    ax3d.xaxis.pane.set_edgecolor('black')
    ax3d.yaxis.pane.set_edgecolor('black')
    ax3d.zaxis.pane.set_edgecolor('black')

    color_map_right = mcp.gen_color(cmap="Reds", n=len(key_points))
    color_map_left = mcp.gen_color(cmap="Blues", n=len(key_points))
    color_map_scatter = mcp.gen_color(cmap="RdBu", n=len(key_points))

    i, j, k = 1, 1, 1
    line_data = []
    line_data_second = []

    for kp, (order, ls) in key_points.items():
        if len(order) > 3:
            if 'L' in kp:
                color = color_map_left[j]
                j += 1
            else:
                color = color_map_right[k]
                k += 1
            line_data.append(
                ax3d.plot(
                    points3d[0, order, 0],
                    points3d[0, order, 1],
                    points3d[0, order, 2],
                    label=kp,
                    linestyle=ls,
                    linewidth=4,
                    color=color,
                    alpha=0.85,
                )[0]
            )
        else:
            line_data.append(
                ax3d.plot(
                    points3d[0, order, 0],
                    points3d[0, order, 1],
                    points3d[0, order, 2],
                    lw=2.5,
                    label=kp,
                    marker=ls,
                    markersize=9,
                    color=color_map_scatter[i],
                    alpha=0.7,
                )[0]
            )
        i += 1

    i, j, k = 1, 1, 1

    if points3d_second is not None:

        key_points_second = key_points if key_points_second is None else key_points_second
        for kp, (order, ls) in key_points_second.items():
            if len(order) > 4:
                if 'L' in kp:
                    color = color_map_left[j]
                    j += 1
                else:
                    color = color_map_right[k]
                    k += 1
                line_data_second.append(
                    ax3d.plot(
                        points3d_second[0, order, 0],
                        points3d_second[0, order, 1],
                        points3d_second[0, order, 2],
                        label=kp,
                        linestyle=ls,
                        linewidth=4,
                        color=color,
                        alpha=0.85,
                    )[0]
                )
            else:
                line_data_second.append(
                    ax3d.plot(
                        points3d_second[0, order, 0],
                        points3d_second[0, order, 1],
                        points3d_second[0, order, 2],
                        lw=2.5,
                        label=kp,
                        marker=ls,
                        markersize=9,
                        color=color_map_scatter[j],
                        alpha=0.7,
                    )[0]
                )

    # Setting the axes properties
    # ax3d.set_xlim((-0.0, 1.5))
    # ax3d.set_ylim((-1., 1.))
    # ax3d.set_zlim((0.2, 1.8))

    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    ax3d.tick_params(axis='x', color='black')
    ax3d.tick_params(axis='y', color='black')
    ax3d.tick_params(axis='z', color='black')
    # ax3d.set_axis_off('z')

    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])

    # ax3d.set_xlabel("x")
    # ax3d.set_ylabel("y")
    # ax3d.set_zlabel("z")
    # ax3d.set_title('DLC and DF3D Results')
    # ax3d.legend(bbox_to_anchor=(1.2, 0.9), frameon=False)
    def update(frame, lines, key_points, lines_second, key_points_second):
        i = 0
        for kp, (kp_range, _) in key_points.items():
            lines[i].set_data(
                points3d[frame, kp_range, 0], points3d[frame, kp_range, 1]
            )
            lines[i].set_3d_properties(points3d[frame, kp_range, 2])
            i += 1
        if lines_second:
            j = 0
            for kp, (kp_range, _) in key_points_second.items():
                lines_second[j].set_data(
                    points3d_second[frame, kp_range, 0], points3d_second[frame, kp_range, 1]
                )
                lines_second[j].set_3d_properties(points3d_second[frame, kp_range, 2])
                j += 1

    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig,
        update,
        frame_no,
        fargs=(
            line_data,
            key_points,
            line_data_second,
            key_points_second),
        interval=10,
        blit=False)
    logging.info("Making animation...")
    export_path = str(export_path)
    export_path += (
        f".{format_video}" if not export_path.endswith((".mp4", ".avi", ".mov")) else ""
    )
    line_ani.save(export_path, fps=fps, dpi=300)
    logging.info(f"Animation is saved at {export_path}")


def _plot_3d_points(points3d, key_points, export_path=None, t=0):
    """Plots 3D points."""
    fig = plt.figure(figsize=(5, 5))
    ax3d = p3.Axes3D(fig)
    ax3d.view_init(azim=0, elev=16)

    color_map_right = mcp.gen_color(cmap="RdPu", n=len(key_points))
    color_map_left = mcp.gen_color(cmap="BuGn", n=len(key_points))
    color_map_scatter = mcp.gen_color(cmap="gist_rainbow_r", n=len(key_points))

    i = 0
    for kp, (order, ls) in key_points.items():
        if len(order) > 3:
            ax3d.plot(
                points3d[t, order, 0],
                points3d[t, order, 1],
                points3d[t, order, 2],
                label=kp,
                linestyle=ls,
                linewidth=3,
                color=color_map_right[i] if "R" in kp else color_map_left[-i - 1],
            )
        else:
            ax3d.plot(
                points3d[t, order, 0],
                points3d[t, order, 1],
                points3d[t, order, 2],
                label=kp,
                marker=ls,
                markersize=9,
                color=color_map_scatter[i],
            )
        i += 1

    # Setting the axes properties
    #     ax3d.set_xlim3d([np.amin(points3d[..., 0]), np.amax(points3d[..., 0])])
    #     ax3d.set_ylim3d([np.amin(points3d[..., 1]), np.amax(points3d[..., 1])])
    #
    #     ax3d.set_xlim3d([-2, 1])
    #     ax3d.set_ylim3d([-1, 1])
    #     ax3d.set_zlim3d([-2, 0.1])
    ax3d.set_xlim3d([-1, 1])
    ax3d.set_ylim3d([-1, 1])
    ax3d.set_zlim3d([-2, 0.1])
    # ax3d.set_xticks([])
    # ax3d.set_yticks([])
    # ax3d.set_zticks([])
    # ax3d.invert_zaxis()

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend(bbox_to_anchor=(1.0, 0.9), ncol=2, frameon=False)

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")

    plt.show()


def plot_3d_points(ax3d, points3d, key_points, export_path=None, t=0):
    """Plots 3D points at time t."""

    color_map_right = mcp.gen_color(cmap="Reds", n=len(key_points) + 1)
    color_map_left = mcp.gen_color(cmap="Blues", n=len(key_points) + 1)
    color_map_scatter = mcp.gen_color(cmap="RdBu", n=len(key_points) + 1)

    i, j, k = 1, 1, 1

    for kp, (order, ls) in key_points.items():
        if 'R' in kp:
            color = color_map_right[i]
            i += 1
        elif 'L' in kp:
            color = color_map_left[j]
            j += 1
        else:
            color = 'grey'

        if len(order) > 3:
            ax3d.plot(
                points3d[t, order, 0],
                points3d[t, order, 1],
                points3d[t, order, 2],
                label=kp,
                linestyle=ls,
                linewidth=2,
                color=color,
            )
        else:
            ax3d.plot(
                points3d[t, order, 0],
                points3d[t, order, 1],
                points3d[t, order, 2],
                label=kp,
                marker=ls,
                markersize=7,
                color=color,
            )

    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight")


def plot_trailing_kp(ax3d, points3d, key_points, export_path=None, t=0, trail=5):
    """ Plots the traces of key points from t-trail to t. """

    color_map_right = mcp.gen_color(cmap="Reds", n=len(key_points) + 1)
    color_map_left = mcp.gen_color(cmap="Blues", n=len(key_points) + 1)

    i, j = 1, 1
    for kp, (order, ls) in key_points.items():
        if 'R' in kp:
            color = color_map_right[i]
            i += 1
        elif 'L' in kp:
            color = color_map_left[j]
            j += 1
        else:
            color = 'grey'

        ax3d.scatter(
            points3d[max(0, t - trail):t, order, 0],
            points3d[max(0, t - trail):t, order, 1],
            points3d[max(0, t - trail):t, order, 2],
            label=kp,
            marker=ls,
            #             markersize=9,
            color=color,
        )

    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight")


def plot_joint_angle(
    ax: plt.Axes,
    kinematics_data: Dict[str, np.ndarray],
    angles_to_plot: List,
    degrees: bool = True,
    until_t: int = -1,
    stim_lines: List[int] = None,
    show_legend: bool = True,
    export_path: Path = None
):
    """Plot joint angles from a given kinematics data.

    Parameters
    ----------
    ax : plt.Axes
        Axis where the plot will be displayed.
    kinematics_data : Dict[str, np.ndarray]
        Dictionary containing the kienmatics, pose and angle
    angles_to_plot : List
        Angles to plot. Exact column name should be given.
    degrees : bool, optional
        Convert to degrees, by default True
    until_t : int, optional
        Plots the angles until t^th frame, by default -1
    stim_lines : List[int], optional
        Plots vertical lines in indicated locations,
        by default None
    show_legend : bool, optional
        Shows legend, by default True
    export_path : Path, optional
        Path where the plot will be saved, by default None
    """
    colors = mcp.gen_color(cmap="Set2", n=len(angles_to_plot))

    for i, joint_name in enumerate(angles_to_plot):
        joint_angles = kinematics_data[joint_name]

        label = " ".join((joint_name.split("_")[-2], joint_name.split("_")[-1]))

        convert2deg = 180 / np.pi if degrees else 1

        ax.plot(
            np.array(joint_angles[:until_t]) * convert2deg,
            "o-",
            ms=4.5,
            markevery=[-1],
            label=label,
            color=colors[i],
        )

    if stim_lines is not None:
        ax.vlines(stim_lines, -200, 200, 'red', lw=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(bbox_to_anchor=(1.2, 1), frameon=False, borderaxespad=0.)

    if export_path is not None:
        plt.savefig(export_path, bbox_inches="tight")


def plot_grid(
    img: np.ndarray,
    aligned_pose: Dict[str, np.ndarray],
    joint_angles: Dict[str, np.ndarray],
    leg_angles_to_plot: List[str],
    head_angles_to_plot: List[str],
    key_points_3d: Dict[str, Tuple[np.ndarray, str]],
    key_points_3d_trail: Dict[str, Tuple[np.ndarray, str]],
    t: int,
    t_start: int,
    t_end: int,
    t_interval: int = 20,
    fps: int = 100,
    trail: int = 30,
    stim_lines: List[int] = None,
    export_path: Path = None,
):
    """
    Plots an instance of the animal recording, 3D pose,
    head and leg joint angles in a grid layout. This code
    is intended to use in a for loop to plot and save all
    the frames from `t_start` to `t_end` to make a video afterwards.

    NOTE: This function is intended to plot leg and head joint angles together,
    it will not work if any of these data is missing.

    Parameters
    ----------
    img : Path
        Image of the fly at frame t.
    aligned_pose : Dict[str, np.ndarray]
        Aligned 3D pose.
    joint_angles : Dict[str, np.ndarray]
        Joint angles.
    leg_angles_to_plot : List[str]
        List containing leg joint angle names without the side.
        Example:
            leg_joint_angles = [
                "ThC_yaw",
                "ThC_pitch",
                "ThC_roll",
                "CTr_pitch",
                "CTr_roll",
                "FTi_pitch",
                "TiTa_pitch",
                ]
    head_angles_to_plot : List[str]
        List containing exact names of head joint angle names.
    key_points_3d : Dict[str, Tuple[np.ndarray, str]]
        Dictionary mapping key points names to their indices
        and line styles.
        Example:
            KEY_POINTS_DICT = {
                "RF": (np.arange(0, 5), "solid"),
                "R Ant": (np.arange(10, 12), "o"),
                "Neck": (np.arange(14, 15), "x"),
                "L Ant": (np.arange(12, 14), "o"),
                "LF": (np.arange(5, 10), "solid"),
            }
    key_points_3d_trail : Dict[str, Tuple[np.ndarray, str]]
        Dictionary mapping key points names to their indices
        and line styles for trailing key points.
    t : int
        Frame number t.
    t_start : int
        Start of the time series, i.e., joint angles.
    t_end : int
        End of the time series, i.e., joint angles.
    t_interval : int, optional
        Interval of frame numbers in between x ticks, by default 20
    fps : int, optional
        Frames per second, by default 100
    trail : int, optional
        Number of previous frames where the key point will be visible,
        by default 30
    stim_lines : List[int], optional
        Stimulation indicators, by default None
    export_path : Path, optional
        Path where the plot will be saved, by default None

    Returns
    -------
    Fig
        Figure.
    """

    assert t_start <= t <= t_end, "t_start should be smaller than t_end, t should be in between"
    # import pylustrator
    # pylustrator.start()

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(14, 6), dpi=120)

    gs = GridSpec(3, 4, figure=fig)
    # 7cam recording
    ax_img = fig.add_subplot(gs[0, :1])
    # 3D pose
    ax1 = fig.add_subplot(gs[1:, :2], projection="3d")
    # head, right leg, left leg joint angles
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[1, 2:])
    ax4 = fig.add_subplot(gs[2, 2:])

    # load the image

    # img = cv2.imread(str(img_path / f'frame_{t}.jpg'), 0)
    ax_img.imshow(img, vmin=0, vmax=255, cmap='gray')

    plot_3d_points(ax1, aligned_pose, key_points=key_points_3d, t=t)
    plot_trailing_kp(ax1, aligned_pose, key_points=key_points_3d_trail, trail=trail, t=t)
    plot_joint_angle(ax2, joint_angles, angles_to_plot=head_angles_to_plot, until_t=t, stim_lines=stim_lines)
    plot_joint_angle(
        ax3,
        joint_angles,
        angles_to_plot=[
            f"Angle_LF_{ja}" for ja in leg_angles_to_plot],
        until_t=t,
        stim_lines=stim_lines)
    plot_joint_angle(
        ax4,
        joint_angles,
        angles_to_plot=[
            f"Angle_RF_{ja}" for ja in leg_angles_to_plot],
        until_t=t,
        show_legend=False,
        stim_lines=stim_lines)

    ax_img.axis('off')
    # ax1 properties
    ax1.view_init(azim=7, elev=10)
    ax1.set_xlim3d([-0.8, 0.8])
    ax1.set_ylim3d([-0.7, 0.7])
    ax1.set_zlim3d([0.2, 1.2])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.axis("off")

    # ax2 properties
    ax2.set_xlim((t_start, t_end))
    ax2.spines["bottom"].set_visible(False)
    ax2.set_xlim((t_start, t_end))
    ax2.set_ylim((-90, 70))
    ax2.set_xticks([])
    ax2.set_yticks(ticks=[-90, 0, 70])
    ax2.set_yticklabels(labels=[-90, 0, 70])

    # ax3 properties
    ax3.set_xlim((t_start, t_end))
    ax3.spines["bottom"].set_visible(False)
    ax3.set_ylim((-160, 160))
    ax3.set_xticks([])
    ax3.set_yticks(ticks=[-160, 0, 160])
    ax3.set_yticklabels(labels=[-160, 0, 160])

    # ax4 properties
    ax4.set_xlim((t_start, t_end))
    ax4.set_ylim((-160, 160))
    ax4.set_yticks(ticks=[-160, 0, 160])
    ax4.set_yticklabels(labels=[-160, 0, 160])

    ax4.set_xticks(ticks=np.arange(t_start, t_end + t_interval, t_interval))
    ax4.set_xticklabels(labels=np.arange(t_start, t_end + t_interval, t_interval) / fps)
    ax4.set_xlabel("Time (sec)")

    # #% start: automatic generated code from pylustrator
    fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}
    getattr(fig, '_pylustrator_init', lambda: ...)()
    fig.set_size_inches(24.520000 / 2.54, 11.450000 / 2.54, forward=True)

    fig.text(
        0.5263, 0.8830, 'Head joint angles (deg)', transform=fig.transFigure,
    )
    fig.text(
        0.5263,
        0.6169,
        'Left leg joint angles (deg)',
        transform=fig.transFigure,
    )  # id=fig.texts[0].new
    fig.text(
        0.5263,
        0.3458,
        'Right leg joint angles (deg)',
        transform=fig.transFigure,
    )  # id=fig.texts[1].new

    fig.axes[0].set(position=[0.0385, 0.5848, 0.2853, 0.3084])
    fig.axes[1].set(position=[0.07833, 0.04041, 0.2226, 0.4813])
    fig.axes[2].legend(loc=(1.056, -0.03196), frameon=False)
    fig.axes[2].set(position=[0.4071, 0.6564, 0.397, 0.2135])
    fig.axes[3].set(position=[0.4071, 0.3835, 0.397, 0.2135])
    fig.axes[3].legend(loc=(1.056, -0.5187), frameon=False)
    fig.axes[4].set(position=[0.4071, 0.1107, 0.397, 0.2135])

    fig.texts[0].set(position=(0.4061, 0.883))
    fig.texts[1].set(position=(0.4061, 0.6169))
    fig.texts[2].set(position=(0.4061, 0.3458))
    # % end: automatic generated code from pylustrator

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")
        print(f'Figure saved at {str(export_path)}')

    return fig


def plot_grid_generator(
    fly_frames: np.ndarray,
    aligned_pose: Dict[str, np.ndarray],
    joint_angles: Dict[str, np.ndarray],
    leg_angles_to_plot: List[str],
    head_angles_to_plot: List[str],
    key_points_3d: Dict[str, Tuple[np.ndarray, str]],
    key_points_3d_trail: Dict[str, Tuple[np.ndarray, str]],
    t_start: int,
    t_end: int,
    t_interval: int = 20,
    fps: int = 100,
    trail: int = 30,
    stim_lines: List[int] = None,
    export_path: Path = None,
):
    """ Generator for plotting grid."""

    for t, fly_img in enumerate(fly_frames):
        fig = plot_grid(
            img=fly_img,
            aligned_pose=aligned_pose,
            joint_angles=joint_angles,
            leg_angles_to_plot=leg_angles_to_plot,
            head_angles_to_plot=head_angles_to_plot,
            key_points_3d=key_points_3d,
            key_points_3d_trail=key_points_3d_trail,
            t=t + t_start,
            t_start=t_start,
            t_end=t_end,
            t_interval=t_interval,
            fps=fps,
            trail=trail,
            stim_lines=stim_lines,
            export_path=export_path
        )
        plt.close(fig)
        yield fig_to_array(fig)


def fig_to_array(fig):
    """ Converts a matplotlib figure into an array. """
    canvas = FigureCanvas(fig)
    canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


if __name__ == '__main__':

    start = time.time()
    today = date.today()

    fps = 100

    DATA_PATH = Path('../data/anipose/normal_case/pose-3d')
    out_dir = DATA_PATH / f"inverse_kinematics_results_{today}.mp4"

    anipose_data = DATA_PATH / "pose3d_aligned.pkl"
    forward_kinematics = DATA_PATH / "forward_kinematics.pkl"

    with open(anipose_data, "rb") as f:
        aligned_pose = pickle.load(f)
    with open(forward_kinematics, "rb") as f:
        forward_kin = pickle.load(f)

    # from IPython import embed; embed()

    points_aligned_all = np.concatenate(
        (
            aligned_pose["RF_leg"],
            aligned_pose["LF_leg"],
            aligned_pose["R_head"],
            aligned_pose["L_head"],
            np.tile(aligned_pose["Neck"], (aligned_pose["RF_leg"].shape[0], 1)).reshape(-1, 1, 3),
        ),
        axis=1,
    )

    points_fk = np.concatenate(
        (
            forward_kin["RF_leg"],
            forward_kin["LF_leg"],
            forward_kin["R_head"],
            forward_kin["L_head"],
            np.tile(aligned_pose["Neck"], (aligned_pose["RF_leg"].shape[0], 1)).reshape(-1, 1, 3),
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

    KEY_POINTS_DICT2 = {
        "RF": (np.arange(0, 8), ":"),
        "R Ant": (np.arange(16, 21), "x"),
        "Neck": (np.arange(26, 27), "x"),
        "L Ant": (np.arange(21, 26), "x"),
        "LF": (np.arange(8, 16), ":"),
    }

    animate_3d_points(
        points_aligned_all,
        KEY_POINTS_DICT,
        points3d_second=points_fk,
        key_points_second=KEY_POINTS_DICT2,
        export_path=out_dir.as_posix(),
        frame_no=6000,
        elev=20,
        azim=30,
    )

    end = time.time()
    total_time = end - start

    print(f'Total time taken to execute the code: {total_time}')
