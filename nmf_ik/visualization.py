""" Plotting and animation. """
import pickle
import logging
import time
import cv2
import pandas as pd
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mycolorpy import colorlist as mcp
from matplotlib.gridspec import GridSpec
import mpl_toolkits.mplot3d.axes3d as p3
import subprocess

# Change the logging level here
logging.basicConfig(
    level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s"
)

def get_frames_from_video(path):

    vidcap = cv2.VideoCapture(str(path))
    success,image = vidcap.read()
    count = 0
    write_path = path.parents[0] / str(path.name).replace('.mp4','_frames')
    write_path.mkdir()
    print('Frames will be saved at: ', write_path)

    while success:
      cv2.imwrite(str(write_path / f"frame_{count}.jpg"), image)     # save frame as JPEG file
      success,image = vidcap.read()
#       print('Read a new frame: ', success)
      count += 1

def get_frames_from_video_ffmpeg(path):

    write_path = path.parents[0] / str(path.name).replace('.mp4','_frames')
    write_path.mkdir()
    cmd = ['ffmpeg', '-i', str(path), '-r', '1',  str(write_path / 'frame_%d.jpg') ]
    subprocess.run(cmd)


def load_grid_plot_data(data_path: Path):
    if (data_path / "body_joint_angles.pkl").is_file():
        joint_angles = pd.read_pickle(data_path / "body_joint_angles.pkl")
    else:
        head_joint_angles = pd.read_pickle(data_path / "head_joint_angles.pkl")
        leg_joint_angles = pd.read_pickle(data_path / "leg_joint_angles.pkl")
        joint_angles = {**head_joint_angles, **leg_joint_angles}
    aligned_pose = pd.read_pickle(data_path / "pose3d_aligned.pkl")

    return joint_angles, aligned_pose


def animate_3d_points(
    points3d,
    key_points,
    export_path,
    points3d_second=None,
    key_points_second=None,
    fps=100,
    frame_no=1000,
    format_video="mp4",
    elev=10,
    azim=90,
):
    """Makes an animation from 3D plot."""
    # plt.style.use('dark_background')
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
    """Plots 3D points."""

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
            color='grey'

        if len(order) > 3:
            ax3d.plot(
                points3d[t, order, 0],
                points3d[t, order, 1],
                points3d[t, order, 2],
                label=kp,
                linestyle=ls,
                linewidth=3,
                color=color,
            )
        else:
            ax3d.plot(
                points3d[t, order, 0],
                points3d[t, order, 1],
                points3d[t, order, 2],
                label=kp,
                marker=ls,
                markersize=9,
                color=color,
            )



    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")


def plot_trailing_kp(ax3d, points3d, key_points, export_path=None, t=0, trail=5):
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
            color='grey'

        ax3d.scatter(
            points3d[max(0, t-trail):t, order, 0],
            points3d[max(0, t-trail):t, order, 1],
            points3d[max(0, t-trail):t, order, 2],
            label=kp,
            marker=ls,
#             markersize=9,
            color=color,
        )

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")


def plot_joint_angle(
    ax, kinematics_data, angles_to_plot, x_axis=None, degrees=True, fps=100, until_t=-1
):
    """[summary]

    Parameters
    ----------
    kinematics_data : [type]
        [description]
    leg : str
        Name of the leg. e.g., RF
    ax : [type], optional
        [description], by default None
    """
    colors = mcp.gen_color(cmap="Set2", n=len(angles_to_plot))

    for i, joint_name in enumerate(angles_to_plot):
        joint_angles = kinematics_data[joint_name]

        label = " ".join((joint_name.split("_")[-2], joint_name.split("_")[-1]))
        if degrees:
            ax.plot(
                np.array(joint_angles[:until_t]) * 180 / np.pi,
                "o-",
                ms=5,
                markevery=[-1],
                label=label,
                color=colors[i],
            )
        else:
            ax.plot(
                np.array(joint_angles[:until_t]),
                "o-",
                ms=5,
                markevery=[-1],
                label=label,
                color=colors[i],
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(bbox_to_anchor=(1.2, 1), frameon=False, borderaxespad=0.)



def plot_grid(
    img_path,
    aligned_pose,
    joint_angles,
    leg_angles_to_plot,
    head_angles_to_plot,
    key_points_3d,
    key_points_3d_trail,
    t,
    t_start,
    t_end,
    fps=100,
    trail=30,
    export_path=None,
):

    # import pylustrator
    # pylustrator.start()
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(14, 6), dpi=120)

    gs = GridSpec(3, 4, figure=fig)
    ax_img = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1:, :2], projection="3d")
    ax2 = fig.add_subplot(gs[1, 2:])
    ax3 = fig.add_subplot(gs[2, 2:])

    # load the image
    img = cv2.imread(str(img_path / f'frame_{t}.jpg'), 0)
    ax_img.imshow(img, vmin=0, vmax=255, cmap='gray')

    plot_3d_points(ax1, aligned_pose, key_points=key_points_3d, t=t)
    plot_trailing_kp(ax1, aligned_pose, key_points=key_points_3d_trail, trail=trail, t=t)
    plot_joint_angle(ax3, joint_angles, angles_to_plot=leg_angles_to_plot, until_t=t)
    plot_joint_angle(ax2, joint_angles, angles_to_plot=head_angles_to_plot, until_t=t)

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
    # ax2.set_ylabel("Head joint angles (deg)")
    ax2.spines["bottom"].set_visible(False)
    ax2.set_xticks([])
    ax2.set_xlim((t_start, t_end))

    ax2.set_ylim((-60, 60))
    ax2.set_yticks(ticks=[-60, 0, 60])
    ax2.set_yticklabels(labels=[-60, 0, 60])

    # ax3 properties
    ax3.set_xlim((t_start, t_end))
    ax3.set_ylim((-160, 160))
    ax3.set_yticks(ticks=[-160, 0, 160])
    ax3.set_yticklabels(labels=[-160, 0, 160])

    ax3.set_xticks(ticks=np.arange(t_start, t_end + 50, 50))
    ax3.set_xticklabels(labels=np.arange(t_start, t_end + 50, 50) / fps)

    # ax3.set_ylabel("Leg joint angles (deg)")
    ax3.set_xlabel("Time (sec)")

    #% start: automatic generated code from pylustrator
    fig.ax_dict = {ax.get_label(): ax for ax in fig.axes}
    getattr(fig, '_pylustrator_init', lambda: ...)()
    fig.set_size_inches(22.870000/2.54, 15.130000/2.54, forward=True)
    fig.axes[0].set(position=[0.2786, 0.664, 0.4316, 0.326])
    fig.axes[1].set(position=[0.00265, 0.1157, 0.2995, 0.4525])
    fig.axes[2].legend(loc=(0.9902, -0.06284), frameon=False)
    fig.axes[2].set_position([0.342251, 0.376048, 0.523162, 0.221085])
    fig.axes[2].yaxis.labelpad = -20.137022
    fig.axes[3].legend(loc=(0.9902, -0.01824), frameon=False)
    fig.axes[3].set(position=[0.5293, 0.08108, 0.3707, 0.2265])
    fig.axes[3].set_position([0.342251, 0.082519, 0.523162, 0.221085])
    fig.axes[3].yaxis.labelpad = -26.465817
    fig.text(0.3378, 0.6156, 'Head joint angles (deg)', transform=fig.transFigure, )
    fig.text(0.3378, 0.3281, 'Leg joint angles (deg)',transform=fig.transFigure, )
    #% end: automatic generated code from pylustrator

    if export_path is not None:
        fig.savefig(export_path, bbox_inches="tight")
        print(f'Figure saved at {str(export_path)}')

    return fig

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

