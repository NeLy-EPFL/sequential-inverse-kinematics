""" Plotting and animation. """
import pickle
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mycolorpy import colorlist as mcp
import mpl_toolkits.mplot3d.axes3d as p3

# Change the logging level here
logging.basicConfig(
    level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s"
)


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


def plot_3d_points(points3d, key_points, export_path=None, t=0):
    """Plots 3D points."""
    fig = plt.figure(figsize=(5, 5))
    ax3d = p3.Axes3D(fig)
    ax3d.view_init(azim=9, elev=16)

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
