""" Utilities. """
from pathlib import Path
import logging
from typing import Dict, List
import pickle
import numpy as np
import cv2
from nptyping import NDArray
from scipy.interpolate import pchip_interpolate


def get_fps_from_video(video_dir):
    """ Finds the fps of a video. """
    cap = cv2.VideoCapture(str(video_dir))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return fps


def load_stim_data(main_dir: Path):
    """ Loads the stimulus info from txt."""
    stim_dir = main_dir / "stimulusSequence.txt"

    try:
        with open(stim_dir) as f:
            lines = f.readlines()
        return lines
    except FileNotFoundError:
        logging.error(f"{stim_dir} does not exist!!")
        return False


def get_stim_array(
        lines: list,
        frame_rate: int,
        scale: int = 1,
        hamming_window_size: int = 0,
        time_scale: int = 1e-3):
    """ Computes a stimulus array from the stimulus information.

    Parameters
    ----------
    main_dir : str
        Path of the file containing the stim info.
    frame_rate : int
        Frame rate of the recordings.
    scale : int, optional
        Interpolation rate applied in df3dPP, by default 1 (no interpolation!)
    hamming_window_size : int, optional
        Size of the filter applied in df3dPP, by default 28
    time_scale : int, optional
        Time scale of the stimulus info (msec), by default 1e-3

    Returns
    -------
    np.ndarray
        np.array(length, ) of boolean values indicating the stimulus time points.
    """
    stim_duration = int(lines[0].split()[-2])
    frame_no = stim_duration * frame_rate * scale
    repeat = int(lines[-1].split()[-1])
    stim_array = np.zeros(frame_no)
    # Get only the sequence data
    start = 0
    for repeat_no in range(repeat):
        for line_no in range(2, len(lines) - 1):
            duration = int(int(lines[line_no].split()[-1])
                           * frame_rate * time_scale * scale)
            stim_array[start:start +
                       duration] = False if lines[line_no].startswith("off") else True
            start += duration

    trim_ind = int(hamming_window_size * 0.5)
    return stim_array[trim_ind: stim_array.shape[0] - trim_ind]


def get_stim_intervals(stim_data):
    """ Reads stimulus array and returns the stim intervals for plotting purposes.
    Use get_stim_array otherwise. """
    stim_on = np.where(stim_data)[0]
    stim_start_end = [stim_on[0]]
    for ind in list(np.where(np.diff(stim_on) > 1)[0]):
        stim_start_end.append(stim_on[ind])
        stim_start_end.append(stim_on[ind + 1])

    if not stim_on[-1] in stim_start_end:
        stim_start_end.append(stim_on[-1])
    return stim_start_end


def calculate_body_size(
    body_template: Dict[str, NDArray],
    legs_list: List[str] = ["RF", "LF", "RM", "LM", "RH", "LH"],
) -> Dict[str, NDArray]:
    """ Calculates body segment sizes from the template data."""
    if set(legs_list).difference(set(["RF", "LF", "RM", "LM", "RH", "LH"])):
        raise NameError(
            f"""
            legs_list could only contain ["RF", "LF", "RM", "LM", "RH", "LH"],
            currently, it contains {legs_list}
            """
        )
    body_size = {}
    leg_segments = ["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]

    for i, segment_name in enumerate(leg_segments):
        for leg in legs_list:
            # If Claw, calculate the length of the entire leg
            if segment_name == "Claw":
                body_size[leg] = body_size[f"{leg}_Coxa"] + body_size[f"{leg}_Femur"] + \
                    body_size[f"{leg}_Tibia"] + body_size[f"{leg}_Tarsus"]
            else:
                body_size[f"{leg}_{segment_name}"] = np.linalg.norm(
                    body_template[f"{leg}_{segment_name}"] -
                    body_template[f"{leg}_{leg_segments[i+1]}"]
                )
    # Assuming right and left hand-side are symmetric, checking for one side is enough
    if "R_Antenna_base" in body_template:
        body_size["Antenna"] = np.linalg.norm(body_template["R_Antenna_base"] - body_template["R_Antenna_edge"])
        body_size["Antenna_mid_thorax"] = np.linalg.norm(
            body_template["R_Antenna_base"] - body_template["Thorax_mid"])

    return body_size


def drop_level_dlc(data_frame):
    """ Converts DLC type dataframe into one level df. """
    data_frame.columns = data_frame.columns.droplevel()
    data_frame.columns = ["_".join(col) for col in data_frame.columns.values]

    return data_frame


def fix_coxae_pos(points3d, right_coxa_kp="thorax_coxa_R", left_coxa_kp="thorax_coxa_L"):
    """ Calculates the fixed coxae location based on the quantiles. """
    coxa_right = get_array(right_coxa_kp, points3d)
    coxa_left = get_array(left_coxa_kp, points3d)
    coxa_right_fixed = (
        np.quantile(coxa_right, 0.3, axis=1) + np.quantile(coxa_right, 0.7, axis=1)
    ) * 0.5
    coxa_left_fixed = (
        np.quantile(coxa_left, 0.3, axis=1) + np.quantile(coxa_left, 0.7, axis=1)
    ) * 0.5

    return {"R": coxa_right_fixed, "L": coxa_left_fixed}


def compute_length_of_segment(points3d, segment_beg, segment_end):
    """Computes the length of a segment."""
    return np.sqrt(
        (points3d[f"{segment_beg}_x"] - points3d[f"{segment_end}_x"]) ** 2
        + (points3d[f"{segment_beg}_y"] - points3d[f"{segment_end}_y"]) ** 2
        + (points3d[f"{segment_beg}_z"] - points3d[f"{segment_end}_z"]) ** 2
    )


def leg_length_model(nmf_size: dict, leg_name: str, claw_is_ee: bool):
    if claw_is_ee:
        return nmf_size[leg_name]
    else:
        # print(leg_name)
        return nmf_size[leg_name] - nmf_size[f"{leg_name}_Tarsus"]


def get_length_of_segments(points3d, claw_is_ee=False):
    """ Returns a dictionary with segment sizes. """
    segments = {
        "Coxa": ("thorax_coxa", "coxa_femur"),
        "Femur": ("coxa_femur", "femur_tibia"),
        "Tibia": ("femur_tibia", "tibia_tarsus"),
        # "Antenna": ("base_anten", "tip_anten"),
    }
    if claw_is_ee:
        segments = {**segments, "Tarsus": ("tibia_tarsus", "claw")}
    lengths = {}
    for segment_name, (segment_start, segment_end) in segments.items():
        for side in ["R", "L"]:
            lengths[f"{side}_{segment_name}"] = compute_length_of_segment(
                points3d, f"{segment_start}_{side}", f"{segment_end}_{side}"
            )
    return lengths


def compute_mean_length_of_segments(segment_lengths):
    """Computes mean length of each segment. """
    mean_segment_length = {}
    for segment, length_array in segment_lengths.items():
        mean_segment_length[segment] = np.mean(length_array)

    return mean_segment_length


def get_mean_length_of_segments(points3d):
    """ Gets a dictionary with segment lengths. """
    lengths = get_length_of_segments(points3d)
    return compute_mean_length_of_segments(lengths)


def get_leg_length(mean_segment_size):
    """ Returns the leg size from the mean segment lengths."""
    leg_length_r = np.sum(
        [size for name, size in mean_segment_size.items() if "R_" in name]
    )
    leg_length_l = np.sum(
        [size for name, size in mean_segment_size.items() if "L_" in name]
    )

    return {"R": leg_length_r, "L": leg_length_l}


def get_array(kp_name, kp_dict):
    """Returns 3D points of a key point in an array format."""
    return np.array(
        [
            kp_dict[f"{kp_name}_x"],
            kp_dict[f"{kp_name}_y"],
            kp_dict[f"{kp_name}_z"],
        ]
    ).reshape(-1, 3)


def get_mean_quantile(vector, quantile_diff=0.05):
    return 0.5 * (
        np.quantile(vector, q=0.5 - quantile_diff) + np.quantile(vector, q=0.5 + quantile_diff)
    )


def dist_calc(v1, v2):
    return np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)


def get_distance_btw_vecs(vector1, vector2):
    return np.linalg.norm(vector1 - vector2, axis=1)


def save_file(out_fname, data):
    """Save file."""
    with open(out_fname, "wb") as f:
        pickle.dump(data, f)


def load_file(output_fname):
    """Load file."""
    with open(output_fname, "rb") as f:
        pts = pickle.load(f)
    return pts


def from_anipose_to_array(points3d, claw_is_end_effector=False):
    """Convert usual dataframe format into a three dimensional array of size (N,KeyPoints,3)."""

    kps = ["thorax_coxa", "coxa_femur", "femur_tibia", "tibia_tarsus"]
    if claw_is_end_effector:
        kps += ["claw"]
        key_points = [f"{kp}_{side}" for side in ["R", "L"] for kp in kps]
    else:
        key_points = [f"{kp}_{side}" for side in ["R", "L"] for kp in kps]

    frames_no = points3d.shape[0]
    position_array = np.empty(
        (frames_no, len(key_points), 3)
    )  # timestep, key points, axes

    for i, kp in enumerate(key_points):
        position_array[:, i, :] = get_array(kp, points3d).T

    return position_array


def df_to_nparray(data_frame, side, claw_is_end_effector, segment="F"):
    """ Convert usual dataframe format into a three dimensional array. """
    if claw_is_end_effector:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]
    else:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus"]

    position_array = np.empty(
        (data_frame[f"Pose_{side}F_Coxa_x"].shape[0],
         len(key_points),
         3))  # timestep, key points, axes

    for i, kp in enumerate(key_points):
        position_array[:,i,:] = np.array(
            [
                data_frame[f"Pose_{side}{segment}_{kp}_x"].to_numpy(),
                data_frame[f"Pose_{side}{segment}_{kp}_y"].to_numpy(),
                data_frame[f"Pose_{side}{segment}_{kp}_z"].to_numpy()
            ]
        )

    return position_array


def dict_to_nparray_pose(pose_dict, claw_is_end_effector):
    """ Convert usual df3dPP dictionary format into a three dimensional array. """

    if claw_is_end_effector:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]
    else:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus"]

    position_array = np.empty(
        (pose_dict["Coxa"]["raw_pos_aligned"].shape[0],
         len(key_points),
         3))  # timestep, key points, axes

    for i, kp in enumerate(key_points):
        position_array[:, i, :] = np.array(
            pose_dict[kp]["raw_pos_aligned"])

    return position_array


def dict_to_nparray_angle(angle_dict, leg, claw_is_end_effector):
    """ Convert usual df3dPP dictionary format into a three dimensional array. """

    if claw_is_end_effector:
        dofs = ["ThC_roll", "ThC_yaw", "ThC_pitch", "CTr_pitch", "CTr_roll", "FTi_pitch", "TiTa_pitch"]
    else:
        dofs = ["ThC_roll", "ThC_yaw", "ThC_pitch", "CTr_pitch", "CTr_roll", "FTi_pitch"]

    angle_array = np.empty(
        (len(angle_dict[f"{leg}_leg"][dofs[0]]),
         len(dofs)))  # timestep, dofs

    for i, kp in enumerate(dofs):
        angle_array[:, i, ] = np.array(
            angle_dict[f"{leg}_leg"][kp])

    return angle_array


def interpolate_signal(signal, original_ts, new_ts):
    """ Interpolates signals. """
    total_time = signal.shape[0] * original_ts
    original_x = np.arange(0, total_time, original_ts)
    new_x = np.arange(0, total_time, new_ts)

    try:
        interpolated = np.array(
            pchip_interpolate(original_x, signal, new_x)
        )
    except BaseException:
        signal[np.isinf(signal)] = 0
        signal[-1] = 0
        interpolated = np.array(
            pchip_interpolate(original_x, signal, new_x)
        )

    return interpolated


def interpolate_joint_angles(joint_angles_dict, **kwargs):
    """ Interpolates joint angles. """
    interpolated_joint_angles = {}

    for dof in joint_angles_dict:
        interpolated_joint_angles[dof] = interpolate_signal(signal=joint_angles_dict[dof], **kwargs)

    return interpolated_joint_angles
