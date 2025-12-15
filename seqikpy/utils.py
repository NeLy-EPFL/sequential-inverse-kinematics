"""Utilities."""

import logging
import pickle
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from scipy.interpolate import pchip_interpolate


def get_fps_from_video(video_dir):
    """Finds the fps of a video."""
    cap = cv2.VideoCapture(str(video_dir))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return fps


def load_stim_data(main_dir: Path):
    """Loads the stimulus info from txt."""
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
    time_scale: int = 1e-3,
):
    """Computes a stimulus array from the stimulus information.

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
            duration = int(
                int(lines[line_no].split()[-1]) * frame_rate * time_scale * scale
            )
            stim_array[start : start + duration] = (
                False if lines[line_no].startswith("off") else True
            )
            start += duration

    trim_ind = int(hamming_window_size * 0.5)
    return stim_array[trim_ind : stim_array.shape[0] - trim_ind]


def get_stim_intervals(stim_data):
    """Reads stimulus array and returns the stim intervals for plotting purposes.
    Use get_stim_array otherwise."""
    stim_on = np.where(stim_data)[0]
    stim_start_end = [stim_on[0]]
    for ind in list(np.where(np.diff(stim_on) > 1)[0]):
        stim_start_end.append(stim_on[ind])
        stim_start_end.append(stim_on[ind + 1])

    if not stim_on[-1] in stim_start_end:
        stim_start_end.append(stim_on[-1])
    return stim_start_end


def calculate_body_size(
    body_template: Dict[str, np.ndarray],
    legs_list: List[str] = ["RF", "LF", "RM", "LM", "RH", "LH"],
) -> Dict[str, np.ndarray]:
    """Calculates body segment sizes from the template data."""
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
            # If Claw, calculate the length of the entire leg
            if segment_name == "Claw":
                body_size[leg] = (
                    body_size[f"{leg}_Coxa"]
                    + body_size[f"{leg}_Femur"]
                    + body_size[f"{leg}_Tibia"]
                    + body_size[f"{leg}_Tarsus"]
                )
            else:
                body_size[f"{leg}_{segment_name}"] = np.linalg.norm(
                    body_template[f"{leg}_{segment_name}"]
                    - body_template[f"{leg}_{leg_segments[i+1]}"]
                )
    # Assuming right and left hand-side are symmetric, checking for one side is enough
    if "R_Antenna_base" in body_template:
        body_size["Antenna"] = np.linalg.norm(
            body_template["R_Antenna_base"] - body_template["R_Antenna_edge"]
        )
        body_size["Antenna_mid_thorax"] = np.linalg.norm(
            body_template["R_Antenna_base"] - body_template["Thorax_mid"]
        )

    return body_size


def fix_coxae_pos(
    points3d, right_coxa_kp="thorax_coxa_R", left_coxa_kp="thorax_coxa_L"
):
    """Calculates the fixed coxae location based on the quantiles."""
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


def leg_length_model(nmf_size: dict, leg_name: str, claw_is_ee: bool) -> float:
    """Returns the leg size from the nmf size dictionary."""
    if claw_is_ee:
        return nmf_size[leg_name]

    return nmf_size[leg_name] - nmf_size[f"{leg_name}_Tarsus"]


def get_length_of_segments(points3d, claw_is_ee=False):
    """Returns a dictionary with segment sizes."""
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
    """Computes mean length of each segment."""
    mean_segment_length = {}
    for segment, length_array in segment_lengths.items():
        mean_segment_length[segment] = np.mean(length_array)

    return mean_segment_length


def get_mean_length_of_segments(points3d):
    """Gets a dictionary with segment lengths."""
    lengths = get_length_of_segments(points3d)
    return compute_mean_length_of_segments(lengths)


def get_leg_length(mean_segment_size):
    """Returns the leg size from the mean segment lengths."""
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
        np.quantile(vector, q=0.5 - quantile_diff)
        + np.quantile(vector, q=0.5 + quantile_diff)
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


def from_anipose_to_array(
    points3d,
    claw_is_end_effector=False,
    kps=["thorax_coxa", "coxa_femur", "femur_tibia", "tibia_tarsus"],
) -> np.ndarray:
    """Convert usual dataframe format into a three dimensional array of size (N,KeyPoints,3)."""

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
    """Convert usual dataframe format into a three dimensional array."""
    if claw_is_end_effector:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]
    else:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus"]

    position_array = np.empty(
        (data_frame[f"Pose_{side}F_Coxa_x"].shape[0], len(key_points), 3)
    )  # timestep, key points, axes

    for i, kp in enumerate(key_points):
        position_array[:, i, :] = np.array(
            [
                data_frame[f"Pose_{side}{segment}_{kp}_x"].to_numpy(),
                data_frame[f"Pose_{side}{segment}_{kp}_y"].to_numpy(),
                data_frame[f"Pose_{side}{segment}_{kp}_z"].to_numpy(),
            ]
        )

    return position_array


def dict_to_nparray_pose(pose_dict, claw_is_end_effector):
    """Convert usual df3dPP dictionary format into a three dimensional array."""

    if claw_is_end_effector:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus", "Claw"]
    else:
        key_points = ["Coxa", "Femur", "Tibia", "Tarsus"]

    position_array = np.empty(
        (pose_dict["Coxa"]["raw_pos_aligned"].shape[0], len(key_points), 3)
    )  # timestep, key points, axes

    for i, kp in enumerate(key_points):
        position_array[:, i, :] = np.array(pose_dict[kp]["raw_pos_aligned"])

    return position_array


def dict_to_nparray_angle(angle_dict, leg, claw_is_end_effector):
    """Convert usual df3dPP dictionary format into a three dimensional array."""

    if claw_is_end_effector:
        dofs = [
            "ThC_roll",
            "ThC_yaw",
            "ThC_pitch",
            "CTr_pitch",
            "CTr_roll",
            "FTi_pitch",
            "TiTa_pitch",
        ]
    else:
        dofs = [
            "ThC_roll",
            "ThC_yaw",
            "ThC_pitch",
            "CTr_pitch",
            "CTr_roll",
            "FTi_pitch",
        ]

    angle_array = np.empty(
        (len(angle_dict[f"{leg}_leg"][dofs[0]]), len(dofs))
    )  # timestep, dofs

    for i, kp in enumerate(dofs):
        angle_array[
            :,
            i,
        ] = np.array(angle_dict[f"{leg}_leg"][kp])

    return angle_array


def interpolate_signal(signal, original_ts, new_ts):
    """Interpolates signals."""
    total_time = signal.shape[0] * original_ts
    original_x = np.arange(0, total_time, original_ts)
    new_x = np.arange(0, total_time, new_ts)

    try:
        interpolated = np.array(pchip_interpolate(original_x, signal, new_x))
    except BaseException:
        signal[np.isinf(signal)] = 0
        signal[-1] = 0
        interpolated = np.array(pchip_interpolate(original_x, signal, new_x))

    return interpolated


def interpolate_joint_angles(joint_angles_dict, **kwargs):
    """Interpolates joint angles."""
    interpolated_joint_angles = {}

    for dof in joint_angles_dict:
        interpolated_joint_angles[dof] = interpolate_signal(
            signal=joint_angles_dict[dof], **kwargs
        )

    return interpolated_joint_angles


def from_sdf(sdf_file: str):
    """Extracts a body template and joint bounds from an SDF file.

    Parameters
    ----------
    sdf_file : str
        Path to the SDF file.

    Returns
    -------
    Dict, Dict
        Body template and joint bounds.
    """
    # Parse the SDF file
    sdf_in = ET.parse(sdf_file)
    root_in = sdf_in.getroot()
    model_in = root_in.find("world").find("model")
    # Get links and joints
    links = model_in.findall("link")
    joints = model_in.findall("joint")
    # Extract the body template
    body_template = {}
    for link in links:
        if not any(dof in link.attrib["name"] for dof in ["roll", "pitch", "yaw"]):
            # Get location of the joint
            joint_loc_str = link.find("pose").text
            # Convert string into a numpy array
            joint_loc = np.array([float(val) for val in joint_loc_str.split(" ")])
            # Get only the x, y, z coordinates
            body_template[link.attrib["name"]] = joint_loc[:3]

    # Extract the joint bounds
    joint_bounds = {
        joint.attrib["name"]: (
            float(joint.find("axis").find("limit").find("lower").text),
            float(joint.find("axis").find("limit").find("upper").text),
        )
        for joint in joints
    }
    return body_template, joint_bounds


def split_arrays_into_chunks(
    arrays: list[np.ndarray],
    approx_n_chunks_total: int,
    overlap: int = 20,
    min_chunk_size: int = 100,
) -> list[tuple[int, int, np.ndarray]]:
    """Splits arrays into chunks with overlap between chunks. This is useful for
    processing long sequences from multiple kinematic chains in parallel.

    Parameters
    ----------
    arrays : list[np.ndarray]
        List of arrays to be split into chunks. Each array should have shape
        (sequence_length, ...) where `...` can be any number of dimensions.
    approx_n_chunks_total : int
        Approximate number of chunks to split the arrays into. The actual number of
        chunks is generally different in order to make the total number of chunks a
        multiple of the number of arrays (i.e. kinematic chains).
    overlap : int, optional
        Number of overlapping frames between chunks, by default 20.
    min_chunk_size : int, optional
        Minimum size of each chunk, by default 100.

    Returns
    -------
    list[tuple[int, int, np.ndarray]]
        List of tuples containing
            - array_idx: int
            - start_idx_within_array: int
            - chunk_of_array: np.ndarray
    """
    assert overlap < min_chunk_size, "Overlap must be smaller than min_chunk_size."
    n_arrays = len(arrays)
    assert n_arrays > 0, "Number of arrays must be positive"
    assert (
        approx_n_chunks_total >= n_arrays
    ), "Approximate number of chunks must be at least equal to the number of arrays."
    n_chunks_per_array = int(approx_n_chunks_total / n_arrays)  # guaranteed to be >= 1
    for arr in arrays:
        assert arr.shape == arrays[0].shape, "All arrays must have the same shape."
    assert arrays[0].ndim >= 1, "Arrays must be at least 1-dimensional."
    seq_length = arrays[0].shape[0]

    # If the chunk size is too small, return the entire arrays as single chunks
    if seq_length <= min_chunk_size:
        return [(arr_idx, 0, arr) for arr_idx, arr in enumerate(arrays)]

    # Create chunks
    chunk_size = seq_length // n_chunks_per_array  # last chunk is generally larger
    if chunk_size < min_chunk_size:
        n_chunks_per_array = seq_length // min_chunk_size
        chunk_size = seq_length // n_chunks_per_array
        logging.warning(
            f"Requested number of chunks results in chunk size smaller than "
            f"min_chunk_size. Reducing number of chunks per array to "
            f"{n_chunks_per_array}."
        )

    chunks = []
    for array_idx, arr in enumerate(arrays):
        start_idx = 0
        for chunk_idx in range(n_chunks_per_array):
            # Determine the end index of the chunk before adding overlap
            if chunk_idx == n_chunks_per_array - 1:
                end_idx_before_overlap = seq_length
                end_idx_after_overlap = seq_length
            else:
                end_idx_before_overlap = start_idx + chunk_size
                end_idx_after_overlap = end_idx_before_overlap + overlap
                assert end_idx_after_overlap <= seq_length, (
                    "Chunk end idx with overlap exceeds array length - shouldn't "
                    "happen because the present chunk is not the last one, and the "
                    "chunk after it is guaranteed to be at least `min_chunk_size` "
                    "long, which is guaranteed to be larger than `overlap`."
                )
            assert end_idx_before_overlap - start_idx >= min_chunk_size, (
                "Chunk size is smaller than minimum chunk size - shouldn't happen "
                "because the total array length is guaranteed to be at least "
                "`min_chunk_size` long and the `n_chunks` is rounded down."
            )

            chunk_arr = arr[start_idx:end_idx_after_overlap]
            chunks.append((array_idx, start_idx, chunk_arr))
            start_idx = end_idx_before_overlap
    return chunks


def merge_chunks_into_arrays(
    chunks: list[tuple[int, int, np.ndarray]],
    n_arrays: int,
    seq_length: int,
    overlap: int = 20,
) -> list[np.ndarray]:
    """Merges chunks back into arrays.

    Parameters
    ----------
    chunks : list[tuple[int, int, np.ndarray]]
        List of tuples containing
            - array_idx: int
            - start_idx_within_array: int
            - chunk_of_array: np.ndarray of shape (chunk_length_with_overlap, ...)
    n_arrays : int
        Number of arrays (i.e. kinematic chains).
    seq_length : int
        Length of each array.
    overlap : int, optional
        Number of overlapping frames between chunks, by default 20.

    Returns
    -------
    list[np.ndarray]
        List of merged arrays, each of shape (seq_length, ...) where `...` matches the
        shape of chunked arrays in the input.
    """
    assert n_arrays > 0, "Number of arrays must be positive"
    ref_arr = chunks[0][2]
    assert ref_arr.ndim >= 1, "Arrays must be at least 1-dimensional"
    for _, _, chunk_arr in chunks:
        assert (
            chunk_arr.shape[1:] == ref_arr.shape[1:]
        ), "All chunks must have the same shape except for the 0th dimension"

    # Blend chunks into full arrays
    merged_arrays = [
        np.zeros((seq_length, *ref_arr.shape[1:]), dtype=ref_arr.dtype)
        for _ in range(n_arrays)
    ]
    for array_idx, start_idx, chunk_arr in chunks:
        # The first part of the chunk overlaps with the previous chunk. The policy is:
        #   - For the first half of the overlap, just use data from the previous chunk
        #     while the current chunk is still "warming up" from the initial condition
        #   - For the second half of the overlap, use a linear "blend" between the
        #     previous chunk and the current chunk, with the weight of the current chunk
        #     increasing linearly from 0 to 1.
        chunk_length_with_overlap = chunk_arr.shape[0]
        blend_weight_curr = np.ones(chunk_length_with_overlap)
        if start_idx > 0:
            # Ramp up the weight of the current chunk over the overlap region
            blend_weight_curr[:overlap] = np.linspace(-1, 1, overlap, endpoint=False)
            blend_weight_curr = np.clip(blend_weight_curr, 0, 1)
        target_slice = np.s_[start_idx : start_idx + chunk_length_with_overlap]
        mask_shape = (chunk_length_with_overlap,) + (1,) * (chunk_arr.ndim - 1)
        blend_weight_curr = blend_weight_curr.reshape(mask_shape)
        blend_weight_prev = 1 - blend_weight_curr
        if n_arrays == 1:
            # special case: if dofs are already separated by leg, then we should save
            # data to the only array even though array_idx might be > 0 (because they
            # can be originally from all legs)
            array_idx = 0
        target_arr = merged_arrays[array_idx]
        target_arr[target_slice] = (
            target_arr[target_slice] * blend_weight_prev + chunk_arr * blend_weight_curr
        )
    return merged_arrays
