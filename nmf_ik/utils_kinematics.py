import os
from pathlib import Path
import pickle
from sys import platform
import re
import logging
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Check the OS to decide the NAS folder.
if platform in ("linux", "linux2"):
    SOURCE = '/mnt/nas/GO/7cam/'
elif platform == "darwin":
    SOURCE = '/Volumes/data/GO/7cam/'


pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(
    level=logging.DEBUG,
    format=' %(asctime)s - %(levelname)s- %(message)s')


def classify_grooming(data, threshold=0.88, add_distance_as_column=True):
    """ Adds a new column where grooming happens.
    Thereshold is determined empirically as 0.88 (was 0.97 before)
    """
    data['Grooming'] = np.zeros(data.shape[0])

    dist_right = calculate_distance(data, 'R')
    dist_left = calculate_distance(data, 'L')
    dist = (dist_right + dist_left) * 0.5
    grooming = dist < threshold
    data.loc[grooming, 'Grooming'] = True

    if add_distance_as_column:
        data['Distance_R'] = dist_right
        data['Distance_L'] = dist_left

    return data


def save_exp_in_csv(file_name: str):
    """ Saves exp directories in a CSV file. """
    file_name += '' if file_name.endswith('.txt') else '.txt'

    with open(file_name, 'r') as f:
        path_list = sorted([line.rstrip().replace(
            SOURCE, '').replace('/behData/images', '') for line in f])

    data_frame_make = [line.split('/')[-3:] for line in path_list]
    df = pd.DataFrame(
        data_frame_make,
        columns=[
            'Genotype',
            'Fly',
            'Experiment'])
    df.to_csv(file_name.replace('txt', 'csv'), index=True)


def get_trial_data(trial_index, kinematics_data):
    """ Takes a data frame and returns the experimental data of the desired trial.
        E.g. the entire experimental data of trial 1
    """
    df_tot = pd.DataFrame()
    for index in trial_index:
        df_tot = df_tot.append(
            kinematics_data[
                kinematics_data.index.get_level_values("Trial") == index
            ]
        )
    return df_tot


def get_trial_number(kinematics_data):
    """ Take a dataframe and returns the trials as a list.
        E.g. [1,2,3,4]
    """
    try:
        trials = np.unique(
            kinematics_data.index.get_level_values("Trial")
        )
    except KeyError:
        trials = np.unique(
            kinematics_data["Trial"]
        )
    return trials


def compute_statistics(kinematics_data, key, trials=None, clean_data=False):
    """ Takes a dataframe and computes mean, std, and stderror of joint data across trials.
    """
    stat_dict = {}
    if trials is None:
        trials = get_trial_number(kinematics_data)
    if clean_data:
        stack_array = np.vstack(
            [
                kinematics_data[
                    kinematics_data["Trial"] == ind
                ][key]
            ] for ind in trials
        )
    else:
        stack_array = np.vstack(
            [
                kinematics_data[
                    kinematics_data.index.get_level_values("Trial") == ind
                ][key]
            ] for ind in trials
        )

    stat_dict["mean"] = np.mean(stack_array, axis=0)
    stat_dict["stderr"] = np.std(
        stack_array, ddof=1, axis=0) / np.sqrt(stack_array.shape[0])
    stat_dict["std"] = np.std(stack_array, axis=0)

    return stat_dict


def save_pickle(directory, data):
    """ Saves pickle. """
    with open(directory, 'wb') as f:
        pickle.dump(data, f)


def get_exp_info(input_dir: str):
    """ Returns experimental info within a dictionary.
    E.g. input_dir = "/mnt/nas/GO/7cam/210307_aJO-GAL4>UAS-CsChr/Fly2"
    """

    assert os.path.isdir(input_dir), f'Directory {input_dir} does not exist!'
    input_dir = input_dir[:-1] if input_dir.endswith('/') else input_dir

    experiment = {}
    exp_info = input_dir.replace(SOURCE, '').split('/')

    assert 'fly' in exp_info[-1].lower(
    ), '''Directory is not correct! Please provide the Fly folder.
    Example directory is: "/Volumes/data/GO/7cam/210728_aJO-GAL4xUAS-CsChr/Fly001"'''

    experiment['fly_no'] = int(re.split('(\\d+)', exp_info[-1])[1])
    experiment['date'] = exp_info[0].split('_')[0]
    experiment['genotype'] = exp_info[0].split('_')[1]
    return experiment


def get_fly_dir(**kwargs):
    """ Returns the fly directories inside an exp folder. """
    fly_no = kwargs.get('fly_no')
    date = kwargs.get('date')
    genotype = kwargs.get('genotype')

    exp_dir = f"{SOURCE}/{date}_{genotype}/"
    fly_no = [f for f in os.listdir(exp_dir) if not f.startswith('.')]
    fly_and_trial_dirs = [
        trial_dir
        for fly in fly_no
        for trial_dir in get_trial_dir(
            fly_no=int(fly[-3:]),
            date=date, genotype=genotype,
            add_df3d=False)
    ]
    return fly_and_trial_dirs


def get_trial_dir(**kwargs):
    """ Gets trials inside an experiment folder. """
    fly_no = kwargs.get('fly_no')
    date = kwargs.get('date')
    genotype = kwargs.get('genotype')
    add_anipose = kwargs.get('add_df3d', True)
    anipose_output_fname = kwargs.get('anipose_output_fname', 'pose-3d')

    exp_dir = f"{SOURCE}/{date}_{genotype}/Fly{fly_no:03d}/"
    pos_dir = f"behData/{anipose_output_fname}" if add_anipose else "behData"
    trial_no = os.listdir(exp_dir)
    trial_dirs = [os.path.join(exp_dir, trial, pos_dir)
                  for trial in trial_no if not trial.startswith('.')]
    return sorted(trial_dirs)


def get_trial_number_and_type(trial_dir):
    """Returns trial number and type to separate different experiments done on the same fly."""
    index_of_behdata = trial_dir.find('/behData')  # strip after behData
    trial_dir = trial_dir[:index_of_behdata]
    trial_folder_name = trial_dir.split("/")[-1]
    trial_number = trial_folder_name.split("_")[0]
    trial_type = "_".join((trial_folder_name.split("_")[1:]))
    return (trial_type, int(trial_number))


def compute_leg_joint_angles(anipose_dir: str, overwrite: bool = False, **kwargs):
    """ Computes joint angles and aligned pose and saves them as .pkl files.
    Parameters
    ----------
    df3d_dir : str
        Path of the file containing the df3d results.
    overwrite : int
        Calculates angles from scratch if exists.

    *Provide settings related to df3dPP in kwargs like interpolation or smoothing.
    """
    from nmf_ik.inverse_kinematics import InverseKinematics

    angles_dir = list(Path(anipose_dir).rglob('leg_joint_angles*.pkl'))
    if len(angles_dir) and not overwrite:
        return pd.read_pickle(angles_dir[0])

    if (Path(anipose_dir) / 'pose3d.h5').is_file():
        with open(Path(anipose_dir) / 'pose3d.h5', 'rb') as f:
            pose_3d = pickle.load(f)
    elif (Path(anipose_dir) / 'pose3d.csv').is_file():
        pose_3d = pd.read_csv(Path(anipose_dir) / 'pose3d.csv')
    else:
        logging.error(
            f'{anipose_dir} does not have pose estimation results! Quitting...')
        return None

    class_ik = InverseKinematics(aligned_pos=pose_3d, legs=['R', 'L'], claw_is_end_effector=False)
    joint_angles = class_ik.calculate_joint_angles(export_path=anipose_dir)

    logging.debug('Saving joint angles at {}'.format(anipose_dir))
    return joint_angles


def compute_head_joint_angles(anipose_dir: str, overwrite: bool = False, **kwargs):
    """ Computes joint angles and aligned pose and saves them as .pkl files.
    Parameters
    ----------
    df3d_dir : str
        Path of the file containing the df3d results.
    overwrite : int
        Calculates angles from scratch if exists.

    *Provide settings related to df3dPP in kwargs like interpolation or smoothing.
    """
    from nmf_ik.head_kinematics import HeadKinematics

    angles_dir = list(Path(anipose_dir).rglob('head_joint_angles*.pkl'))
    if len(angles_dir) and not overwrite:
        return pd.read_pickle(angles_dir[0])

    if (Path(anipose_dir) / 'pose3d.h5').is_file():
        with open(Path(anipose_dir) / 'pose3d.h5', 'rb') as f:
            pose_3d = pickle.load(f)
    elif (Path(anipose_dir) / 'pose3d.csv').is_file():
        pose_3d = pd.read_csv(Path(anipose_dir) / 'pose3d.csv')
    else:
        logging.error(
            f'{anipose_dir} does not have pose estimation results! Quitting...')
        return None

    class_hk = HeadKinematics(points3d=pose_3d)
    joint_angles = class_hk.compute_head_angles(export_path=anipose_dir)

    logging.debug('Saving joint angles at {}'.format(anipose_dir))
    return joint_angles


def get_pose_data(anipose_dir: str):
    """ Returns the aligned pose data. """

    if (Path(anipose_dir) / 'pose3d.h5').is_file():
        with open(Path(anipose_dir) / 'pose3d.h5', 'rb') as f:
            pose_3d = pickle.load(f)
    elif (Path(anipose_dir) / 'pose3d.csv').is_file():
        pose_3d = pd.read_csv(Path(anipose_dir) / 'pose3d.csv')
    else:
        logging.error(
            f'{anipose_dir} does not have pose estimation results! Quitting...')
        return None
    # from IPython import embed; embed()

    columns_to_drop = [
        col for col in pose_3d.columns if any(
            nonwanted in col for nonwanted in (
                'error', 'ncams', 'score'))]

    pose_3d_dropped = pose_3d.drop(columns=columns_to_drop)
    return pose_3d_dropped.to_dict('list')


def load_stim_data(main_dir: str):
    """ Loads the stimulus info from txt."""
    stim_dir = os.path.join(main_dir, 'stimulusSequence.txt')

    try:
        with open(stim_dir) as f:
            lines = f.readlines()
        return lines
    except FileNotFoundError:
        logging.error(f'{stim_dir} does not exist!!')
        return False


def get_time_array(
        frames: np.ndarray,
        frame_rate: int,
        scale: int,
        hamming_window_size: int = 28):
    """ Returns the time array considering the size of the filter. """
    return (frames + hamming_window_size * 0.5) / (frame_rate * scale)


def get_stim_array(
        lines: list,
        frame_rate: int,
        scale: int = 1,
        hamming_window_size: int = 28,
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
                       duration] = False if lines[line_no].startswith('off') else True
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


def get_two_stim_array(stim_array, stim_even=1, stim_odd=2):
    """ Changes the second stimulation indicator.
    - When two stimuli are present, then stim_even=1 stim_odd=2
    - When one stimuli is present and it is airpuff stim_even=0 stim_odd=1
    - When one stimuli is present and it is opto stim_even=1 stim_odd=0
    """
    stim_intervals = get_stim_intervals(stim_array)
    stim_groups = zip(*(iter(stim_intervals),) * 2)

    for stim_no, (start, end) in enumerate(stim_groups):
        if stim_no % 2 == 0:
            stim_array[start:end] = stim_even
        else:
            stim_array[start:end] = stim_odd

    return stim_array


def get_stim_color(
    two_stim_array, color_even=(
        255, 10, 10), color_odd=(
            10, 10, 255)):
    """ Gets the color array for the corresponding stimulation type.
    NOTE: You don't need to use this function if you have one type of stimulation.
    """
    return [color_even if stim == 1 else color_odd for stim in two_stim_array]


def check_fly_data(
        date,
        fly_number,
        genotype,
        exclude_trial=[],
        exp_type=None,
        data_folder='kinematic-analysis'):
    """ Checks the angle calculation for the grooming data.
        Enter the date, fly number of the data.
        e.g.:
            fly_one = check_fly_data(date='211102', fly_number=1, exclude_trial=['2'])

    """
    path = os.path.join(
        SOURCE,
        data_folder,
        f'{date}_{genotype}_Fly{fly_number:03d}.pkl')
    fly_data = pd.read_pickle(path)

    fly_data = fly_data.reset_index()

    if exp_type is None:
        exp_type = np.unique(fly_data['Exp_Type'])

    fly_clean = fly_data[fly_data.Exp_Type.isin(
        exp_type) & ~fly_data.Trial.isin(exclude_trial)]

    trials = dict(zip(fly_clean['Trial'], fly_clean['Exp_Type']))

    # for trial, exp_type in sorted(trials.items()):
    #     fig, axs = plt.subplots(2, 1, figsize=(12, 7))
    #     plt.suptitle(f'Right and Left Front Leg Trial {trial} {exp_type}')
    #     plot_joint_angle(fly_clean[fly_clean['Trial'] == trial], 'RF', axs[0])
    #     plot_joint_angle(fly_clean[fly_clean['Trial'] == trial], 'LF', axs[1])
    #     # axs[0].set_xlim(3,20)
    #     # axs[1].set_xlim(3,20)
    #     plt.tight_layout()
    #     plt.show()

    return fly_clean


def compute_derivative(data, time_step):
    """ Computes the first order derivative. """
    return np.diff(np.array(data)) / time_step


def from_rad2deg(dataframe: pd.DataFrame):
    """ Converts all angle columns in rad into degrees. """
    angle_df = dataframe.filter(regex='Angle')
    for column_name, angle in angle_df.items():
        dataframe[column_name] = np.rad2deg(angle)

    return dataframe


def convert_roll_angles(data):
    """ Convert negative roll angles. """
    for col in data.keys():
        if ('roll' in col) and ('Angle_R' in col):
            data[col] = -1 * data[col].to_numpy()

    return data


def filter_angle_pos(dataframe: pd.DataFrame, win_size: int = 9):
    """ Filters position and angle data. """
    pos_angle = dataframe.filter(regex='Pose.*|Angle.*')
    for column_name, angle in pos_angle.items():
        dataframe[column_name] = savgol_filter(angle, win_size, 3)

    return dataframe


def normalize(data):
    """ Normalize an array. """
    return (data - min(data)) / (max(data) - min(data))


def zscore(x):
    """ z-score normalization of an array. """
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    return (x - x_mean) / x_std

# def filter_data(data, frame_rate, window_length=29, scale=10):
#     """ Filters the data. """
#     frames, dims = data.shape
#     time = np.arange(0, frames / frame_rate, 1 / frame_rate)
#     time_new = np.arange(0, frames / frame_rate, 1 / (scale * frame_rate))

#     smoothed_signal = np.zeros((frames * scale - window_length + 1, dims))

#     interpolated_signal = pchip_interpolate(time, data, time_new)
#     hamming_window = np.hamming(window_length)

#     for i in range(dims):
#         smoothed_signal[:, i] = np.convolve(
# hamming_window / hamming_window.sum(), interpolated_signal[:, i],
# mode='valid')

#     return smoothed_signal


def cart2pol(x, y):
    """ Transformation from cartesian to polar coordinates. """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    """ Transformation from polar to cartesian coordinates. """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def calculate_velocity(dataframe, frame_rate):
    """ Calculates angular and translational velocity of right front angles. """
    df_kinematics = dataframe.filter(
        regex=('Angle_RF.*|Pose_RF.*|Angle_LF.*|Pose_LF.*'))
    for column in df_kinematics.columns:
        new_column_name = column.replace(
            'Pose', 'Velocity').replace(
            'Angle', 'Velocity')
        dataframe[new_column_name] = savgol_filter(
            dataframe[column], polyorder=2, window_length=3, deriv=1) * frame_rate

    return dataframe.copy()


def calculate_speed(dataframe, frame_rate):
    """ Calculates speed from velocity vectors. """
    dataframe_velocity_added = calculate_velocity(dataframe, frame_rate)
    df_kinematics = dataframe_velocity_added.filter(
        regex=('Velocity_RF.*_x|Velocity_LF.*_x'))

    for column_x in df_kinematics.columns:
        column_y = column_x.replace('_x', '_y')
        column_z = column_x.replace('_x', '_z')
        column_speed = column_x.replace('Velocity', 'Speed').replace('_x', '')

        dataframe[column_speed] = np.sqrt(
            dataframe[column_x] ** 2 +
            dataframe[column_y] ** 2 +
            dataframe[column_z] ** 2)

    return dataframe.copy()


def calculate_distance(dataframe):
    """ Calculates distance between the origin and the joints.
    TODO: This should be implemented through velocity.
    """
    df_kinematics = dataframe.filter(regex=('Pose_.*R_x|Pose_.*L_x'))
    for column_x in df_kinematics.columns:
        column_y = column_x.replace('_x', '_y')
        column_z = column_x.replace('_x', '_z')
        column_distance = column_x.replace(
            'Pose', 'Distance').replace(
            '_x', '')

        dataframe[column_distance] = np.sqrt(
            dataframe[column_x] ** 2 +
            dataframe[column_y] ** 2 +
            dataframe[column_z] ** 2)

    return dataframe.copy()


def process_data(data, **settings):
    """ Process data. """
    convert_rad = settings.get('convert_rad', True)
    filter_data = settings.get('filter_data', True)
    convert_roll = settings.get('convert_roll', True)
    classify = settings.get('classify', True)
    physics = settings.get('physics', True)
    frame_rate = settings.get('frame_rate', 100)

    if convert_rad:
        data = from_rad2deg(data)
    if filter_data:
        data = filter_angle_pos(data)
    if convert_roll:
        data = convert_roll_angles(data)
    if classify:
        data = classify_grooming(data)
    if physics:
        data = calculate_speed(data, frame_rate)
        data = calculate_distance(data)

    return data


def perform_significance_test(method, *args, **kwargs):
    """ Performs the significance test on data. """
    if method == 'kruskal':
        from scipy.stats import kruskal
        res = kruskal(*args, **kwargs)
    elif method == 'ks_2samp':
        from scipy.stats import ks_2samp
        assert len(args) == 2, f'Only 2 arrays could be compared with {method}'
        res = ks_2samp(*args, **kwargs)
    elif method == 'pymannkendall':
        import pymannkendall
        res = pymannkendall.original_test(*args, **kwargs)
    elif method == 'friedman':
        from scipy.stats import friedmanchisquare
        res = friedmanchisquare(*args)
    elif method == 'mannwhitneyu':
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(*args, **kwargs)
    elif method == 'median_test':
        from scipy.stats import median_test
        res = median_test(*args)
    else:
        raise ValueError(
            'Wrong method is selected! Available ones are kruskal, ks_2samp, pymannkendall, friedman, median_test, mannwhitneyu.')

    return res
