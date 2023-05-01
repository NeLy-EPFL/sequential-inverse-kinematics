""" Creates a metadata file for my experimental dataset.

    Usage:
    >>> python create_metadata.py -s /Volumes/data/GO/7cam -g aJO PR

"""
import os
import re
import argparse
from pathlib import Path
import logging
from sys import platform
import pandas as pd
import cv2
from tqdm import tqdm


# Change the logging level here
logging.basicConfig(level=logging.INFO, format=" %(asctime)s - %(levelname)s- %(message)s")


def parse_args():
    """ Argument parser. """
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '-s',
        '--source',
        type=str,
        default='/data/GO',
        help='Source dir to search folders in',
    )
    parser.add_argument(
        '-date',
        '--date',
        type=int,
        default=0,
        help='Looks at the folders created after this date',
    )
    parser.add_argument(
        '-g',
        '--genotypes',
        action='store',
        nargs='*',
        type=str,
        default=[''],
        help='Looks at folders only with this genotype',
    )
    return parser.parse_args()

# Util functions


def get_exp_info(input_dir: str, source: str):
    """ Returns experimental info within a dictionary.
    E.g. input_dir = "/mnt/nas/GO/7cam/210307_aJO-GAL4>UAS-CsChr/Fly2"
    """

    assert os.path.isdir(input_dir), f'Directory {input_dir} does not exist!'
    input_dir = input_dir[:-1] if input_dir.endswith('/') else input_dir

    experiment = {}
    exp_info = input_dir.replace(source, '').split('/')

    assert 'fly' in exp_info[-1].lower(
    ), '''Directory is not correct! Please provide the Fly folder.
    Example directory is: "/Volumes/data/GO/7cam/210728_aJO-GAL4xUAS-CsChr/Fly001"'''

    experiment['fly_no'] = int(re.split('(\\d+)', exp_info[-1])[1])
    experiment['date'] = exp_info[-2].split('_')[0]
    experiment['genotype'] = exp_info[-2].split('_')[1]
    return experiment


def get_trial_type_and_number(trial_dir: str):
    """ Returns trial info within a dictionary.
    E.g. input_dir = "/mnt/nas/GO/7cam/210307_aJO-GAL4>UAS-CsChr/Fly2/001_Beh/behData
    """
    trial_name = Path(trial_dir).parts[-2]
    trial_number, exp_type = int(trial_name.split('_')[0]), '_'.join(trial_name.split('_')[1:])
    return {'trial_no': trial_number, 'exp_type': exp_type}


def get_stim_type(trial_name):
    if trial_name in ['Beh', 'opto', 'opt']:
        stim_type = 'optogenetics'
    elif trial_name in ['ap', 'airpuff', 'opt']:
        stim_type = 'airpuff'
    elif trial_name in ['opt_ap', 'ap_opt']:
        stim_type = 'opto_airpuff'
    else:
        stim_type = 'unknown'
    return stim_type


def get_stim_dur_repeat(trial_dir: str):
    """ Loads the stimulus info from txt."""
    stim_path = Path(trial_dir) / 'stimulusData' / 'stimulusSequence.txt'

    try:
        with open(stim_path) as f:
            lines = f.readlines()
        stim_dur = [int(line.strip().split(',')[-1]) * 1e-3 for line in lines if 'on' in line]
        stim_repeat = int(lines[-1].strip().split(',')[-1])

    except FileNotFoundError:
        logging.warning(f'{stim_path} does not exist!!')
        stim_dur, stim_repeat = 0, 0

    return {'stim_dur': stim_dur, 'stim_repeat': stim_repeat}


def get_fps_duration_from_video(video_dir):
    """ Finds the fps of a video. """
    cap = cv2.VideoCapture(str(video_dir))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    return {'fps': fps, 'duration': duration}


def create_empty_df(columns=None):
    """ Creates an empty dataframe with the given columns. """
    if columns is None:
        # pandas dataframe
        columns = [
            'date',
            'genotype',
            'fly',
            'exp_type',
            'frame_rate',
            'duration',
            'trial',
            'stim_type',
            'stim_duration',
            'stim_repeat',
            'pose_2d',
            'pose_3d',
            'ik_head',
            'ik_leg',
            'ik_body',
            'beh_quality',
            'pose_quality',
            'fly_dir',
            'trial_dir',
            'exclude',
            'comment',
        ]

    empty_df = pd.DataFrame(columns=columns)
    return empty_df.copy()


def get_experiment_folders(source, date, genotypes):
    """ Returns all the experiment folders in the source directory. """

    selected_folders = [os.path.join(source, folder) for folder in os.listdir(source) if any(
        genotype.lower() in folder.lower() for genotype in genotypes) and
        int(folder.split('_')[0]) > date and os.path.isdir(os.path.join(source, folder))
    ]

    return selected_folders


if __name__ == '__main__':

    args = parse_args()
    source = args.source
    date = args.date
    genotypes = args.genotypes

    selected_experiments = get_experiment_folders(source, date, genotypes)

    columns = [
        'date',
        'genotype',
        'fly',
        'exp_type',
        'frame_rate',
        'duration',
        'trial',
        'stim_type',
        'stim_duration',
        'stim_repeat',
        'pose_2d',
        'pose_3d',
        'ik_head',
        'ik_leg',
        'ik_body',
        'beh_quality',
        'pose_quality',
        'fly_dir',
        'trial_dir',
        'exclude',
        'comment',
    ]
    empty_df = create_empty_df(columns)
    metadata_df = create_empty_df(columns)

    for exp in tqdm(selected_experiments):
        fly_paths = list(path_name for path_name in Path(exp).iterdir() if str(
            path_name.parts[-1]).startswith('Fly') and path_name.is_dir())

        for fly_path in fly_paths:

            trial_paths = list(fly_path.rglob('*/behData'))
            fly_path = str(fly_path)
            exp_info = get_exp_info(fly_path, source=source)

            for trial_path in trial_paths:
                new_row = empty_df.copy()

                trial_path = str(trial_path)
                trial_info = get_trial_type_and_number(trial_path)
                stim_info = get_stim_dur_repeat(trial_path)
                video_dir_list = list(Path(trial_path).rglob('*camera_3.mp4'))

                if len(video_dir_list):
                    video_dir = video_dir_list[-1]
                    video_info = get_fps_duration_from_video(video_dir)
                else:
                    logging.warning(f'Videos do not exist in {trial_path}!!')
                    video_info = {'fps': 0, 'duration': 0}

                # Let's fill the new row data frame
                new_row['stim_duration'] = stim_info['stim_dur']
                new_row['stim_repeat'] = stim_info['stim_repeat']
                new_row['pose_2d'] = (Path(trial_path) / 'pose-2d').is_dir()
                new_row['pose_3d'] = (Path(trial_path) / 'pose-3d').is_dir()
                new_row['ik_head'] = (Path(trial_path) / 'pose-3d' / 'head_joint_angles.pkl').is_file()
                new_row['ik_leg'] = (Path(trial_path) / 'pose-3d' / 'leg_joint_angles.pkl').is_file()
                new_row['ik_body'] = (Path(trial_path) / 'pose-3d' / 'body_joint_angles.pkl').is_file()
                new_row['date'] = exp_info['date']
                new_row['genotype'] = exp_info['genotype']
                new_row['fly'] = exp_info['fly_no']
                new_row['exp_type'] = trial_info['exp_type']
                new_row['frame_rate'] = video_info['fps']
                new_row['duration'] = video_info['duration']
                new_row['trial'] = trial_info['trial_no']
                new_row['stim_type'] = get_stim_type(trial_info['exp_type'])

                new_row.fly_dir = fly_path
                new_row.trial_dir = trial_path

                new_row.comment = None
                new_row.pose_quality = None
                new_row.beh_quality = None
                new_row.exclude = False

                metadata_df = pd.concat([metadata_df, new_row], axis=0).reset_index(drop=True)

    metadata_df = metadata_df.reset_index()
    metadata_df.to_csv(os.path.join(source, 'metadata.csv'))
