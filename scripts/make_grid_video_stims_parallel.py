"""
    Runs `make_grid_video_stims.py` in parallel on trials provided in the text file.

    Example usage:
    >>> python make_grid_video_stims_parallel.py --txt_path 'bout_example.txt' --export_path './grid_videos'

"""
import argparse
from pathlib import Path
import numpy as np

import subprocess
import multiprocessing
from multiprocessing import Pool
from subprocess import Popen, PIPE


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline to visualize camera positions",
    )
    parser.add_argument(
        "-tp",
        "--txt_path",
        type=str,
        default=None,
        help="Path of pose files",
    )
    parser.add_argument(
        "-export",
        "--export_path",
        type=str,
        default=None,
        help="Path where the grid video to be saved",
    )
    return parser.parse_args()


def worker_wrapper(arg):
    """ Provide kwargs during multiprocessing. """
    return subprocess.run(arg, check=True)


if __name__ == '__main__':
    args = parse_args()

    txt_path = args.txt_path
    export_path = args.export_path

    with open(txt_path) as f:
        lines = f.readlines()

    # from IPython import embed; embed()

    data_paths = []
    for p in lines:
        path_name = p.strip()
        data_paths += Path(path_name).rglob('pose-3d')

    cmds_list = [
        ['python', './make_grid_video_stims.py',
         '--data_path', str(dp),
         '--video_path', str(dp).replace('pose-3d', 'videos'),
         '--export_path', str(export_path)] for dp in data_paths]

    # First solution, using Popen - in the second option you can limit the NO CORES,
    # in this case, I am not so sure
    # print(' '.join(cmds_list[0]))
    procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE) for cmd in cmds_list]

    for proc in procs_list:
        proc.wait()

    # Second solution - I liked the first better
    # because there is no shell output, good for tqdm

    # NO_CORES = multiprocessing.cpu_count()

    # # Max number of cores
    # with multiprocessing.Pool(processes=NO_CORES) as pool:
    #     pool.map(
    #         worker_wrapper,
    #         [cmd for cmd in cmds_list]
    #     )
