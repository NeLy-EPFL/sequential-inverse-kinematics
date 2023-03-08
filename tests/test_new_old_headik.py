from nmf_ik.data import NMF_TEMPLATE
from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
import nmf_ik
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def calc_head_and_plot(aligned_pose, head_joint_angles, plot=True, export_path=None, title=''):
    class_hk = HeadInverseKinematics(
        aligned_pos=aligned_pose,
        nmf_template=NMF_TEMPLATE,
    )
    head_joint_angles_derotated = class_hk.compute_head_angles()

    if plot:
        fig, axs = plt.subplots(3, 2, figsize=(10, 6))

        axs[0, 0].plot(-np.rad2deg(head_joint_angles['Angle_antenna_pitch_R']))
        axs[0, 0].plot(np.rad2deg(head_joint_angles_derotated['Angle_antenna_pitch_R']),
                       color='firebrick', ls='--')
        axs[0, 0].set_ylabel('Right')
        # axs[0,0].plot(np.rad2deg(angles_new_r), ls=':')

        axs[0, 1].plot(-np.rad2deg(head_joint_angles['Angle_antenna_pitch_L']), label='original')
        axs[0, 1].plot(np.rad2deg(head_joint_angles_derotated['Angle_antenna_pitch_L']),
                       color='firebrick', ls='--', label='derotated')
        axs[0, 1].set_ylabel('Left')
        axs[0, 1].legend()

        axs[1, 0].plot(np.rad2deg(head_joint_angles['Angle_antenna_yaw_R']))
        axs[1, 0].plot(np.rad2deg(head_joint_angles_derotated['Angle_antenna_yaw_R']),
                       color='firebrick', ls='--')
        axs[1, 0].set_ylabel('Right')
        # axs[0,0].plot(np.rad2deg(angles_new_r), ls=':')

        axs[1, 1].plot(np.rad2deg(head_joint_angles['Angle_antenna_yaw_L']), label='original')
        axs[1, 1].plot(np.rad2deg(head_joint_angles_derotated['Angle_antenna_yaw_L']),
                       color='firebrick', ls='--', label='derotated')
        axs[1, 1].set_ylabel('Left')
        axs[1, 1].legend()

        axs[2, 0].plot(np.rad2deg(head_joint_angles['Angle_head_pitch']))
        axs[2, 0].set_ylabel('Angle_head_pitch')
        axs[2, 1].plot(np.rad2deg(head_joint_angles['Angle_head_roll']))
        axs[2, 1].set_ylabel('Angle_head_roll')

        plt.suptitle(title)

        if export_path is not None:
            plt.savefig(export_path, dpi=100)

        plt.show()

    return head_joint_angles_derotated


if __name__ == '__main__':
    pose_dirs = [
        '/Volumes/data2/GO/7cam/221221_aJO-GAL4xUAS-CsChr/Fly005/003_Beh/behData/pose-3d/',
        '/Volumes/data2/GO/7cam/221221_aJO-GAL4xUAS-CsChr/Fly005/004_Beh/behData/pose-3d/',
        '/Volumes/data2/GO/7cam/221221_aJO-GAL4xUAS-CsChr/Fly005/005_RLF_HF/behData/pose-3d/',
        '/Volumes/data2/GO/7cam/221222_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d/',
        '/Volumes/data2/GO/7cam/221222_aJO-GAL4xUAS-CsChr/Fly001/003_Beh/behData/pose-3d/',
        '/Volumes/data2/GO/7cam/221222_aJO-GAL4xUAS-CsChr/Fly001/004_Beh/behData/pose-3d/',
        '/Volumes/data2/GO/7cam/221222_aJO-GAL4xUAS-CsChr/Fly001/011_RLF/behData/pose-3d/',
        '/Volumes/data2/GO/7cam/221223_aJO-GAL4xUAS-CsChr/Fly001/002_Beh/behData/pose-3d/',
    ]

    for posedir in pose_dirs:
        aligned_pose = pd.read_pickle(Path(posedir) / 'pose3d_aligned.pkl')
        head_joint_angles = pd.read_pickle(Path(posedir) / 'head_joint_angles.pkl')
        calc_head_and_plot(
            aligned_pose,
            head_joint_angles,
            plot=True,
            export_path=Path(posedir) /
            'head_angles.png',
            title=posedir.replace('/Volumes/data2/GO/7cam/', '').replace('/', '_'))
