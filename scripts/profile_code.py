import cProfile
import glob
import pandas as pd
import numpy as np

from nmf_ik.data import NMF_SIZE

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

data_path = '../data/220410_aJO-GAL4xUAS-CsChr_Fly001_002_Beh'


def create_kinematic_chain(chain_name, nmf_size, side):
    """Creates a kinematic chain for the NMF leg."""
    leg_chain = Chain(
        name=chain_name,
        links=[
            OriginLink(),
            URDFLink(
                name=f"{side}F_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
            ),
            URDFLink(
                name=f"{side}F_ThC_yaw",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
            ),
            URDFLink(
                name=f"{side}F_ThC_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            URDFLink(
                name=f"{side}F_CTr_pitch",
                origin_translation=[0, 0, -nmf_size[f"{side}F_Coxa"]],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            URDFLink(
                name=f"{side}F_CTr_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
            ),
            URDFLink(
                name=f"{side}F_FTi_pitch",
                origin_translation=[0, 0, -nmf_size[f"{side}F_Femur"]],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            URDFLink(
                name=f"{side}F_TiTa_pitch",
                origin_translation=[0, 0, -nmf_size[f"{side}F_Tibia"]],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
            ),
            # I am not sure about this ma
            URDFLink(
                name=f"{side}F_Claw",
                origin_translation=[0, 0, -nmf_size[f"{side}F_Tarsus"]],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
            ),
        ],
    )

    return leg_chain


def calculate_ik(leg_chain, target_pos, initial_angles):
    return leg_chain.inverse_kinematics(target_position=target_pos, initial_position=initial_angles)


def calculate_fk(leg_chain, joint_angles):
    fk = leg_chain.forward_kinematics(joint_angles, full_kinematics=True)
    positions = np.zeros((len(leg_chain.links), 3))
    for link in range(len(leg_chain.links)):
        positions[link, :] = fk[link][:3, 3]

    return positions


def main(right_leg, angles, pos):

    angles_sorted_right = np.array(
        [
            np.zeros((len(angles['Angle_RF_ThC_yaw']),)),
            angles['Angle_RF_ThC_roll'],
            angles['Angle_RF_ThC_yaw'],
            angles['Angle_RF_ThC_pitch'],
            angles['Angle_RF_CTr_pitch'],
            angles['Angle_RF_CTr_roll'],
            angles['Angle_RF_FTi_pitch'],
            angles['Angle_RF_TiTa_pitch'],
            np.zeros((len(angles['Angle_RF_ThC_yaw']),)),
        ]
    )

    frames_no = pos['RF_Claw_raw_pos_aligned'].shape[0]

    initial_angles_right = angles_sorted_right[:, 0]
    calculated_angles_right = np.zeros((9, frames_no))
    calculated_pos_right = np.zeros((9, frames_no, 3))

    coxa_fixed_pos_right = np.array(pos['RF_Coxa_fixed_pos_aligned'])
    pos_claw = np.array(pos['RF_Claw_raw_pos_aligned'])

    for ts in range(frames_no):

        initial_angles = initial_angles_right if ts == 0 else calculated_angles_right[:, ts - 1]

        calculated_angles_right[:, ts] = calculate_ik(
            right_leg, pos_claw[ts, :] - coxa_fixed_pos_right, initial_angles)
        calculated_pos_right[:, ts, :] = calculate_fk(right_leg, calculated_angles_right[:, ts])


if __name__ == '__main__':

    import time
    right_leg = create_kinematic_chain('right_leg', NMF_SIZE, 'R')

    angles = pd.read_pickle(glob.glob(f'{data_path}/joint_angles*.pkl')[0])
    pos = pd.read_pickle(glob.glob(f'{data_path}/aligned*.pkl')[0])

    # cProfile.run('main(right_leg, angles, pos)')
    then = time.time()
    main(right_leg, angles, pos)
    print(time.time() - then)
