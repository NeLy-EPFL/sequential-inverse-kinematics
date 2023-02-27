""" Example usage of head inverse kinematics module. """
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nmf_ik.head_inverse_kinematics import HeadInverseKinematics
from nmf_ik.data import NMF_TEMPLATE

DATA_PATH = Path('../data/anipose_220525_aJO_Fly001_001/pose-3d')

f_path = DATA_PATH / "pose3d_aligned.pkl"

with open(f_path, "rb") as f:
    data = pickle.load(f)

class_hk = HeadInverseKinematics(
    aligned_pos=data,
    nmf_template=NMF_TEMPLATE,
    angles_to_calculate=[
        'Angle_head_roll',
        'Angle_head_pitch',
        'Angle_head_yaw',
        'Angle_antenna_pitch_L',
        'Angle_antenna_pitch_R',
        'Angle_antenna_yaw_L',
        'Angle_antenna_yaw_R'
    ]
)
joint_angles = class_hk.compute_head_angles(export_path=DATA_PATH)


fig, ax = plt.subplots()

time_step = 1e-2
time = np.arange(0, joint_angles['Angle_head_roll'].shape[0], 1) * time_step

for kp, angle in joint_angles.items():
    ax.plot(time, np.rad2deg(angle), label=kp[6:].replace('_', ' '))

plt.xlabel('Time (sec)')
plt.ylabel('Angles(deg)')
plt.title('Head inverse kinematics')
plt.legend()
plt.grid(True)
plt.show()
