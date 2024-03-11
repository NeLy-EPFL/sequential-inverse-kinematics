""" Example usage of head inverse kinematics module. """

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from seqikpy.head_inverse_kinematics import HeadInverseKinematics
from seqikpy.data import NMF_TEMPLATE
from seqikpy.utils import load_file

DATA_PATH = Path('../data/anipose_220525_aJO_Fly001_001/pose-3d')

f_path = DATA_PATH / "pose3d_aligned.pkl"

data = load_file(f_path)

class_hk = HeadInverseKinematics(
    aligned_pos=data,
    body_template=NMF_TEMPLATE,
)
joint_angles = class_hk.compute_head_angles(
    export_path=DATA_PATH,
    compute_ant_angles=True
)


fig, ax = plt.subplots()

time_step = 1e-2
time = np.arange(0, joint_angles['Angle_head_roll'].shape[0], 1) * time_step

for kp, angle in joint_angles.items():
    ax.plot(time, np.rad2deg(angle), label=kp[6:].replace('_', ' '))

plt.xlabel('Time (sec)')
plt.ylabel('Angles (deg)')
plt.title('Head joint angles')
plt.legend()
plt.grid(True)
plt.show()
