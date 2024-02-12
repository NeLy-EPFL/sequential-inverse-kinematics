""" Example usage of leg inverse kinematics module. """
import pickle
from pathlib import Path
import time

from seqikpy.kinematic_chain import KinematicChain
from seqikpy.leg_inverse_kinematics import LegInverseKinematics
from seqikpy.data import BOUNDS, INITIAL_ANGLES, NMF_TEMPLATE


DATA_PATH = Path('../data/anipose_220525_aJO_Fly001_001/pose-3d')

start = time.time()

f_path = DATA_PATH / "pose3d_aligned.pkl"

with open(f_path, "rb") as f:
    aligned_pos = pickle.load(f)


seq_ik = LegInverseKinematics(
    aligned_pos=aligned_pos,
    kinematic_chain_class=KinematicChain(
        bounds_dof=BOUNDS,
        nmf_size=None,
    ),
    initial_angles=INITIAL_ANGLES
)

leg_joint_angles, forward_kinematics = seq_ik.run_ik_and_fk(export_path=DATA_PATH)

end = time.time()
total_time = (end - start) / 60.0

print(f'Total time taken to execute the code: {total_time} mins')
