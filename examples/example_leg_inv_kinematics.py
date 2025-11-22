"""
    Example usage of leg inverse kinematics module.
    Note that running this script will take about 30 minutes.
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt

from seqikpy.kinematic_chain import KinematicChainSeq, KinematicChainGeneric
from seqikpy.leg_inverse_kinematics import LegInvKinSeq, LegInvKinGeneric
from seqikpy.body_config import neuromechfly_body_config
from seqikpy.utils import load_file, from_sdf, calculate_body_size


DATA_PATH = Path("../data/anipose_220525_aJO_Fly001_001/pose-3d")


f_path = DATA_PATH / "pose3d_aligned.pkl"

aligned_pos = load_file(f_path)

body_config = neuromechfly_body_config.deepcopy()

LOAD_FROM_SDF = True
# If you want to load the template from an sdf file
if LOAD_FROM_SDF:
    sdf_path = Path("../tests/nmf_example.sdf")
    body_config.template, body_config.dof_bounds_rad = from_sdf(sdf_path)

start = time.time()

# Sequential inverse kinematics
seq_ik = LegInvKinSeq(
    aligned_pos=aligned_pos,
    kinematic_chain_class=KinematicChainSeq(
        bounds_dof=body_config.dof_bounds_rad,
        legs_list=["RF", "LF"],
        body_size=calculate_body_size(body_config.template),
    ),
    initial_angles=body_config.initial_angles_rad,
)

leg_joint_angles_seq, forward_kinematics_seq = seq_ik.run_ik_and_fk(
    export_path=DATA_PATH, hide_progress_bar=True
)

end = time.time()
total_time = (end - start) / 60.0

print(f"Sequential IK took {total_time} mins")

start = time.time()

# Traditional inverse kinematics
gen_ik = LegInvKinGeneric(
    aligned_pos=aligned_pos,
    kinematic_chain_class=KinematicChainGeneric(
        bounds_dof=body_config.dof_bounds_rad,
        legs_list=["RF", "LF"],
        body_size=None,
    ),
    initial_angles=body_config.initial_angles_rad,
)

leg_joint_angles_gen, forward_kinematics_gen = gen_ik.run_ik_and_fk(
    export_path=DATA_PATH, hide_progress_bar=True
)

end = time.time()
total_time = (end - start) / 60.0

print(f"Traditional IK took {total_time} mins")

# Compare the joint angles
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for key in leg_joint_angles_seq:
    plt.plot(leg_joint_angles_seq[key], label=key[6:], lw=2)
    plt.plot(leg_joint_angles_gen[key], ls=":", lw=2)


plt.xlabel("Frames (AU)")
plt.ylabel("Angles (rad)")
plt.title("Leg joint angles from SeqIK (solid) and GenIK (dotted)")
plt.legend()
plt.grid(True)
plt.show()
