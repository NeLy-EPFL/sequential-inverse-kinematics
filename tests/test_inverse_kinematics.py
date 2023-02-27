import pytest
import numpy as np

from nmf_ik.alignment import AlignPose
from nmf_ik.inverse_kinematics_ap import InverseKinematics


@pytest.fixture
def main_folder():
    return '../data/anipose/failed_case/pose-3d'


def test_inheritence(main_folder):
    align = AlignPose(main_folder)
    aligned_pose = align.align_pose(False)

    ik = InverseKinematics(aligned_pose)

    assert hasattr(ik, 'create_kinematic_chain_leg')
    assert hasattr(ik, 'create_kinematic_chain_head')
