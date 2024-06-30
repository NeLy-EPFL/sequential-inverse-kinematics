""" Test kinematic chain generation"""
import pytest

from seqikpy.kinematic_chain import KinematicChainSeq, KinematicChainGeneric
from seqikpy.data import BOUNDS, INITIAL_ANGLES
from seqikpy.utils import load_file


@pytest.fixture
def setup_kinematic_chain_generic():
    return KinematicChainGeneric(BOUNDS, ["RF", "LF"], None)


@pytest.fixture
def setup_kinematic_chain_seq():
    return KinematicChainSeq(BOUNDS, ["RF", "LF"], None)


@pytest.mark.parametrize('leg_name', ['RF', 'LF'])
def test_kin_chain_seq(leg_name, setup_kinematic_chain_seq):

    leg_angles = load_file("leg_joint_angles.pkl")

    kin_chain = setup_kinematic_chain_seq

    assert hasattr(kin_chain, 'bounds_dof')
    assert hasattr(kin_chain, 'body_size')
    assert hasattr(kin_chain, 'create_leg_chain_stage_1')
    assert hasattr(kin_chain, 'create_leg_chain_stage_2')
    assert hasattr(kin_chain, 'create_leg_chain_stage_3')
    assert hasattr(kin_chain, 'create_leg_chain_stage_4')

    leg_chain_stage1 = kin_chain.create_leg_chain(leg_name, stage=1)
    chain1_link_names = [link.name for link in leg_chain_stage1.links]
    assert set(chain1_link_names) == {
        "Base link",
        f"{leg_name}_ThC_yaw",
        f"{leg_name}_ThC_pitch",
        f"{leg_name}_CTr_pitch"}
    leg_chain_stage2 = kin_chain.create_leg_chain(leg_name, angles=leg_angles, stage=2)
    chain2_link_names = [link.name for link in leg_chain_stage2.links]
    assert set(chain2_link_names) == {
        "Base link",
        f"{leg_name}_ThC_yaw",
        f"{leg_name}_ThC_pitch",
        f"{leg_name}_ThC_roll",
        f"{leg_name}_CTr_pitch",
        f"{leg_name}_FTi_pitch"}
    leg_chain_stage3 = kin_chain.create_leg_chain(leg_name, angles=leg_angles, stage=3)
    chain3_link_names = [link.name for link in leg_chain_stage3.links]
    assert set(chain3_link_names) == {
        "Base link",
        f"{leg_name}_ThC_yaw",
        f"{leg_name}_ThC_pitch",
        f"{leg_name}_ThC_roll",
        f"{leg_name}_CTr_pitch",
        f"{leg_name}_CTr_roll",
        f"{leg_name}_FTi_pitch",
        f"{leg_name}_TiTa_pitch"}

    leg_chain_stage4 = kin_chain.create_leg_chain(leg_name, angles=leg_angles, stage=4)
    chain4_link_names = [link.name for link in leg_chain_stage4.links]
    assert set(chain4_link_names) == {
        "Base link",
        f"{leg_name}_ThC_yaw",
        f"{leg_name}_ThC_pitch",
        f"{leg_name}_ThC_roll",
        f"{leg_name}_CTr_pitch",
        f"{leg_name}_CTr_roll",
        f"{leg_name}_FTi_pitch",
        f"{leg_name}_TiTa_pitch",
        f"{leg_name}_Claw"
    }

    # check invalid name
    with pytest.raises(ValueError):
        kin_chain.create_leg_chain("XX", stage=1)

    # check invalid stage
    with pytest.raises(ValueError):
        kin_chain.create_leg_chain(leg_name, stage=5)


@pytest.mark.parametrize('leg_name', ['RF', 'LF'])
def test_kin_chain_generic(leg_name, setup_kinematic_chain_generic):
    kin_chain = setup_kinematic_chain_generic

    assert hasattr(kin_chain, 'bounds_dof')
    assert hasattr(kin_chain, 'body_size')
    assert hasattr(kin_chain, 'create_leg_chain')
    # check leg chain
    leg_chain = kin_chain.create_leg_chain(leg_name)
    link_names = [link.name for link in leg_chain.links]
    assert set(link_names) == {
        "Base link",
        f"{leg_name}_ThC_yaw",
        f"{leg_name}_ThC_pitch",
        f"{leg_name}_ThC_roll",
        f"{leg_name}_CTr_pitch",
        f"{leg_name}_CTr_roll",
        f"{leg_name}_FTi_pitch",
        f"{leg_name}_TiTa_pitch",
        f"{leg_name}_Claw"
    }

    # check invalid name
    with pytest.raises(ValueError):
        kin_chain.create_leg_chain("XX")
