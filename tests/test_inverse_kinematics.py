""" Test inverse kinematics tools """

import pytest
from pathlib import Path
import numpy as np

import seqikpy
from seqikpy.kinematic_chain import KinematicChainSeq
from seqikpy.leg_inverse_kinematics import LegInvKinSeq
from seqikpy.utils import load_file, from_sdf, calculate_body_size

PKG_PATH = Path(seqikpy.__path__[0]).parent


@pytest.fixture
def bounds():
    return {
        "RF_ThC_roll": (-np.pi, np.pi),
        "RF_ThC_yaw": (-np.pi, np.pi),
        "RF_ThC_pitch": (-np.pi, np.pi),
        "RF_CTr_pitch": (-np.pi, np.pi),
        "RF_CTr_roll": (-np.pi, np.pi),
        "RF_FTi_pitch": (-np.pi, np.pi),
        "RF_TiTa_pitch": (-np.pi, np.pi),
    }


@pytest.fixture
def initial_angles():
    return {
        "RF": {
            # Base ThC yaw pitch CTr pitch
            "stage_1": np.array([0.0, 0.0, 0.0, 0.0]),
            # Base ThC yaw pitch roll CTr pitch CTr roll
            "stage_2": np.array([0.0, 0.0, 0.0, 0.0, np.deg2rad(-90), 0.0]),
            # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch
            "stage_3": np.array(
                [0.0, 0.0, 0.0, 0.0, np.deg2rad(-90), 0.0, np.deg2rad(90), 0]
            ),
            # Base ThC yaw pitch roll CTr pitch CTr roll FTi pitch TiTa pitch
            "stage_4": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    np.deg2rad(-90),
                    0.0,
                    np.deg2rad(90),
                    np.deg2rad(-90),
                    0.0,
                ]
            ),
        }
    }


@pytest.fixture
def body_template():
    return {
        "RF_Coxa": np.array([0.35, -0.27, 0.400]),
        "RF_Femur": np.array([0.35, -0.27, -0.025]),
        "RF_Tibia": np.array([0.35, -0.27, -0.731]),
        "RF_Tarsus": np.array([0.35, -0.27, -1.249]),
        "RF_Claw": np.array([0.35, -0.27, -1.912]),
        "LF_Coxa": np.array([0.35, 0.27, 0.400]),
        "LF_Femur": np.array([0.35, 0.27, -0.025]),
        "LF_Tibia": np.array([0.35, 0.27, -0.731]),
        "LF_Tarsus": np.array([0.35, 0.27, -1.249]),
        "LF_Claw": np.array([0.35, 0.27, -1.912]),
        "RM_Coxa": np.array([0, -0.125, 0]),
        "RM_Femur": np.array([0, -0.125, -0.182]),
        "RM_Tibia": np.array([0, -0.125, -0.965]),
        "RM_Tarsus": np.array([0, -0.125, -1.633]),
        "RM_Claw": np.array([0, -0.125, -2.328]),
        "LM_Coxa": np.array([0, 0.125, 0]),
        "LM_Femur": np.array([0, 0.125, -0.182]),
        "LM_Tibia": np.array([0, 0.125, -0.965]),
        "LM_Tarsus": np.array([0, 0.125, -1.633]),
        "LM_Claw": np.array([0, 0.125, -2.328]),
        "RH_Coxa": np.array([-0.215, -0.087, -0.073]),
        "RH_Femur": np.array([-0.215, -0.087, -0.272]),
        "RH_Tibia": np.array([-0.215, -0.087, -1.108]),
        "RH_Tarsus": np.array([-0.215, -0.087, -1.793]),
        "RH_Claw": np.array([-0.215, -0.087, -2.588]),
        "LH_Coxa": np.array([-0.215, 0.087, -0.073]),
        "LH_Femur": np.array([-0.215, 0.087, -0.272]),
        "LH_Tibia": np.array([-0.215, 0.087, -1.108]),
        "LH_Tarsus": np.array([-0.215, 0.087, -1.793]),
        "LH_Claw": np.array([-0.215, 0.087, -2.588]),
    }


@pytest.fixture
def fake_3d_pose():
    fake_right_leg_pos = np.array(
        [
            # Thorax-coxa joint
            [0.35, -0.27, 0.400],
            # Coxa-femur joint, assume it is right below the thorax-coxa joint
            [0.35, -0.27, -0.025],
            # Femur-tibia, 90 degree pitched
            [0.35 + 0.706, -0.27, -0.025],
            # Tibia-tarsus, 90 degree pitched
            [0.35 + 0.706, -0.27, -0.543],
            # Tarsus-tip, 90 degree pitched
            [0.35 + 0.706 + 0.663, -0.27, -0.543],
        ]
    )
    fake_right_leg_pos = np.tile(fake_right_leg_pos, (10, 1, 1))
    fake_aligned_pos = {"RF_leg": fake_right_leg_pos}

    return fake_aligned_pos


@pytest.fixture
def true_joint_angles():
    return {
        "Angle_RF_ThC_roll": np.deg2rad(0),
        "Angle_RF_ThC_pitch": np.deg2rad(0),
        "Angle_RF_ThC_yaw": np.deg2rad(0),
        "Angle_RF_CTr_pitch": np.deg2rad(-90),
        "Angle_RF_CTr_roll": np.deg2rad(0),
        "Angle_RF_FTi_pitch": np.deg2rad(90),
        "Angle_RF_TiTa_pitch": np.deg2rad(-90),
    }


def test_from_sdf(body_template):
    sdf_path = PKG_PATH / "tests/nmf_example.sdf"
    nmf_template, _ = from_sdf(sdf_path)

    assert any(key in body_template for key in nmf_template)

    for key in body_template:
        assert np.isclose(nmf_template[key], body_template[key]).all()


def test_seq_ik(fake_3d_pose, body_template, bounds, initial_angles, true_joint_angles):
    # Sequential inverse kinematics
    seq_ik = LegInvKinSeq(
        aligned_pos=fake_3d_pose,
        kinematic_chain_class=KinematicChainSeq(
            bounds_dof=bounds,
            legs_list=["RF"],
            body_size=calculate_body_size(body_template),
        ),
        initial_angles=initial_angles,
    )

    assert hasattr(seq_ik, "calculate_ik_stage")
    assert hasattr(seq_ik, "run_ik_and_fk")

    # Test unknown entries
    with pytest.raises(ValueError):
        seq_ik.calculate_ik_stage(
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            initial_angles,
            segment_name="AB",
        )

    with pytest.raises(ValueError):
        seq_ik.calculate_ik_stage(
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            initial_angles,
            segment_name="RF",
            stage=5,
        )

    leg_joint_angles_seq, forward_kinematics_seq = seq_ik.run_ik_and_fk(
        export_path=None, hide_progress_bar=True
    )

    # See if the expected results are close to the actual results
    for key in leg_joint_angles_seq:
        assert np.isclose(
            leg_joint_angles_seq[key], true_joint_angles[key], atol=1e-2
        ).all()
