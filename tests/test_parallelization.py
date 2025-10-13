"""Test parallelization functionality of inverse kinematics."""

import pytest
import numpy as np
import time
from pathlib import Path

import seqikpy
from seqikpy.leg_inverse_kinematics import LegInvKinSeq, LegInvKinGeneric
from seqikpy.kinematic_chain import KinematicChainSeq, KinematicChainGeneric
from seqikpy.data import BOUNDS, INITIAL_ANGLES
from seqikpy.utils import calculate_body_size
from seqikpy.data import NMF_TEMPLATE

PKG_PATH = Path(seqikpy.__path__[0]).parent


@pytest.fixture
def synthetic_pose_data():
    """Create synthetic test data for validation."""
    n_frames = 50  # Smaller for faster tests
    n_joints = 5
    
    aligned_pos = {}
    leg_names = ["RF", "LF"]  # Test with 2 legs for speed
    
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    for leg_name in leg_names:
        # Create synthetic 3D pose data for each leg
        # Shape: (n_frames, n_joints, 3)
        positions = np.random.randn(n_frames, n_joints, 3) * 0.1
        
        # Add some realistic structure - joints should be connected
        for t in range(n_frames):
            for joint in range(1, n_joints):
                positions[t, joint] = positions[t, joint-1] + positions[t, joint]
        
        aligned_pos[f"{leg_name}_leg"] = positions
    
    return aligned_pos


@pytest.fixture
def test_body_size():
    """Create body size for test legs."""
    return calculate_body_size(NMF_TEMPLATE, ["RF", "LF"])


class TestParallelization:
    """Test class for parallelization functionality."""
    
    def test_sequential_ik_parallelization(self, synthetic_pose_data, test_body_size):
        """Test that LegInvKinSeq gives identical results for sequential vs parallel processing."""
        
        kinematic_chain_seq = KinematicChainSeq(
            bounds_dof=BOUNDS,
            legs_list=["RF", "LF"],
            body_size=test_body_size,
        )
        
        # Sequential processing
        seq_ik_sequential = LegInvKinSeq(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        joint_angles_seq, fk_seq = seq_ik_sequential.run_ik_and_fk(
            stages=[1, 2, 3, 4], hide_progress_bar=True, n_jobs=1
        )
        
        # Parallel processing
        seq_ik_parallel = LegInvKinSeq(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        joint_angles_par, fk_par = seq_ik_parallel.run_ik_and_fk(
            stages=[1, 2, 3, 4], hide_progress_bar=True, n_jobs=-1
        )
        
        # Verify results are identical
        assert len(joint_angles_seq) == len(joint_angles_par), "Different number of joint angles"
        
        for key in joint_angles_seq:
            if key in joint_angles_par:
                np.testing.assert_array_almost_equal(
                    joint_angles_seq[key], 
                    joint_angles_par[key],
                    decimal=10,
                    err_msg=f"Results differ for {key}"
                )
        
        # Verify forward kinematics results are also identical
        assert len(fk_seq) == len(fk_par), "Different number of forward kinematics results"
        
        for key in fk_seq:
            if key in fk_par:
                np.testing.assert_array_almost_equal(
                    fk_seq[key],
                    fk_par[key],
                    decimal=10,
                    err_msg=f"Forward kinematics differ for {key}"
                )
    
    def test_generic_ik_parallelization(self, synthetic_pose_data, test_body_size):
        """Test that LegInvKinGeneric gives identical results for sequential vs parallel processing."""
        
        kinematic_chain_gen = KinematicChainGeneric(
            bounds_dof=BOUNDS,
            legs_list=["RF", "LF"],
            body_size=test_body_size,
        )
        
        # Sequential processing
        gen_ik_sequential = LegInvKinGeneric(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_gen,
            initial_angles=INITIAL_ANGLES,
        )
        
        joint_angles_seq, fk_seq = gen_ik_sequential.run_ik_and_fk(
            hide_progress_bar=True, n_jobs=1
        )
        
        # Parallel processing
        gen_ik_parallel = LegInvKinGeneric(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_gen,
            initial_angles=INITIAL_ANGLES,
        )
        
        joint_angles_par, fk_par = gen_ik_parallel.run_ik_and_fk(
            hide_progress_bar=True, n_jobs=-1
        )
        
        # Verify results are identical
        assert len(joint_angles_seq) == len(joint_angles_par), "Different number of joint angles"
        
        for key in joint_angles_seq:
            if key in joint_angles_par:
                np.testing.assert_array_almost_equal(
                    joint_angles_seq[key], 
                    joint_angles_par[key],
                    decimal=10,
                    err_msg=f"Results differ for {key}"
                )
        
        # Verify forward kinematics results are also identical
        assert len(fk_seq) == len(fk_par), "Different number of forward kinematics results"
        
        for key in fk_seq:
            if key in fk_par:
                np.testing.assert_array_almost_equal(
                    fk_seq[key],
                    fk_par[key],
                    decimal=10,
                    err_msg=f"Forward kinematics differ for {key}"
                )
    
    def test_n_jobs_parameter_values(self, synthetic_pose_data, test_body_size):
        """Test different values of n_jobs parameter."""
        
        kinematic_chain_seq = KinematicChainSeq(
            bounds_dof=BOUNDS,
            legs_list=["RF", "LF"],
            body_size=test_body_size,
        )
        
        seq_ik = LegInvKinSeq(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        # Test with n_jobs=1 (sequential)
        joint_angles_1, _ = seq_ik.run_ik_and_fk(
            stages=[1], hide_progress_bar=True, n_jobs=1
        )
        
        # Test with n_jobs=2 (limited parallel)
        seq_ik_2 = LegInvKinSeq(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        joint_angles_2, _ = seq_ik_2.run_ik_and_fk(
            stages=[1], hide_progress_bar=True, n_jobs=2
        )
        
        # Test with n_jobs=-1 (all cores)
        seq_ik_all = LegInvKinSeq(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        joint_angles_all, _ = seq_ik_all.run_ik_and_fk(
            stages=[1], hide_progress_bar=True, n_jobs=-1
        )
        
        # All should give identical results
        for key in joint_angles_1:
            np.testing.assert_array_almost_equal(
                joint_angles_1[key], joint_angles_2[key], decimal=10
            )
            np.testing.assert_array_almost_equal(
                joint_angles_1[key], joint_angles_all[key], decimal=10
            )
    
    def test_single_leg_processing(self, test_body_size):
        """Test that parallelization works correctly with a single leg."""
        
        # Create data for only one leg
        np.random.seed(42)
        single_leg_data = {
            "RF_leg": np.random.randn(20, 5, 3) * 0.1
        }
        
        kinematic_chain_seq = KinematicChainSeq(
            bounds_dof=BOUNDS,
            legs_list=["RF"],
            body_size={k: v for k, v in test_body_size.items() if "RF" in k},
        )
        
        seq_ik_sequential = LegInvKinSeq(
            aligned_pos=single_leg_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        joint_angles_seq, _ = seq_ik_sequential.run_ik_and_fk(
            stages=[1], hide_progress_bar=True, n_jobs=1
        )
        
        seq_ik_parallel = LegInvKinSeq(
            aligned_pos=single_leg_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        joint_angles_par, _ = seq_ik_parallel.run_ik_and_fk(
            stages=[1], hide_progress_bar=True, n_jobs=-1
        )
        
        # Results should be identical
        for key in joint_angles_seq:
            np.testing.assert_array_almost_equal(
                joint_angles_seq[key], joint_angles_par[key], decimal=10
            )
    
    def test_empty_leg_list_handling(self, test_body_size):
        """Test that the code handles empty leg lists gracefully."""
        
        # Create data with no leg segments
        no_leg_data = {
            "head": np.random.randn(10, 3, 3),
            "antenna": np.random.randn(10, 2, 3),
        }
        
        kinematic_chain_seq = KinematicChainSeq(
            bounds_dof=BOUNDS,
            legs_list=["RF", "LF"],
            body_size=test_body_size,
        )
        
        seq_ik = LegInvKinSeq(
            aligned_pos=no_leg_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        joint_angles, fk = seq_ik.run_ik_and_fk(
            stages=[1], hide_progress_bar=True, n_jobs=-1
        )
        
        # Should return empty dictionaries
        assert len(joint_angles) == 0, "Should have no joint angles for non-leg data"
        assert len(fk) == 0, "Should have no forward kinematics for non-leg data"


@pytest.mark.performance
class TestPerformance:
    """Performance tests for parallelization (marked as performance to allow skipping)."""
    
    def test_parallelization_performance(self, synthetic_pose_data, test_body_size):
        """Test that parallelization provides performance improvement (when multiple legs present)."""
        
        # Only run if we have multiple legs
        leg_count = len([k for k in synthetic_pose_data.keys() if "leg" in k.lower()])
        if leg_count <= 1:
            pytest.skip("Need multiple legs for meaningful performance test")
        
        kinematic_chain_seq = KinematicChainSeq(
            bounds_dof=BOUNDS,
            legs_list=["RF", "LF"],
            body_size=test_body_size,
        )
        
        # Sequential processing
        seq_ik_sequential = LegInvKinSeq(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        start_time = time.time()
        seq_ik_sequential.run_ik_and_fk(
            stages=[1, 2, 3, 4], hide_progress_bar=True, n_jobs=1
        )
        sequential_time = time.time() - start_time
        
        # Parallel processing
        seq_ik_parallel = LegInvKinSeq(
            aligned_pos=synthetic_pose_data,
            kinematic_chain_class=kinematic_chain_seq,
            initial_angles=INITIAL_ANGLES,
        )
        
        start_time = time.time()
        seq_ik_parallel.run_ik_and_fk(
            stages=[1, 2, 3, 4], hide_progress_bar=True, n_jobs=-1
        )
        parallel_time = time.time() - start_time
        
        # Parallel should be at least as fast (allowing for some overhead)
        # In practice, it should be faster with multiple legs, but we don't enforce a strict speedup
        # since performance can vary based on system load and number of cores
        assert parallel_time <= sequential_time * 1.5, (
            f"Parallel processing ({parallel_time:.3f}s) should not be much slower "
            f"than sequential ({sequential_time:.3f}s)"
        )