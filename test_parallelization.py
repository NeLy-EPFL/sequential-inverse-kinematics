#!/usr/bin/env python3
"""
Test script to verify the parallelized inverse kinematics implementation.
"""
import numpy as np
import time
from pathlib import Path

# Add the seqikpy module to path
import sys
sys.path.append('/home/sibwang/Projects/sequential-inverse-kinematics')

from seqikpy.leg_inverse_kinematics import LegInvKinSeq, LegInvKinGeneric
from seqikpy.kinematic_chain import KinematicChainSeq, KinematicChainGeneric
from seqikpy.data import BOUNDS, INITIAL_ANGLES
from seqikpy.utils import load_file
from seqikpy.alignment import calculate_body_size
from seqikpy.data import NMF_TEMPLATE

def create_test_data():
    """Create simple test data for validation."""
    n_frames = 100
    n_joints = 5
    n_legs = 6
    
    aligned_pos = {}
    leg_names = ["RF", "LF", "RM", "LM", "RH", "LH"]
    
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

def test_sequential_vs_parallel():
    """Test that sequential and parallel processing give the same results."""
    print("Creating test data...")
    aligned_pos = create_test_data()
    
    # Create kinematic chain
    body_size = calculate_body_size(NMF_TEMPLATE, ["RF", "LF"])
    
    kinematic_chain_seq = KinematicChainSeq(
        bounds_dof=BOUNDS,
        legs_list=["RF", "LF"],  # Test with 2 legs only for speed
        body_size=body_size,
    )
    
    # Test LegInvKinSeq
    print("\nTesting LegInvKinSeq...")
    
    # Sequential processing
    seq_ik_sequential = LegInvKinSeq(
        aligned_pos={k: v for k, v in aligned_pos.items() if k in ["RF_leg", "LF_leg"]},
        kinematic_chain_class=kinematic_chain_seq,
        initial_angles=INITIAL_ANGLES,
    )
    
    print("Running sequential processing...")
    start_time = time.time()
    joint_angles_seq, fk_seq = seq_ik_sequential.run_ik_and_fk(
        stages=[1, 2], hide_progress_bar=True, n_workers=1
    )
    sequential_time = time.time() - start_time
    
    # Parallel processing
    seq_ik_parallel = LegInvKinSeq(
        aligned_pos={k: v for k, v in aligned_pos.items() if k in ["RF_leg", "LF_leg"]},
        kinematic_chain_class=kinematic_chain_seq,
        initial_angles=INITIAL_ANGLES,
    )
    
    print("Running parallel processing...")
    start_time = time.time()
    joint_angles_par, fk_par = seq_ik_parallel.run_ik_and_fk(
        stages=[1, 2], hide_progress_bar=True, n_workers=-1
    )
    parallel_time = time.time() - start_time
    
    # Compare results
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Check if results are the same
    assert len(joint_angles_seq) == len(joint_angles_par), "Different number of joint angles"
    
    for key in joint_angles_seq:
        if key in joint_angles_par:
            diff = np.max(np.abs(joint_angles_seq[key] - joint_angles_par[key]))
            print(f"Max difference in {key}: {diff:.2e}")
            assert diff < 1e-10, f"Results differ for {key}: {diff}"
    
    print("✓ LegInvKinSeq: Sequential and parallel results match!")
    
    # Test LegInvKinGeneric
    print("\nTesting LegInvKinGeneric...")
    
    kinematic_chain_gen = KinematicChainGeneric(
        bounds_dof=BOUNDS,
        legs_list=["RF", "LF"],
        body_size=body_size,
    )
    
    # Sequential processing
    gen_ik_sequential = LegInvKinGeneric(
        aligned_pos={k: v for k, v in aligned_pos.items() if k in ["RF_leg", "LF_leg"]},
        kinematic_chain_class=kinematic_chain_gen,
        initial_angles=INITIAL_ANGLES,
    )
    
    print("Running sequential processing...")
    start_time = time.time()
    joint_angles_seq_gen, fk_seq_gen = gen_ik_sequential.run_ik_and_fk(
        hide_progress_bar=True, n_workers=1
    )
    sequential_time_gen = time.time() - start_time
    
    # Parallel processing
    gen_ik_parallel = LegInvKinGeneric(
        aligned_pos={k: v for k, v in aligned_pos.items() if k in ["RF_leg", "LF_leg"]},
        kinematic_chain_class=kinematic_chain_gen,
        initial_angles=INITIAL_ANGLES,
    )
    
    print("Running parallel processing...")
    start_time = time.time()
    joint_angles_par_gen, fk_par_gen = gen_ik_parallel.run_ik_and_fk(
        hide_progress_bar=True, n_workers=-1
    )
    parallel_time_gen = time.time() - start_time
    
    # Compare results
    print(f"Sequential time: {sequential_time_gen:.3f}s")
    print(f"Parallel time: {parallel_time_gen:.3f}s")
    print(f"Speedup: {sequential_time_gen/parallel_time_gen:.2f}x")
    
    # Check if results are the same
    assert len(joint_angles_seq_gen) == len(joint_angles_par_gen), "Different number of joint angles"
    
    for key in joint_angles_seq_gen:
        if key in joint_angles_par_gen:
            diff = np.max(np.abs(joint_angles_seq_gen[key] - joint_angles_par_gen[key]))
            print(f"Max difference in {key}: {diff:.2e}")
            assert diff < 1e-10, f"Results differ for {key}: {diff}"
    
    print("✓ LegInvKinGeneric: Sequential and parallel results match!")
    
    print("\n✅ All tests passed! Parallelization working correctly.")

if __name__ == "__main__":
    test_sequential_vs_parallel()