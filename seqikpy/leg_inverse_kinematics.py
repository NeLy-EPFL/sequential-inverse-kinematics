"""Module to calculate inverse kinematics from 3D pose based on IKPy."""

import logging
import warnings
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union, Literal, Optional

import numpy as np
from tqdm import trange
from joblib import Parallel, delayed
from ikpy.chain import Chain

from seqikpy.utils import save_file, split_arrays_into_chunks, merge_chunks_into_arrays
from seqikpy.data import INITIAL_ANGLES
from seqikpy.kinematic_chain import (
    KinematicChainBase,
    KinematicChainSeq,
    KinematicChainGeneric,
)

# Ignore the warnings
warnings.filterwarnings("ignore")

_logger = logging.getLogger(__name__)


class LegInvKinBase(ABC):
    """Abstract class to calculate inverse kinematics for leg joints.

    Parameters
    ----------
    aligned_pos : Dict[str, np.ndarray]
        Aligned pose from the AlignPose class.
        Should have the following structure

        >>> pose_data_dict = {
                "<side><segment>_leg": np.ndarray[N_frames,N_key_points,3],
            }
    kinematic_chain : KinematicChainBase
        Kinematic chain of the leg.
    initial_angles : Dict[str, np.ndarray], optional
        Initial angles of DOFs.
        If not provided, the default values from data.py will be used.
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"
    """

    def __init__(
        self,
        aligned_pos: Dict[str, np.ndarray],
        kinematic_chain_class: KinematicChainBase,
        initial_angles: Optional[Dict[str, np.ndarray]] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        self.aligned_pos = aligned_pos
        self.kinematic_chain_class = kinematic_chain_class
        self.initial_angles = (
            INITIAL_ANGLES if initial_angles is None else initial_angles
        )

        # Get the logger of the module
        _logger = logging.getLogger(self.__class__.__name__)
        numeric_level = getattr(logging, log_level.upper(), None)
        _logger.setLevel(numeric_level)

    def calculate_ik(
        self,
        kinematic_chain: Chain,
        target_pos: np.ndarray,
        initial_angles: np.ndarray = None,
    ) -> np.ndarray:
        """Calculates the joint angles in the leg chain."""
        # don"t take the last and first ones into account
        return kinematic_chain.inverse_kinematics(
            target_position=target_pos, initial_position=initial_angles
        )

    def calculate_fk(
        self, kinematic_chain: Chain, joint_angles: np.ndarray
    ) -> np.ndarray:
        """Calculates the forward kinematics from the joint dof angles."""
        fk = kinematic_chain.forward_kinematics(joint_angles, full_kinematics=True)
        end_effector_positions = np.zeros((len(kinematic_chain.links), 3))
        for link in range(len(kinematic_chain.links)):
            end_effector_positions[link, :] = fk[link][:3, 3]
        return end_effector_positions

    def get_scale_factor(self, vector: np.ndarray, length: float) -> float:
        """Gets scale the ratio between two vectors."""
        vector_diff = np.linalg.norm(np.diff(vector, axis=0), axis=1)
        norm_sum = np.sum(vector_diff)
        return length / norm_sum

    @abstractmethod
    def calculate_ik_stage(
        self,
        end_effector_pos: np.ndarray,
        origin: np.ndarray,
        initial_angles: np.ndarray,
        segment_name: str,
        **kwargs,
    ) -> np.ndarray:
        """For a given trial pose data, calculates the inverse kinematics.

        Parameters
        ----------
        end_effector_pos : np.ndarray
            3D array containing the position of the end effector pos
        origin : np.ndarray
            Origin of the kinematic chain, i.e., Thorax-Coxa joint
        initial_angles : np.ndarray
            Initial angles for the optimization seed
        segment_name : str
            Leg side, i.e., RF, LF, RM, LM, RH, LH

        Returns
        -------
        np.ndarray
            Array containing the cartesian coordinates of the joint positions.
            The joint angles are saved in a class attribute
        """

    @abstractmethod
    def run_ik_and_fk(
        self, export_path: Union[Path, str] = None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Runs inverse and forward kinematics for leg joints.

        Parameters
        ----------
        export_path : Union[Path, str], optional
            Path where the results will be saved,
            if None, nothing is saveed, by default None

        Returns
        -------
        Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]
            Two dictionaries containing joint angles and forward
            kinematics, respectively.
        """


class LegInvKinSeq(LegInvKinBase):
    """Class to calculate inverse kinematics for leg joints in a
    sequential manner. This method finds the optimal joint angles
    within the bounds to match each koint as closely as possible.

    Parameters
    ----------
    aligned_pos : Dict[str, np.ndarray]
        Aligned pose from the AlignPose class.
        Should have the following structure

        >>> pose_data_dict = {
                "<side><segment>_leg": np.ndarray[N_frames,N_key_points,3],
            }
    kinematic_chain : KinematicChainSeq
        Kinematic chain of the leg.
    initial_angles : Dict[str, np.ndarray], optional
        Initial angles of DOFs.
        If not provided, the default values from data.py will be used.
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"

    Examples
    --------

    >>> from pathlib import Path
    >>> from seqikpy.kinematic_chain import KinematicChainSeq
    >>> from seqikpy.leg_inverse_kinematics import LegInvKinSeq
    >>> from seqikpy.data import BOUNDS, INITIAL_ANGLES
    >>> from seqikpy.utils import load_file

    >>> DATA_PATH = Path("../data/anipose_220525_aJO_Fly001_001/pose-3d")
    >>> f_path = DATA_PATH / "pose3d_aligned.pkl"

    >>> aligned_pos = load_file(f_path)

    >>> seq_ik = LegInvKinSeq(
            aligned_pos=aligned_pos,
            kinematic_chain_class=KinematicChainSeq(
                bounds_dof=BOUNDS,
                legs_list=["RF", "LF"],
                body_size=None,
            ),
            initial_angles=INITIAL_ANGLES
        )

    >>> leg_joint_angles, forward_kinematics = seq_ik.run_ik_and_fk(
            export_path=DATA_PATH,
            hide_progress_bar=False
        )

    """

    def __init__(
        self,
        aligned_pos: Dict[str, np.ndarray],
        kinematic_chain_class: KinematicChainSeq,
        initial_angles: Optional[Dict[str, np.ndarray]] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        super().__init__(aligned_pos, kinematic_chain_class, initial_angles, log_level)
        # Create an empty dict for joint angles
        self.joint_angles_dict = {}

    @staticmethod
    def _process_single_leg_sequence(
        segment_name,
        segment_array,
        stages,
        kinematic_chain_class,
        initial_angles_dict,
        hide_progress_bar,
    ):
        """Process a single leg through all stages."""
        leg_name = segment_name.split("_")[0]

        if not leg_name in kinematic_chain_class.body_size:
            return segment_name, None, {}

        # Create a temporary instance to process this leg
        temp_instance = LegInvKinSeq.__new__(LegInvKinSeq)
        temp_instance.kinematic_chain_class = kinematic_chain_class
        temp_instance.initial_angles = initial_angles_dict
        temp_instance.joint_angles_dict = {}

        origin = segment_array[:, 0, :]
        forward_kinematics_result = None

        for stage in stages:
            end_effector_pos = segment_array[:, stage, :]
            initial_angles = initial_angles_dict[leg_name][f"stage_{stage}"]

            forward_kinematics_result = temp_instance.calculate_ik_stage(
                end_effector_pos=end_effector_pos,
                origin=origin,
                initial_angles=initial_angles,
                stage=stage,
                segment_name=leg_name,
                hide_progress_bar=hide_progress_bar,
            )

        return segment_name, forward_kinematics_result, temp_instance.joint_angles_dict

    def calculate_ik_stage(
        self,
        end_effector_pos: np.ndarray,
        origin: np.ndarray,
        initial_angles: np.ndarray,
        segment_name: str,
        **kwargs,
    ) -> np.ndarray:
        """For a given trial pose data, calculates the inverse kinematics.

        Parameters
        ----------
        end_effector_pos : np.ndarray
            3D array containing the position of the end effector pos
        origin : np.ndarray
            Origin of the kinematic chain, i.e., Thorax-Coxa joint
        initial_angles : np.ndarray
            Initial angles for the optimization seed
        segment_name (kwargs) : str
            Leg side, i.e., RF, LF, RM, LM, RH, LH
        stage (kwargs) : int
            Stage in the sequential calculation, should be between 1 and 4

        Returns
        -------
        np.ndarray
            Array containing the cartesian coordinates of the joint positions.
            The joint angles are saved in a class attribute
        """
        stage = kwargs.get("stage", 1)
        hide_progress_bar = kwargs.get("hide_progress_bar", False)

        if not segment_name in ["RF", "LF", "RM", "LM", "RH", "LH"]:
            raise ValueError(f"Segment name ({segment_name}) is not valid.")

        if not 1 <= stage <= 4:
            raise ValueError(f"Stage ({stage}) should be between 1 and 4.")

        frames_no = end_effector_pos.shape[0]
        end_effector_pos_diff = end_effector_pos - origin

        # Joint angles shape: (number of frames, number of DOFs)
        joint_angles = np.empty((frames_no, len(initial_angles)))
        # Forward kinematics shape: (number of frames, number of
        # joints in the kinematics chain, axes-x,y,z)
        forward_kinematics = np.empty((frames_no, len(initial_angles), 3))

        # if the origin is a vector, convert it to a matrix
        if origin.size == 3:
            origin = np.tile(origin, (frames_no, 1))

        # Get the kinematic chain for Stage 1
        if stage == 1:
            kinematic_chain = self.kinematic_chain_class.create_leg_chain(
                stage=stage,
                leg_name=segment_name,
            )

        # Start the inverse kinematics calculation
        for t in trange(
            frames_no,
            disable=hide_progress_bar,
            desc=f"{segment_name} stage {stage}",
        ):
            # Get the kinematic chain for the other stages
            if stage in [2, 3, 4]:
                kinematic_chain = self.kinematic_chain_class.create_leg_chain(
                    leg_name=segment_name,
                    stage=stage,
                    angles=self.joint_angles_dict,
                    t=t,
                )
                # from IPython import embed; embed()

            # For the first frame, use the given initial angles, for the rest
            # use the calculated joint angles from the previous time step
            initial_angles = initial_angles if t == 0 else joint_angles[t - 1, :]
            # Calculate the inverse kinematics
            joint_angles[t, :] = self.calculate_ik(
                kinematic_chain, end_effector_pos_diff[t, :], initial_angles
            )

            # Calculate the forward kinematics for the last stage only
            if stage == 4:
                forward_kinematics[t, :] = (
                    self.calculate_fk(kinematic_chain, joint_angles[t, :])
                    + origin[t, :]
                )

        # Link names
        link_names = [link.name for link in kinematic_chain.links]

        # Store the joint angles based on the stage number
        # Stage 1: Thorax-Coxa pitch and yaw
        if stage == 1:
            self.joint_angles_dict[f"Angle_{segment_name}_ThC_yaw"] = joint_angles[
                :, link_names.index(f"{segment_name}_ThC_yaw")
            ]
            self.joint_angles_dict[f"Angle_{segment_name}_ThC_pitch"] = joint_angles[
                :, link_names.index(f"{segment_name}_ThC_pitch")
            ]
            _logger.debug("Stage 1 is completed!")
        # Stage 2: Thorax-Coxa roll, Coxa-Trochanter pitch
        elif stage == 2:
            self.joint_angles_dict[f"Angle_{segment_name}_ThC_roll"] = joint_angles[
                :, link_names.index(f"{segment_name}_ThC_roll")
            ]
            self.joint_angles_dict[f"Angle_{segment_name}_CTr_pitch"] = joint_angles[
                :, link_names.index(f"{segment_name}_CTr_pitch")
            ]
            _logger.debug("Stage 2 is completed!")
        # Stage 3: Coxa-Trochanter roll, Femur-Tibia pitch
        elif stage == 3:
            self.joint_angles_dict[f"Angle_{segment_name}_CTr_roll"] = joint_angles[
                :, link_names.index(f"{segment_name}_CTr_roll")
            ]
            self.joint_angles_dict[f"Angle_{segment_name}_FTi_pitch"] = joint_angles[
                :, link_names.index(f"{segment_name}_FTi_pitch")
            ]
            _logger.debug("Stage 3 is completed!")
        # Stage 4: Tibia-Tarsus pitch
        elif stage == 4:
            self.joint_angles_dict[f"Angle_{segment_name}_TiTa_pitch"] = joint_angles[
                :, link_names.index(f"{segment_name}_TiTa_pitch")
            ]
            _logger.debug("Stage 4 is completed!")

        return forward_kinematics

    def run_ik_and_fk(
        self,
        export_path: Union[Path, str] = None,
        stages: list[int] = [1, 2, 3, 4],
        n_workers: int = 1,
        parallel_over_time: bool = True,
        chunk_overlap: int = 20,
        avg_workloads_per_worker: int = 2,
        min_chunk_size: int = 100,
        hide_progress_bar: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Runs inverse and forward kinematics for leg joints.

        Parameters
        ----------
        export_path : Union[Path, str], optional
            Path where the results will be saved,
            if None, nothing is saveed, by default None
        stages (kwargs) : List[int], optional
            Stages to run the inverse kinematics. Default is all stages
            ([1, 2, 3, 4]).
        n_workers (kwargs) : int, optional
            Number of parallel jobs for leg processing. -1 uses all cores,
            1 disables parallelization, by default 1.
        parallel_over_time (kwargs) : bool, optional
            Ignored unless n_workers > 1. This flag determins whether to
            parallelize over time and legs or legs only. By default True.
        avg_workloads_per_worker (kwargs) : int, optional
            Ignored unless n_workers > 1 and parallel_over_time is True.
            This is the rough number of workloads to be assigned to each
            worker. Larger numbers lead to load balancing at the cost of
            higher overhead. This number is approximate - scheduler will
            change it for better rounding. By default 2.
        min_chunk_size (kwargs) : int, optional
            Ignored unless n_workers > 1 and parallel_over_time is True.
            Minimum chunk size (in frames) for each time series payload.
            If the time series is shorter than this size, the number of
            workers will be reduced accordingly. By default 100.
        hide_progress_bar (kwargs) : Optional[bool], optional
            Hide the progress bar, by default False.

        Returns
        -------
        Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]
            Two dictionaries containing joint angles and forward
            kinematics, respectively.
        """
        if max(stages) > 4 or not all(np.diff(stages) == 1):
            raise ValueError(
                "Maximum stage number is 4 and the list should be strictly incremental."
            )

        forward_kinematics_dict = {}

        # Get parallel processing parameters
        _logger.info("Computing joint angles and forward kinematics...")

        # Filter leg segments
        leg_segments = [
            (name, array)  # each array has shape (N_frames, N_key_points, 3)
            for name, array in self.aligned_pos.items()
            if "leg" in name.lower()
        ]
        seq_length = leg_segments[0][1].shape[0]

        if n_workers == 1 or len(leg_segments) <= 1:
            # Sequential processing
            for segment_name, segment_array in leg_segments:
                result_name, result_fk, local_joint_angles = (
                    self._process_single_leg_sequence(
                        segment_name,
                        segment_array,
                        stages,
                        self.kinematic_chain_class,
                        self.initial_angles,
                        hide_progress_bar,
                    )
                )
                if result_fk is not None:
                    forward_kinematics_dict[result_name] = result_fk
                    self.joint_angles_dict.update(local_joint_angles)

        elif not parallel_over_time:
            # Parallel processing over legs only (each leg gets a process)
            results = Parallel(n_jobs=n_workers)(
                delayed(self._process_single_leg_sequence)(
                    segment_name,
                    segment_array,
                    stages,
                    self.kinematic_chain_class,
                    self.initial_angles,
                    hide_progress_bar,
                )
                for segment_name, segment_array in leg_segments
            )
            # Collect results
            for segment_name, result_fk, local_joint_angles in results:
                if result_fk is not None:
                    forward_kinematics_dict[segment_name] = result_fk
                    self.joint_angles_dict.update(local_joint_angles)

        else:
            # Parallel processing over legs and time (time series is split into chunks)
            # Figure out number of frames per payload
            parallel_executor = Parallel(n_jobs=n_workers)
            n_workers_effective = parallel_executor._effective_n_jobs()
            input_chunks = split_arrays_into_chunks(
                [array for name, array in leg_segments],
                approx_n_chunks_total=n_workers_effective * avg_workloads_per_worker,
                overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
            )

            # Define processing function for each payload
            def process_chunk(array_idx, chunk_arr):
                segment_name = leg_segments[array_idx][0]
                return self._process_single_leg_sequence(
                    segment_name,
                    chunk_arr,
                    stages,
                    self.kinematic_chain_class,
                    self.initial_angles,
                    hide_progress_bar,
                )

            # Execute in parallel
            results_by_chunk = parallel_executor(
                delayed(process_chunk)(array_idx, chunk_arr)
                for array_idx, start_idx, chunk_arr in input_chunks
            )

            # Get results
            result_chunks_fk = []
            result_chunks_joint_angles = defaultdict(list)
            for chunk_idx in range(len(input_chunks)):
                array_idx, start_idx, _ = input_chunks[chunk_idx]
                _, result_fk_chunk, local_joint_angles = results_by_chunk[chunk_idx]
                result_chunks_fk.append((array_idx, start_idx, result_fk_chunk))
                for key, arr in local_joint_angles.items():
                    result_chunks_joint_angles[key].append((array_idx, start_idx, arr))

            # Merge output chunks and store in dicts
            output_arrays_fk = merge_chunks_into_arrays(
                result_chunks_fk,
                n_arrays=len(leg_segments),
                seq_length=seq_length,
                overlap=chunk_overlap,
            )
            for i, (segment_name, _) in enumerate(leg_segments):
                forward_kinematics_dict[segment_name] = output_arrays_fk[i]
            for key, arr_chunks in result_chunks_joint_angles.items():
                self.joint_angles_dict[key] = merge_chunks_into_arrays(
                    arr_chunks,
                    n_arrays=1,
                    seq_length=seq_length,
                    overlap=chunk_overlap,
                )[0]

        _logger.debug("Joint angles and forward kinematics are computed.")

        if export_path is not None:
            save_file(
                Path(export_path) / "forward_kinematics.pkl",
                forward_kinematics_dict,
            )
            save_file(
                Path(export_path) / "leg_joint_angles.pkl",
                self.joint_angles_dict,
            )
            _logger.info(
                "Joint angles and forward kinematics are saved at %s",
                export_path,
            )

        return self.joint_angles_dict, forward_kinematics_dict


class LegInvKinGeneric(LegInvKinBase):
    """Class to calculate inverse kinematics for leg joints in a
    generic manner to only match the given end effector.

    Parameters
    ----------
    aligned_pos : Dict[str, np.ndarray]
        Aligned pose from the AlignPose class.
        Should have the following structure

        >>> pose_data_dict = {
                "<side><segment>_leg": np.ndarray[N_frames,N_key_points,3],
            }
    kinematic_chain : KinematicChainGeneric
        Kinematic chain of the leg.
    initial_angles : Dict[str, np.ndarray], optional
        Initial angles of DOFs.
        If not provided, the default values from data.py will be used.
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"

    Examples
    --------

    >>> from pathlib import Path
    >>> from seqikpy.kinematic_chain import KinematicChainGeneric
    >>> from seqikpy.leg_inverse_kinematics import LegInvKinGeneric
    >>> from seqikpy.data import BOUNDS, INITIAL_ANGLES
    >>> from seqikpy.utils import load_file

    >>> DATA_PATH = Path("../data/anipose_220525_aJO_Fly001_001/pose-3d")
    >>> f_path = DATA_PATH / "pose3d_aligned.pkl"

    >>> aligned_pos = load_file(f_path)

    >>> seq_ik = LegInvKinGeneric(
            aligned_pos=aligned_pos,
            kinematic_chain_class=KinematicChainGeneric(
                bounds_dof=BOUNDS,
                legs_list=["RF", "LF"],
                body_size=None,
            ),
            initial_angles=INITIAL_ANGLES
        )

    >>> leg_joint_angles, forward_kinematics = gen_ik.run_ik_and_fk(
            export_path=DATA_PATH,
            hide_progress_bar=False
        )
    """

    def __init__(
        self,
        aligned_pos: Dict[str, np.ndarray],
        kinematic_chain_class: KinematicChainGeneric,
        initial_angles: Optional[Dict[str, np.ndarray]] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        super().__init__(aligned_pos, kinematic_chain_class, initial_angles, log_level)
        # Create an empty dict for joint angles
        self.joint_angles_dict = {}

    @staticmethod
    def _process_single_leg_generic(
        segment_name,
        segment_array,
        kinematic_chain_class,
        initial_angles_dict,
        hide_progress_bar,
    ):
        """Process a single leg for generic inverse kinematics."""
        leg_name = segment_name.split("_")[0]

        if not leg_name in kinematic_chain_class.body_size:
            return segment_name, None, {}

        # Create a temporary instance to process this leg
        temp_instance = LegInvKinGeneric.__new__(LegInvKinGeneric)
        temp_instance.kinematic_chain_class = kinematic_chain_class
        temp_instance.initial_angles = initial_angles_dict
        temp_instance.joint_angles_dict = {}

        # Setup minimal logger
        import logging

        temp_instance.logger = logging.getLogger("LegInvKinGeneric")
        temp_instance.logger.setLevel(logging.INFO)

        origin = segment_array[:, 0, :]
        end_effector_pos = segment_array[:, -1, :]
        initial_angles = initial_angles_dict[leg_name]["stage_4"]

        forward_kinematics_result = temp_instance.calculate_ik_stage(
            end_effector_pos=end_effector_pos,
            origin=origin,
            initial_angles=initial_angles,
            segment_name=leg_name,
            hide_progress_bar=hide_progress_bar,
        )

        return segment_name, forward_kinematics_result, temp_instance.joint_angles_dict

    def calculate_ik_stage(
        self,
        end_effector_pos: np.ndarray,
        origin: np.ndarray,
        initial_angles: np.ndarray,
        segment_name: str,
        **kwargs,
    ) -> np.ndarray:
        """For a given trial pose data, calculates the inverse kinematics.

        Parameters
        ----------
        end_effector_pos : np.ndarray
            3D array containing the position of the end effector pos
        origin : np.ndarray
            Origin of the kinematic chain, i.e., Thorax-Coxa joint
        initial_angles : np.ndarray
            Initial angles for the optimization seed
        segment_name : str
            Leg side, i.e., RF, LF, RM, LM, RH, LH


        Returns
        -------
        np.ndarray
            Array containing the cartesian coordinates of the joint positions.
            The joint angles are saved in a class attribute
        """
        hide_progress_bar = kwargs.get("hide_progress_bar", False)

        if not segment_name in ["RF", "LF", "RM", "LM", "RH", "LH"]:
            raise ValueError(f"Segment name ({segment_name}) is not valid.")

        frames_no = end_effector_pos.shape[0]
        end_effector_pos_diff = end_effector_pos - origin

        # Joint angles shape: (number of frames, number of DOFs)
        joint_angles = np.empty((frames_no, len(initial_angles)))
        # Forward kinematics shape: (number of frames, number of
        # joints in the kinematics chain, axes-x,y,z)
        forward_kinematics = np.empty((frames_no, len(initial_angles), 3))

        # if the origin is a vector, convert it to a matrix
        if origin.size == 3:
            origin = np.tile(origin, (frames_no, 1))

        # Generic IK, the kinematic chain is the entire leg
        kinematic_chain = self.kinematic_chain_class.create_leg_chain(
            leg_name=segment_name,
        )

        # Start the inverse kinematics calculation
        for t in trange(
            frames_no,
            disable=hide_progress_bar,
            desc=f"Calculating IK {segment_name}",
        ):
            # For the first frame, use the given initial angles, for the rest
            # use the calculated joint angles from the previous time step
            initial_angles = initial_angles if t == 0 else joint_angles[t - 1, :]
            # Calculate the inverse kinematics
            joint_angles[t, :] = self.calculate_ik(
                kinematic_chain, end_effector_pos_diff[t, :], initial_angles
            )

            forward_kinematics[t, :] = (
                self.calculate_fk(kinematic_chain, joint_angles[t, :]) + origin[t, :]
            )

        # Link names
        link_names = [link.name for link in kinematic_chain.links]
        # Store the joint angles based on the stage number
        for link_name in link_names:
            if "Base" in link_name or "Claw" in link_name:
                continue
            self.joint_angles_dict[f"Angle_{link_name}"] = joint_angles[
                :, link_names.index(link_name)
            ]

        return forward_kinematics

    def run_ik_and_fk(
        self,
        export_path: Union[Path, str] = None,
        n_workers: int = 1,
        hide_progress_bar=False,
        **kwargs,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Runs inverse and forward kinematics for leg joints.

        Parameters
        ----------
        export_path : Union[Path, str], optional
            Path where the results will be saved,
            if None, nothing is saveed, by default None
        n_workers (kwargs) : int, optional
            Number of parallel jobs for leg processing. -1 uses all cores,
            1 disables  parallelization, by default 1.

        Returns
        -------
        Tuple[Dict[str,np.ndarray], Dict[str,np.ndarray]]
            Two dictionaries containing joint angles and forward
            kinematics, respectively.
        """
        forward_kinematics_dict = {}
        _logger.info("Computing joint angles and forward kinematics...")

        # Filter leg segments
        leg_segments = [
            (name, array)
            for name, array in self.aligned_pos.items()
            if "leg" in name.lower()
        ]

        if n_workers == 1 or len(leg_segments) <= 1:
            # Sequential processing
            for segment_name, segment_array in leg_segments:
                result_name, result_fk, local_joint_angles = (
                    self._process_single_leg_generic(
                        segment_name,
                        segment_array,
                        self.kinematic_chain_class,
                        self.initial_angles,
                        hide_progress_bar,
                    )
                )
                if result_fk is not None:
                    forward_kinematics_dict[result_name] = result_fk
                    self.joint_angles_dict.update(local_joint_angles)
        else:
            # Parallel processing of legs
            results = Parallel(n_jobs=n_workers)(
                delayed(self._process_single_leg_generic)(
                    segment_name,
                    segment_array,
                    self.kinematic_chain_class,
                    self.initial_angles,
                    hide_progress_bar,
                )
                for segment_name, segment_array in leg_segments
            )

            # Collect results
            for segment_name, result_fk, local_joint_angles in results:
                if result_fk is not None:
                    forward_kinematics_dict[segment_name] = result_fk
                    self.joint_angles_dict.update(local_joint_angles)

        _logger.debug("Joint angles and forward kinematics are computed.")

        if export_path is not None:
            save_file(
                Path(export_path) / "forward_kinematics.pkl",
                forward_kinematics_dict,
            )
            save_file(
                Path(export_path) / "leg_joint_angles.pkl",
                self.joint_angles_dict,
            )
            _logger.info(
                "Joint angles and forward kinematics are saved at %s",
                export_path,
            )

        return self.joint_angles_dict, forward_kinematics_dict
