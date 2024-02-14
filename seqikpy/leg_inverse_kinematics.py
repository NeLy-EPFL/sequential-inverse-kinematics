""" Module to calculate inverse kinematics from 3D pose based on IKPy."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Union, Literal, Optional
import logging
import warnings
from nptyping import NDArray

import numpy as np
from tqdm import trange
from ikpy.chain import Chain

from seqikpy.utils import save_file
from seqikpy.data import INITIAL_ANGLES
from seqikpy.kinematic_chain import KinematicChainBase, KinematicChainSeq, KinematicChainGeneric
# Ignore the warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    format=" %(asctime)s - %(levelname)s- %(message)s",
    handlers=[logging.StreamHandler()]
)


class LegInvKinBase(ABC):
    """Abstract class to calculate inverse kinematics for leg joints.

    Parameters
    ----------
    aligned_pos : Dict[str, NDArray]
        Aligned pose from the AlignPose class.
        Should have the following structure:
            "<side><segment>_leg" : np.array([frames, 5, 3])}
    kinematic_chain : KinematicChainBase
        Kinematic chain of the leg.
    initial_angles : Dict[str, NDArray], optional
        Initial angles of DOFs.
        If not provided, the default values from data.py will be used.
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"
    """

    def __init__(
        self,
        aligned_pos: Dict[str, NDArray],
        kinematic_chain_class: KinematicChainBase,
        initial_angles: Optional[Dict[str, NDArray]] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        self.aligned_pos = aligned_pos
        self.kinematic_chain_class = kinematic_chain_class
        self.initial_angles = INITIAL_ANGLES if initial_angles is None else initial_angles

        # Get the logger of the module
        self.logger = logging.getLogger(self.__class__.__name__)
        numeric_level = getattr(logging, log_level.upper(), None)
        self.logger.setLevel(numeric_level)

    def calculate_ik(self, kinematic_chain: Chain, target_pos: NDArray,
                     initial_angles: NDArray = None) -> NDArray:
        """Calculates the joint angles in the leg chain."""
        # don"t take the last and first ones into account
        return kinematic_chain.inverse_kinematics(
            target_position=target_pos,
            initial_position=initial_angles
        )

    def calculate_fk(self, kinematic_chain: Chain, joint_angles: NDArray) -> NDArray:
        """Calculates the forward kinematics from the joint dof angles."""
        fk = kinematic_chain.forward_kinematics(joint_angles, full_kinematics=True)
        end_effector_positions = np.zeros((len(kinematic_chain.links), 3))
        for link in range(len(kinematic_chain.links)):
            end_effector_positions[link, :] = fk[link][:3, 3]
        return end_effector_positions

    def get_scale_factor(self, vector: NDArray, length: float) -> float:
        """Gets scale the ratio between two vectors."""
        vector_diff = np.linalg.norm(np.diff(vector, axis=0), axis=1)
        norm_sum = np.sum(vector_diff)
        return length / norm_sum

    @abstractmethod
    def calculate_ik_stage(
        self,
        end_effector_pos: NDArray,
        origin: NDArray,
        initial_angles: NDArray,
        segment_name: str,
        **kwargs
    ) -> NDArray:
        """For a given trial pose data, calculates the inverse kinematics.

        Parameters
        ----------
        end_effector_pos : NDArray
            3D array containing the position of the end effector pos
        origin : NDArray
            Origin of the kinematic chain, i.e., Thorax-Coxa joint
        initial_angles : NDArray
            Initial angles for the optimization seed
        segment_name : str
            Leg side, i.e., RF, LF, RM, LM, RH, LH

        Returns
        -------
        NDArray
            Array containing the cartesian coordinates of the joint positions.
            The joint angles are saved in a class attribute
        """

    @abstractmethod
    def run_ik_and_fk(
        self,
        export_path: Union[Path, str] = None,
        **kwargs
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """Runs inverse and forward kinematics for leg joints.

        Parameters
        ----------
        export_path : Union[Path, str], optional
            Path where the results will be saved,
            if None, nothing is saveed, by default None

        Returns
        -------
        Tuple[Dict[str,NDArray], Dict[str,NDArray]]
            Two dictionaries containing joint angles and forward
            kinematics, respectively.
        """


class LegInvKinSeq(LegInvKinBase):
    """Class to calculate inverse kinematics for leg joints in a
    sequential manner. This method finds the optimal joint angles
    within the bounds to match each koint as closely as possible.

    Parameters
    ----------
    aligned_pos : Dict[str, NDArray]
        Aligned pose from the AlignPose class.
        Should have the following structure:
            "<side><segment>_leg" : np.array([frames, 5, 3])}
    kinematic_chain : KinematicChainSeq
        Kinematic chain of the leg.
    initial_angles : Dict[str, NDArray], optional
        Initial angles of DOFs.
        If not provided, the default values from data.py will be used.
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"

    Example usage:
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
                nmf_size=None,
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
        aligned_pos: Dict[str, NDArray],
        kinematic_chain_class: KinematicChainSeq,
        initial_angles: Optional[Dict[str, NDArray]] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        super().__init__(aligned_pos, kinematic_chain_class, initial_angles, log_level)
        # Create an empty dict for joint angles
        self.joint_angles_dict = {}

    def calculate_ik_stage(
        self,
        end_effector_pos: NDArray,
        origin: NDArray,
        initial_angles: NDArray,
        segment_name: str,
        **kwargs
    ) -> NDArray:
        """For a given trial pose data, calculates the inverse kinematics.

        Parameters
        ----------
        end_effector_pos : NDArray
            3D array containing the position of the end effector pos
        origin : NDArray
            Origin of the kinematic chain, i.e., Thorax-Coxa joint
        initial_angles : NDArray
            Initial angles for the optimization seed

        Kwargs
        ------
        segment_name : str
            Leg side, i.e., RF, LF, RM, LM, RH, LH
        stage: int
            Stage in the sequential calculation, should be between 1 and 4

        Returns
        -------
        NDArray
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
        for t in trange(frames_no, disable=hide_progress_bar, desc=f"{segment_name} stage {stage}"):
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
                    self.calculate_fk(kinematic_chain, joint_angles[t, :]) + origin[t, :]
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
            self.logger.debug("Stage 1 is completed!")
        # Stage 2: Thorax-Coxa roll, Coxa-Trochanter pitch
        elif stage == 2:
            self.joint_angles_dict[f"Angle_{segment_name}_ThC_roll"] = joint_angles[
                :, link_names.index(f"{segment_name}_ThC_roll")
            ]
            self.joint_angles_dict[f"Angle_{segment_name}_CTr_pitch"] = joint_angles[
                :, link_names.index(f"{segment_name}_CTr_pitch")
            ]
            self.logger.debug("Stage 2 is completed!")
        # Stage 3: Coxa-Trochanter roll, Femur-Tibia pitch
        elif stage == 3:
            self.joint_angles_dict[f"Angle_{segment_name}_CTr_roll"] = joint_angles[
                :, link_names.index(f"{segment_name}_CTr_roll")
            ]
            self.joint_angles_dict[f"Angle_{segment_name}_FTi_pitch"] = joint_angles[
                :, link_names.index(f"{segment_name}_FTi_pitch")
            ]
            self.logger.debug("Stage 3 is completed!")
        # Stage 4: Tibia-Tarsus pitch
        elif stage == 4:
            self.joint_angles_dict[f"Angle_{segment_name}_TiTa_pitch"] = joint_angles[
                :, link_names.index(f"{segment_name}_TiTa_pitch")
            ]
            self.logger.debug("Stage 4 is completed!")

        return forward_kinematics

    def run_ik_and_fk(
        self,
        export_path: Union[Path, str] = None,
        **kwargs
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """Runs inverse and forward kinematics for leg joints.

        Parameters
        ----------
        export_path : Union[Path, str], optional
            Path where the results will be saved,
            if None, nothing is saveed, by default None

        Kwargs
        ------
        stages : List[int], optional
            Stages to run the inverse kinematics.
        hide_progress_bar : Optional[bool], optional
            Hide the progress bar, by default True

        Returns
        -------
        Tuple[Dict[str,NDArray], Dict[str,NDArray]]
            Two dictionaries containing joint angles and forward
            kinematics, respectively.
        """
        stages = kwargs.get("stages", [1, 2, 3, 4])
        hide_progress_bar = kwargs.get("hide_progress_bar", False)

        if max(stages) > 4 or not all(np.diff(stages) == 1):
            raise ValueError(
                "Maximum stage number is 4 and the list should be strictly incremental."
            )
        forward_kinematics_dict = {}

        self.logger.info("Computing joint angles and forward kinematics...")
        for segment_name, segment_array in self.aligned_pos.items():
            if "leg" in segment_name.lower():
                # If segment name is RF_leg so the leg name is RF
                leg_name = segment_name.split("_")[0]

                # If leg_name is not in nmf_size, then continue
                if not leg_name in self.kinematic_chain_class.nmf_size:
                    self.logger.warning(
                        "Leg %s is not in the kinematic chain, continuing...", leg_name
                    )
                    continue

                # First key point of the segment array is the origin
                # of the kinematic chain, i.e., Thorax-Coxa joint
                origin = segment_array[:, 0, :]

                for stage in stages:
                    # for each stage, the end effector is the corresponding joint
                    end_effector_pos = segment_array[:, stage, :]
                    initial_angles = self.initial_angles[leg_name][f"stage_{stage}"]

                    forward_kinematics_dict[segment_name] = self.calculate_ik_stage(
                        end_effector_pos=end_effector_pos,
                        origin=origin,
                        initial_angles=initial_angles,
                        stage=stage,
                        segment_name=leg_name,
                        hide_progress_bar=hide_progress_bar
                    )
            else:
                self.logger.debug("Segment %s is not a leg, continuing...", segment_name)
                continue

        self.logger.debug("Joint angles and forward kinematics are computed.")

        if export_path is not None:
            save_file(
                Path(export_path) / "forward_kinematics.pkl",
                forward_kinematics_dict
            )
            save_file(
                Path(export_path) / "leg_joint_angles.pkl",
                self.joint_angles_dict
            )
            self.logger.info("Joint angles and forward kinematics are saved at %s", export_path)

        return self.joint_angles_dict, forward_kinematics_dict


class LegInvKinGeneric(LegInvKinBase):
    """Class to calculate inverse kinematics for leg joints in a
    generic manner to only match the given end effector.

    Parameters
    ----------
    aligned_pos : Dict[str, NDArray]
        Aligned pose from the AlignPose class.
        Should have the following structure:
            "<side><segment>_leg" : np.array([frames, 5, 3])}
    kinematic_chain : KinematicChainGeneric
        Kinematic chain of the leg.
    initial_angles : Dict[str, NDArray], optional
        Initial angles of DOFs.
        If not provided, the default values from data.py will be used.
    log_level : Literal["DEBUG", "INFO", "WARNING", "ERROR"], optional
        Logging level as a string, by default "INFO"

    Example usage:
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
                nmf_size=None,
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
        aligned_pos: Dict[str, NDArray],
        kinematic_chain_class: KinematicChainGeneric,
        initial_angles: Optional[Dict[str, NDArray]] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    ) -> None:
        super().__init__(aligned_pos, kinematic_chain_class, initial_angles, log_level)
        # Create an empty dict for joint angles
        self.joint_angles_dict = {}

    def calculate_ik_stage(
        self,
        end_effector_pos: NDArray,
        origin: NDArray,
        initial_angles: NDArray,
        segment_name: str,
        **kwargs
    ) -> NDArray:
        """For a given trial pose data, calculates the inverse kinematics.

        Parameters
        ----------
        end_effector_pos : NDArray
            3D array containing the position of the end effector pos
        origin : NDArray
            Origin of the kinematic chain, i.e., Thorax-Coxa joint
        initial_angles : NDArray
            Initial angles for the optimization seed
        segment_name : str
            Leg side, i.e., RF, LF, RM, LM, RH, LH


        Returns
        -------
        NDArray
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
        for t in trange(frames_no, disable=hide_progress_bar, desc=f"Calculating IK {segment_name}"):
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
        **kwargs
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """Runs inverse and forward kinematics for leg joints.

        Parameters
        ----------
        export_path : Union[Path, str], optional
            Path where the results will be saved,
            if None, nothing is saveed, by default None

        Returns
        -------
        Tuple[Dict[str,NDArray], Dict[str,NDArray]]
            Two dictionaries containing joint angles and forward
            kinematics, respectively.
        """
        hide_progress_bar = kwargs.get("hide_progress_bar", False)

        forward_kinematics_dict = {}

        self.logger.info("Computing joint angles and forward kinematics...")
        for segment_name, segment_array in self.aligned_pos.items():
            if "leg" in segment_name.lower():
                # If segment name is RF_leg so the leg name is RF
                leg_name = segment_name.split("_")[0]

                # If leg_name is not in nmf_size, then continue
                if not leg_name in self.kinematic_chain_class.nmf_size:
                    self.logger.warning(
                        "Leg %s is not in the kinematic chain, continuing...", leg_name
                    )
                    continue

                # First key point of the segment array is the origin
                # of the kinematic chain, i.e., Thorax-Coxa joint
                origin = segment_array[:, 0, :]

                # for the generic IK, endeffector is the claw
                end_effector_pos = segment_array[:, -1, :]
                initial_angles = self.initial_angles[leg_name]["stage_4"]

                forward_kinematics_dict[segment_name] = self.calculate_ik_stage(
                    end_effector_pos=end_effector_pos,
                    origin=origin,
                    initial_angles=initial_angles,
                    segment_name=leg_name,
                    hide_progress_bar=hide_progress_bar
                )
            else:
                self.logger.debug("Segment %s is not a leg, continuing...", segment_name)
                continue

        self.logger.debug("Joint angles and forward kinematics are computed.")

        if export_path is not None:
            save_file(
                Path(export_path) / "forward_kinematics.pkl",
                forward_kinematics_dict
            )
            save_file(
                Path(export_path) / "leg_joint_angles.pkl",
                self.joint_angles_dict
            )
            self.logger.info("Joint angles and forward kinematics are saved at %s", export_path)

        return self.joint_angles_dict, forward_kinematics_dict
