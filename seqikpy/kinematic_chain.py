""" Module that contains a set of kinematic chains."""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, List
import warnings
from nptyping import NDArray

import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

from seqikpy.data import NMF_TEMPLATE
from seqikpy.utils import calculate_body_size

# Ignore the warnings
warnings.filterwarnings("ignore")

# Axes as a named tuple to ensure immutability
AxesTuple = namedtuple("AxesTuple", "X_AXIS Y_AXIS Z_AXIS")
Axes = AxesTuple(
    X_AXIS=[1, 0, 0],
    Y_AXIS=[0, 1, 0],
    Z_AXIS=[0, 0, 1]
)


class KinematicChainBase(ABC):
    """Abstract class to create kinematic chains for the legs.

    Parameters
    ----------
    bounds_dof : Dict[str, NDArray]
        Dictionary that contains the bounds of joint degrees-of-freedom.
    legs_list : List[str]
        List of legs for which the kinematic chains are created.
    body_size : Dict[str, float], optional
        Dictionary that contains the size of different body parts,
        by default None
    """

    def __init__(
        self,
        bounds_dof: Dict[str, NDArray],
        legs_list: List[str],
        body_size: Dict[str, float] = None,
    ) -> None:
        # NMF size is calculated internally if size is not provided
        self.body_size = calculate_body_size(
            NMF_TEMPLATE,
            legs_list
        ) if body_size is None else body_size
        self.bounds_dof = bounds_dof

    def __call__(self):
        print("Base kinematic chain is called.")

    @abstractmethod
    def create_leg_chain(
        self,
        leg_name: str,
        **kwargs
    ) -> Chain:
        """Returns the respective leg chain based on the stage.

        Parameters
        ----------
        leg_name : str
            Name of the leg, RF or LF

        Returns
        -------
        KinematicChain of a given leg
        """
        raise NotImplementedError


class KinematicChainSeq(KinematicChainBase):
    """Sequential kinematic chain class to create a kinematic chain
    stage-by-stage.

    Parameters
    ----------
    bounds_dof : Dict[str, NDArray]
        Dictionary that contains the bounds of joint degrees-of-freedom.
    legs_list : List[str]
        List of legs for which the kinematic chains are created.
    body_size : Dict[str, float], optional
        Dictionary that contains the size of different body parts,
        by default None

    NOTE: the implementation follows Yaw-Pitch-Roll order.
    """

    def __call__(self):
        print("Sequential kinematic chain is called.")

    def create_leg_chain(
        self,
        leg_name: str,
        **kwargs
    ) -> Chain:
        """Returns the respective leg chain based on the stage.

        Parameters
        ----------
        leg_name : str
            Name of the leg, i.e., RF, LF, etc.

        Kwargs
        ------
        angles : Dict[str, NDArray]
            Joint angles calculated in the previous step,
            None at the first step, by default None
        stage : int
            Stage number, (1,2,3,4)
        t : int
            Time step, by default 0

        Returns
        -------
        KinematicChain at the respective stage and time step.

        Raises
        ------
        ValueError : If the stage number is not in (1,2,3,4)
        ValueError : If the leg name is not in
        ["RF", "LF", "RM", "LM", "RH", "LH"]

        """
        angles = kwargs.get("angles", None)
        stage = kwargs.get("stage", 1)
        t = kwargs.get("t", 0)

        if not leg_name in ["RF", "LF", "RM", "LM", "RH", "LH"]:
            raise ValueError(f"Unknown leg name ({leg_name}) is provided!")

        if not 1 <= stage <= 4:
            raise ValueError(f"Unknown stage number ({stage}) number is provided!")

        if stage == 1:
            kinematic_chain = self.create_leg_chain_stage_1(leg_name)
        elif stage == 2:
            kinematic_chain = self.create_leg_chain_stage_2(leg_name, angles=angles, t=t)
        elif stage == 3:
            kinematic_chain = self.create_leg_chain_stage_3(leg_name, angles=angles, t=t)
        elif stage == 4:
            kinematic_chain = self.create_leg_chain_stage_4(leg_name, angles=angles, t=t)

        return kinematic_chain

    def create_leg_chain_stage_1(
        self,
        leg_name: str,
    ) -> Chain:
        """Leg chain to calculate thorax/coxa pitch and yaw.

        Parameters
        ----------
        leg_name : str
            Leg name, e.g., RF, LF, RM, LM, RH, LH

        Returns
        -------
        Chain
            Chain of the leg at stage 1, which calculates
            thorax/coxa pitch and yaw joint angles and only
            contains Coxa leg segment.
        """
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_yaw",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.X_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_yaw"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Coxa"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_CTr_pitch"],
            ),
        ]

        return Chain(name="chain_stage_1", links=kinematic_chain)

    def create_leg_chain_stage_2(self, leg_name: str, angles: Dict[str, NDArray], t: int) -> Chain:
        """Leg chain to calculate thorax/coxa roll and coxa/femur pitch.

        Parameters
        ----------
        leg_name : str
            Leg name, e.g., RF, LF, RM, LM, RH, LH

        Returns
        -------
        Chain
            Chain of the leg at stage 2, which calculates
            thorax/coxa roll and coxa/femur pitch joint angles
            and contains Coxa and Femur leg segments.
        """
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_yaw",
                origin_translation=[0, 0, 0],
                origin_orientation=[angles[f"Angle_{leg_name}_ThC_yaw"][t], 0, 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_yaw"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, angles[f"Angle_{leg_name}_ThC_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Z_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Coxa"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_CTr_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_FTi_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Femur"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_FTi_pitch"],
            ),
        ]

        return Chain(name="chain_stage_2", links=kinematic_chain)

    def create_leg_chain_stage_3(self, leg_name: str, angles: Dict[str, NDArray], t: int) -> Chain:
        """Leg chain to calculate coxa/femur roll and femur/tibia pitch.

        Parameters
        ----------
        leg_name : str
            Leg name, e.g., RF, LF, RM, LM, RH, LH

        Returns
        -------
        Chain
            Chain of the leg at stage 3, which calculates
            coxa/femur roll and femur/tibia pitch joint angles
            and contains Coxa, Femur and Tibia leg segments.
        """
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_yaw",
                origin_translation=[0, 0, 0],
                origin_orientation=[angles[f"Angle_{leg_name}_ThC_yaw"][t], 0, 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_yaw"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, angles[f"Angle_{leg_name}_ThC_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, angles[f"Angle_{leg_name}_ThC_roll"][t]],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Coxa"]],
                origin_orientation=[0, angles[f"Angle_{leg_name}_CTr_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_CTr_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Z_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_CTr_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_FTi_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Femur"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_FTi_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_TiTa_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Tibia"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_TiTa_pitch"],
            ),
        ]

        return Chain(name="chain_stage_3", links=kinematic_chain)

    def create_leg_chain_stage_4(self, leg_name: str, angles: Dict[str, NDArray], t: int) -> Chain:
        """Leg chain to calculate tibia/tarsus pitch.

        Parameters
        ----------
        leg_name : str
            Leg name, e.g., RF, LF, RM, LM, RH, LH

        Returns
        -------
        Chain
            Chain of the leg at stage 4, which calculates
            coxa/femur roll and tibia/tarsus pitch joint angles and
            contains the entire leg, i.e., Coxa, Femur, Tibia and Tarsus.
        """
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_yaw",
                origin_translation=[0, 0, 0],
                origin_orientation=[angles[f"Angle_{leg_name}_ThC_yaw"][t], 0, 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_yaw"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, angles[f"Angle_{leg_name}_ThC_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, angles[f"Angle_{leg_name}_ThC_roll"][t]],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Coxa"]],
                origin_orientation=[0, angles[f"Angle_{leg_name}_CTr_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_CTr_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, angles[f"Angle_{leg_name}_CTr_roll"][t]],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_CTr_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_FTi_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Femur"]],
                origin_orientation=[0, angles[f"Angle_{leg_name}_FTi_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_FTi_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_TiTa_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Tibia"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_TiTa_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_Claw",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Tarsus"]],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
                joint_type="revolute",
                bounds=[-np.pi, np.pi],
            ),
        ]

        return Chain(name="chain_stage_4", links=kinematic_chain)


class KinematicChainGeneric(KinematicChainBase):
    """Generic kinematic chain class to create one kinematic chain for
    the entire leg.

    Parameters
    ----------
    bounds_dof : Dict[str, NDArray]
        Dictionary that contains the bounds of joint degrees-of-freedom.
    legs_list : List[str]
        List of legs for which the kinematic chains are created.
    body_size : Dict[str, float], optional
        Dictionary that contains the size of different body parts,
        by default None
    """

    def __call__(self):
        print("Generic kinematic chain is called.")

    def create_leg_chain(
        self,
        leg_name: str,
        **kwargs
    ) -> Chain:
        """Returns the respective leg chain based on the stage.

        Parameters
        ----------
        leg_name : str
            Name of the leg, RF or LF
        angles : Dict[str, NDArray], optional
            Joint angles calculated in the previous step,
            None at the first step, by default None

        Returns
        -------
        The entire KinematicChain of a given leg
        """
        if not leg_name in ["RF", "LF", "RM", "LM", "RH", "LH"]:
            raise ValueError(f"Unknown leg name ({leg_name}) is provided!")

        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Z_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_yaw",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.X_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_yaw"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Coxa"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_CTr_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Z_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_CTr_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_FTi_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Femur"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_FTi_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_TiTa_pitch",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Tibia"]],
                origin_orientation=[0, 0, 0],
                rotation=Axes.Y_AXIS,
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_TiTa_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_Claw",
                origin_translation=[0, 0, -self.body_size[f"{leg_name}_Tarsus"]],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
                joint_type="revolute",
                bounds=[-np.pi, np.pi],
            ),
        ]

        return Chain(name="chain", links=kinematic_chain)
