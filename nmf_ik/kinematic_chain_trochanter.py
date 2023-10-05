""" Module that contains a set of kinematic chains."""
import numpy as np
from typing import Dict
from nptyping import NDArray

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


class KinematicChain:
    """Create kinematic chains at different stages for the legs with new femur position.

    Parameters
    ----------
    nmf_offset : Dict[str, float]
        Dictionary that contains the offset of a body part with respect to
        the previous one.
    bounds_dof : Dict[str, NDArray]
        Dictionary that contains the bounds of joint degrees-of-freedom.
    """

    def __init__(
        self, nmf_offset: Dict[str, float],
        bounds_dof: Dict[str, NDArray],
    ) -> None:
        # NMF size is calculated internally
        self.nmf_offset = nmf_offset
        self.bounds_dof = bounds_dof

    def __call__(self):
        print('kinematic_chain_trochanter is called.')

    def create_leg_chain(
        self,
        stage: int,
        leg_name: str,
        angles: Dict[str, NDArray] = None,
        t: int = None,
        femur_orientation: float = 0  #  radian
    ) -> Chain:
        """Returns the respective leg chain based on the stage.

        Parameters
        ----------
        stage : int
            Stage number, (1,2,3,4)
        leg_name : str
            Name of the leg, RF or LF
        angles : Dict[str, NDArray], optional
            Joint angles calculated in the previous step, None at the first step, by default None
        t : int, optional
            Time step, by default None
        femur_orientation : float, optional
            Orientation of femur with respect to coxa introduced
            by trochanter in radians around the yaw axis, by default 0

        Returns
        -------
        Chain
        """

        if stage == 1:
            kinematic_chain = self.create_leg_chain_stage_1(leg_name)
        elif stage == 2:
            kinematic_chain = self.create_leg_chain_stage_2(
                leg_name, angles=angles, t=t, femur_orientation=femur_orientation)
        elif stage == 3:
            kinematic_chain = self.create_leg_chain_stage_3(
                leg_name, angles=angles, t=t, femur_orientation=femur_orientation)
        elif stage == 4:
            kinematic_chain = self.create_leg_chain_stage_4(
                leg_name, angles=angles, t=t, femur_orientation=femur_orientation)
        else:
            ValueError(f"Unknown stage number ({stage}) number is provided!")
        return kinematic_chain

    # def create_head_chain(self, origin_translation: NDArray) -> Chain:
    #     """Returns the head kinematic chain.

    #     Returns
    #     -------
    #     Chain
    #     """
    #     origin_translation = (
    #         self.nmf_template['R_Antenna_base'] + self.nmf_template['R_Antenna_base']) * 0.5 \
    #             - self.nmf_template['Neck']

    #     links = [
    #         OriginLink(),
    #         URDFLink(
    #             name="Head_roll",
    #             origin_translation=[0, 0, 0],
    #             origin_orientation=[0, 0, 0],
    #             rotation=[0, 0, 1],
    #             joint_type="revolute",
    #             bounds=self.bounds_dof["Head_roll"],
    #         ),

    #         URDFLink(
    #             name="Head_pitch",
    #             origin_translation=[0, 0, 0],
    #             origin_orientation=[0, 0, 0],
    #             rotation=[0, 1, 0],
    #             joint_type="revolute",
    #             bounds=self.bounds_dof["Head_pitch"],
    #         ),
    #         # We need to put origin orientation into account, that's why.
    #         URDFLink(
    #             name="Antenna_base",
    #             origin_translation=origin_translation,
    #             origin_orientation=[0, 0, 0],
    #             rotation=[0, 1, 0],
    #             bounds=(-np.pi, np.pi),
    #         ),
    #     ]
    #     return Chain(name="head", links=links)

    def create_leg_chain_stage_1(
        self,
        leg_name: str,
    ) -> Chain:
        """ Leg chain to calculate thorax/coxa pitch and yaw."""
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_yaw",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_yaw"],
            ),
            URDFLink(
                name=f"{leg_name}_ThC_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_yaw",
                origin_translation=self.nmf_offset[f'{leg_name}_Coxa'],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],
                joint_type="revolute",
                bounds=[-np.pi, np.pi],
            ),
        ]

        return Chain(name="chain_stage_1", links=kinematic_chain)

    def create_leg_chain_stage_2(
        self, leg_name: str, angles: Dict[str, NDArray], t: int, femur_orientation: float = 0
    ) -> Chain:
        """ Leg chain to calculate thorax/coxa roll and coxa/femur pitch."""
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_ThC_roll"],
            ),
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
                name=f"{leg_name}_CTr_yaw",
                origin_translation=self.nmf_offset[f'{leg_name}_Coxa'],
                origin_orientation=[femur_orientation, 0, 0],
                rotation=None,
                joint_type="fixed",
                bounds=[-np.pi, np.pi],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_CTr_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_FTi_pitch",
                origin_translation=self.nmf_offset[f"{leg_name}_Femur"],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_FTi_pitch"],
            ),
        ]

        return Chain(name="chain_stage_2", links=kinematic_chain)

    def create_leg_chain_stage_3(
        self, leg_name: str, angles: Dict[str, NDArray], t: int, femur_orientation: float = 0
    ) -> Chain:
        """ Leg chain to calculate coxa/femur roll and femur/tibia pitch."""
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, angles[f"Angle_{leg_name}_ThC_roll"][t]],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_roll"],
            ),
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
                name=f"{leg_name}_CTr_yaw",
                origin_translation=self.nmf_offset[f'{leg_name}_Coxa'],
                origin_orientation=[femur_orientation, 0, 0],
                rotation=None,
                joint_type="fixed",
                bounds=[-np.pi, np.pi],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, angles[f"Angle_{leg_name}_CTr_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_CTr_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_CTr_roll"],
            ),
            URDFLink(
                name=f"{leg_name}_FTi_pitch",
                origin_translation=self.nmf_offset[f"{leg_name}_Femur"],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_FTi_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_TiTa_pitch",
                origin_translation=self.nmf_offset[f"{leg_name}_Tibia"],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_TiTa_pitch"],
            ),
        ]

        return Chain(name="chain_stage_3", links=kinematic_chain)

    def create_leg_chain_stage_4(
        self, leg_name: str, angles: Dict[str, NDArray], t: int, femur_orientation: float = 0
    ) -> Chain:
        """ Leg chain to calculate tibia/tarsus pitch."""
        kinematic_chain = [
            OriginLink(),
            URDFLink(
                name=f"{leg_name}_ThC_roll",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, angles[f"Angle_{leg_name}_ThC_roll"][t]],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_ThC_roll"],
            ),
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
                name=f"{leg_name}_CTr_yaw",
                origin_translation=self.nmf_offset[f'{leg_name}_Coxa'],
                origin_orientation=[femur_orientation, 0, 0],
                rotation=None,
                joint_type="fixed",
                bounds=[-np.pi, np.pi],
            ),
            URDFLink(
                name=f"{leg_name}_CTr_pitch",
                origin_translation=[0, 0, 0],
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
                origin_translation=self.nmf_offset[f"{leg_name}_Femur"],
                origin_orientation=[0, angles[f"Angle_{leg_name}_FTi_pitch"][t], 0],
                rotation=None,
                joint_type="fixed",
                bounds=self.bounds_dof[f"{leg_name}_FTi_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_TiTa_pitch",
                origin_translation=self.nmf_offset[f"{leg_name}_Tibia"],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                joint_type="revolute",
                bounds=self.bounds_dof[f"{leg_name}_TiTa_pitch"],
            ),
            URDFLink(
                name=f"{leg_name}_Claw",
                origin_translation=self.nmf_offset[f"{leg_name}_Tarsus"],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
                joint_type="revolute",
                bounds=[-np.pi, np.pi],
            ),
        ]

        return Chain(name="chain_stage_4", links=kinematic_chain)