# Inverse kinematics

## Limitations of the conventional methods

The main limitation of the conventional joint DOF angle calculation methods (e.g., dot product approach) is that the joint center of rotations is assumed to be the intersection of two vectors, and the derived joint angles might result in forward kinematics far from the original pose. On the other hand, the optimization-based inverse kinematics approaches, given the kinematic chain of the links (i.e., rotational axes and pivot points of joints), try to match the end effector of the kinematic chain to the real end effector as closely as possible. However, this method disregards the position of the previous joints, leading to unrealistic forward kinematics.

## Sequential inverse kinematics

To reconcile the shortcomings of the two methods above, we developed a sequential inverse kinematics calculation algorithm, also known as *body movement optimization*. This approach is inspired by the control of humanoid robots based on human motion capture data {cite:p}`begon_multibody_2018`.

Specifically, our method begins with the proximal-most leg segment to calculate certain DOF angles of the subsequent joint. It then uses the angles calculated in the previous step and extends the kinematic chain by adding another segment, repeating this process until the tip of the chain is reached. This approach comprises four steps as follows:

- **Stage 1:** kinematic chain consists of the coxa leg segment and is used to calculate Thorax-Coxa yaw (i.e., rotation around the anteroposterior axis) and pitch (i.e., rotation around the mediolateral axis) by following the tip of coxa as an end-effector
- **Stage 2:** kinematic chain consists of the coxa and the trochanter + femur (fused) leg segments and is used to calculate the Thorax-Coxa roll (i.e., rotation around the dorsoventral axis), Coxa-Trochanter pitch (i.e., rotation around the mediolateral axis) by following the tip of the femur as an end-effector
- **Stage 3:** kinematic chain consists of the coxa, the trochanter \& femur (fused), and tibia leg segments and is used to calculate the Trochanter-Femur roll (i.e., rotation around the dorsoventral axis), Femur-Tibia pitch (i.e., rotation around the mediolateral axis) by following the tip of the tibia as an end-effector
- **Stage 4:** The kinematic chain consists of the entire leg segments and is used to calculate the Tibia-Tarsus pitch (i.e., rotation around the mediolateral axis) by following the tip of the leg (i.e., Claw) as an end-effector

Our pipeline builds upon the open-source inverse kinematics library [IKPy](https://github.com/Phylliade/ikpy), which uses an optimizer from SciPy to minimize a scalar function, that is, the Euler distance between the original pose and the forward kinematics from the calculated joint angles.


```{note}
When replaying the joint angles in simulation, it is essential to keep the kinematic chain of the biomechanical model consistent with the kinematic chain used to calculate the joint angles.

For instance, if the thorax-coxa joint angles are calculated in the order of *yaw-pitch-roll*, then the biomechanical model should also have the thorax-coxa joint DOFs in the same order since we use the intrinsic Euler convention to calculate the joint angles.
```

## Head and antennal joint angle calculation

Since the head consists of two moving body parts (left and right antennae) on the main neck joint, the kinematic chain approach is infeasible as it tends to follow one antenna "more closely" than the other, thus introducing errors.

To calculate the neck and antennal joint angles, we resorted to the conventional angle formula using the dot product of two vectors.

The vectors constructed for the head joint angles are as follows:

- **Head roll:** the angle between the vector from the right antenna base to the right antenna base and the global mediolateral axis.
- **Head pitch:** the angle between the vector from the neck to the mid-antennae base and the global anteroposterior axis.
- **Head yaw:** the angle between the vector from the right antenna base to the right antenna base and the global anteroposterior axis.
- **Antennal pitch:** the angle between the vector from the neck to the antenna base and the vector from the antenna base to the tip.
- **Antennal yaw:** the angle between the vector from the right antenna base to the left antenna base and the vector from the antenna base to the antenna tip.

Note that, when the head rotation reaches 90 degrees, the antennal pitch and yaw calculations will swap their roles, causing inaccurate computations. To circumvent this issue, we first calculate the head joint angles and then de-rotate the head key points by the amount of head rotation to calculate the antennal joint angles.