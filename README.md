# Sequential Inverse Kinematics

IKPy implementation of Inverse Kinematics using the NMF Biomechanical Model

## Overview

`sequential-inverse-kinematics` is a Python package that provides an implementation of Inverse Kinematics (IK) that is based on the open-source Python package (IKPy)[https://github.com/Phylliade/ikpy]. It aligns the 3D pose of the fly body parts anf computes the joint angles necessary to closely match the desired 3D pose.

## Features

* Align 3D poses to the NMF template model.
* Calculate joint angles and positions to match a desired pose.
* Visualize the results in 2D and 3D.


## Installation

Download the repository on your local machine by running the following line in the terminal:
```bash
$ git clone https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
```
After the download is complete, navigate to the folder:
```bash
$ cd sequential-inverse-kinematics
```
In this folder, run the following commands to create a virtual environment and activate it:
```bash
$ conda create -n nmf-ik python=3.8.0
$ conda activate nmf-ik
```
Finally, install all the dependencies by running:
```bash
$ pip install -e .
```

Note that the IKPy module is added as a submodule. To initialize the submodule, run:
```bash
$ git submodule add https://github.com/gizemozd/ikpy.git ikpy_submodule
$ git submodule update --init
```

## Methodology

### Head joint angle calculations
Since the head consists of two moving body parts (left and right antennae) on the main neck joint, the kinematic chain approach is not accurate as it tends to follow one antenna "more closely" than the other, thus introducing errors. Therefore, head joint angles are calculated via the conventional dot product method.


### Leg joint angle calculations

The main limitation of the conventional IK approaches is that the previous joint are sacrificed at the expense of the end effector. To overcome this challenge, we can build the kinematic chain gradually, starting from beginning.

Specifically, we will start building
1. Thorax-Coxa pitch and yaw: to follow Coxa-Femur
2. Thorax-Coxa roll and Coxa-Femur pitch: to follow Femur-Tibia
3. Coxa-Femur roll and Femur-Tibia pitch: to follow Tibia-Tarsus
4. Tibia-Tarsus pitch: to follow Claw

Note that, although the leg joint angles are calculated based on the IKPy library, the head kinematics (head roll, pitch, yaw and antenna yaw, pitch) are calculated based on the classical dot product method. For more information, see ```/nmf_ik/head_inverse_kinematics.py```

## Summary of packages

- ```nmf_ik/alignment.py``` : Code for aligning 3D pose to the NMF coordinate system, mostly the same implementation. Alignment consists of scaling the coordinates and translating them to the model joint positions.
- ```nmf_ik/leg_inverse_kinematics.py``` : Code for calculating leg joint angles from the aligned pose, using sequential inverse kinematics.
- ```nmf_ik/head_inverse_kinematics.py``` : Code for calculating head and antenna joint angles from the aligned pose, using vector dot product method.
- ```nmf_ik/data.py``` : Script containing the data shared across modules such as the model joint positions, initial values for the joint angles.
- ```nmf_ik/visualization.py``` : Code for plotting and animating 3D points.
- ```example``` : Samples showing how to run alignment and IK.
- ```data``` : Sample data.
---

## Limitations of head inverse kinematics approach
* Right now, head roll is calculated using a global horizontal axis, which might fail when there is a highly tilted tethered fly. To overcome this, there are two options. The first and preferred option is that tilt is handled by Anipose because we are setting the contralateral key points as one axis, so the Anipose 3D points should be corrected. The second option is that we subract the baseline from the absolute angles, which requires more investigation into the data.

<details closed>

<summary>TO-DO</summary>

  + [x] Adaptation of code to locomotion
  + [ ] Parallelization of leg inv kin calculation (code speed improvement)
  + [x] Head kinematics -> antennal pitch and head roll
</details>
