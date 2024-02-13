# SeqIKPy

A Python package for inverse kinematics in _Drosophila melanogaster_.

# Overview

`SeqIKPy` is a Python package that provides an implementation of inverse kinematics (IK) that is based on the open-source Python package [IKPy](https://github.com/Phylliade/ikpy). In constrast to the current IK approaches that aims to match only the end-effector, `SeqIKPy` is designed to calculate the joint angles of the fly body parts to align the entire kinematic chain to a desired 3D pose.

# Features

* Align of 3D pose data to a fly biomechanical model, e.g., [NeuroMechFly](https://github.com/NeLy-EPFL/NeuroMechFly)
* Calculate leg joint angles using sequential inverse kinematics.
* Calculate head and antenna joint angles using the vector dot product method.
* Visualize and animate the results in 3D.

# Summary of directories

```
.
├── data: Folder containing the sample data.
├── docs: Documentation for the website.
├── examples: Examples and tutorials on how to use the package.
├── seqikpy: Main package.
└── tests: Tests for the package.
```


# Documentation

Documentation can be found [here](https://nely-epfl.github.io/sequential-inverse-kinematics/).

# Installation

You can pip install the package by running the following line in the terminal:
```bash
$ pip install seqikpy
```

Or, you can install the newest version of the package manually by running the following line in the terminal:
```bash
$ pip install https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
```

Note that the IKPy module is added as a submodule. To initialize the submodule, run:
```bash
$ git submodule add https://github.com/gizemozd/ikpy.git ikpy_submodule
$ git submodule update --init
```

# Quick Start



# Contributing
We welcome contributions from the community. If you would like to contribute to the project, please refer to the [contribution guidelines](). Also, read our [code of conduct](). If you have any questions, please feel free to open an issue or contact the developers.

# License
This project is licensed under the [Apache 2.0 License]().

# Issues
If you encounter any bugs or request a new feature, please open an issue in our [issues page]().

# Citing
If you find this package useful in your research, please consider citing it using the following BibTeX entry:

```bibtex
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
