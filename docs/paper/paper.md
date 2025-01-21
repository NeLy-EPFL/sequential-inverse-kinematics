---
title: 'SeqIKPy: a Python package for inverse kinematics in insects'
tags:
  - Python
  - inverse kinematics
  - motion analysis
  - neuroscience
authors:
  - name: Pembe Gizem Özdil
    orcid: 0000-0003-4507-6642
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Chuanfang Ning
    affiliation: 2
  - name: Auke Ijspeert
    orcid: 0000-0003-1417-9980
    affiliation: 2
  - name: Pavan Ramdya
    orcid: 0000-0001-5425-4610
    affiliation: 1
affiliations:
 - name: Neuroengineering Laboratory, Brain Mind Institute, EPFL, Lausanne, Switzerland
   index: 1
   ror: 00hx57361
 - name: Biorobotics Laboratory, Institute of Bioengineering, EPFL, Lausanne, Switzerland
   index: 2
date: 21 January 2025
bibliography: paper.bib
---

# Summary
`SeqIKPy` is a Python package for inverse kinematics (IK) calculation in animal bodies with complex joint configurations. The name stands for Sequential Inverse Kinematics in Python, as our method computes joint angles sequentially by performing IK for each joint along a kinematic chain.

Our framework contains:
- Pose alignment: map tracked key point locations in 3D onto an animal body template.
- Inverse kinematics: calculate joint angles sequentially from 3D poses.
- Visualization: plot and animate the results in 3D.

`SeqIKPy` is aimed at researchers studying detailed joint motion in animals with complex, multiple degrees-of-freedom body appendages. We provide examples for the fruit fly, *Drosophila melanogaster*. However, each module can easily be extended to be used with another model organism; the only requirements are the 3D kinematics of the target animal and its corresponding kinematic chain. Our package requires minimal Python knowledge and we provide extensive tutorials at [https://nely-epfl.github.io/sequential-inverse-kinematics](https://nely-epfl.github.io/sequential-inverse-kinematics).

# Statement of need

Over the past decade, deep-learning based computer vision algorithms have transformed the analysis of behaviors in laboratory animals [@pereira:2020], including for the widely-used model organism, *Drosophila melanogaster*. Recently, researchers have developed deep learning-based 3D pose estimation tools [@gunel:2019; @karashchuk:2021] and detailed biomechanical models [@lobato-rios:2022;  ; @vaxenburg:2024], creating a growing need for tools to obtain more detailed descriptions of how body parts move in joint space. These computed joint angles can be replayed in physics-based simulations to estimate unmeasured physical quantities like joint torques [@lobato-rios:2022].

Inverse Kinematics (IK) spans multiple domains including robotics, biomechanics, and character animation [@aristidou:2018]. In robotics, IK typically computes joint angles to achieve a desired end-effector position while respecting joint constraints. By contrast, in biomechanics, IK algorithms calculate joint angles to track all marker positions rather than only a single end-effector. This process is also known as multi-body kinematics optimization and is a well-established area in human biomechanics research [@delp:2007; @begon:2018; @pagnon:2022; @werling:2023].

However, multi-body kinematics optimization is still an emerging field in insect research. Existing methods fall on two extremes along a spectrum of complexity. Simple approaches have been employed to compute joint angles using the dot product of two consecutive body segment vectors [@lobato-rios:2022;  @karashchuk:2021]. This often results in deviations from reference poses due to the lack of iterative corrections. On the other hand, more advanced gradient-based optimization methods have been developed to estimate joint angles in a biomechanical model of the fly [@vaxenburg:2024], using simulation-based Jacobians in physics-engines like MuJoCo [@todorov:2012]. Although these methods provide higher accuracy, they require complex setups that are not easily accessible to all researchers. Therefore, there is no stand-alone inverse kinematics software satisfying the trade-off between accuracy and complexity.

To address this gap, we have developed `SeqIKPy`, a fast and lightweight Python package for multi-body kinematics optimization in insects. Our package has two main stages: marker registration (aligning joint markers to a template in 3D) and inverse kinematics. The second stage draws upon the open-source IKPy [@Manceron_IKPy] library in a sequential manner. Through 3D visualizations, we demonstrate that our package can reliably reconstruct body kinematics for various fruit fly behaviors. Furthermore, recent studies have shown that the joint angles computed using `SeqIKPy` can accurately replicate animal behavior, both ground walking [@wang:2024] and grooming [@ozdil:2024], in physics-based simulations. Although our examples focus on the fly, our package's modular design allows for customization and extension to other animals with a similar body morphology.

# Overview

`SeqIKPy` assumes that the 3D pose estimation has the following orientation (\autoref{fig:pipeline}, left):
- x-axis: anterioposterior axis
- y-axis: mediolateral axis
- z-axis: dorsoventral axis

After setting this orientation, users can use the `AlignPose` class to map body keypoints to a template body model (\autoref{fig:pipeline}, middle). Despite being optional for inverse kinematics, this step has two benefits:
- It aligns measured kinematics to a standardized body template, facilitating replay of behaviors in body models (\autoref{fig:pipeline}, right).
- It reduces noise and variation in kinematics by standardizing body lengths.

We provide a default body template based on a CT scan of the fly [@lobato-rios:2022]. Users can also define custom templates manually or by importing SDF files. Utility functions are included to convert data into the required formats.

Next, the `KinematicChainSeq` class defines a pre-configured kinematic chain for fly legs. Users need a dictionary containing segment lengths and joint bounds (\autoref{fig:pipeline}, middle). Segment lengths can be derived from 3D kinematics, while joint bounds are optional. Using the defined kinematic chain, the `LegInvKin` class calculates joint angles sequentially for each leg. This process supports parallelization to perform IK on multiple legs simultaneously. Additionally, our package contains features for animation and visualization of data in 3D. For more technical details on the implementation, please refer to the methodology section at [https://nely-epfl.github.io/sequential-inverse-kinematics](https://nely-epfl.github.io/sequential-inverse-kinematics).

On a MacBook Pro with a 2.3 GHz Quad-Core Intel Core i7, the pipeline computes inverse kinematics for all six legs across 100 frames in 36 s when parallelized.

![Overview of the SeqIKPy pipeline. **(Left)** Pose estimation tools (e.g., DeepFly3D and Anipose) estimates 3D positions of body key points. **(Middle)** SeqIKPy aligns these points to a body template, performs sequential inverse kinematics using joint constraints to calculate joint angles. **(Right)** The computed joint angles can be used to replay measured motions in biomechanical models, such as NeuroMechFly.\label{fig:pipeline}](pipeline.png)


# Acknowledgements

We acknowledge the contributors and maintainers of the open-source software tools that SeqIKPy builds upon, including Python, NumPy, SciPy, Matplotlib, and IKPy [@Manceron_IKPy], among others. PGÖ acknowledges support from a Swiss Government Excellence Scholarship for Doctoral Studies and a Google PhD Fellowship. PR acknowledges support from an SNSF Project Grant (175667) and an SNSF Eccellenza Grant (181239).

# References