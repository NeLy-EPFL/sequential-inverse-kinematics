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
  - name: Sibo Wang-Chen
    orcid: 0000-0002-7601-8886
    affiliation: 1
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
 - name: Biorobotics Laboratory, Institute of Bioengineering, EPFL, Lausanne, Switzerland
   index: 2
date: 21 January 2025
bibliography: paper.bib
---

# Summary
`SeqIKPy` is a Python package for inverse kinematics (IK) calculations in animal bodies with complex joint configurations. The name stands for Sequential Inverse Kinematics in Python, as our method computes joint angles sequentially by performing IK for each joint along a kinematic chain.

Our framework contains:

* Pose alignment: map tracked keypoint locations in 3D onto an animal body template
* Inverse kinematics: calculate joint angles sequentially from 3D poses
* Visualization: plot and animate the results in 3D

`SeqIKPy` is aimed at researchers studying detailed joint motion in animals with complex, multiple degrees of freedom body appendages. We provide examples for the fruit fly, *Drosophila melanogaster*. However, each module can easily be extended to be used with other model organisms; the only requirements are the 3D kinematics of the target animal and its corresponding kinematic chain. Our package requires minimal Python knowledge, and extensive tutorials are available to novice users at [https://nely-epfl.github.io/sequential-inverse-kinematics](https://nely-epfl.github.io/sequential-inverse-kinematics).

# Statement of need

Over the past decade, deep-learning–based computer vision algorithms have transformed behavioral analysis in laboratory animals [@pereira:2020]. More recently, deep-learning–based 3D pose estimation tools [@gunel:2019; @karashchuk:2021] and detailed biomechanical models [@lobato-rios:2022; @wang:2024; @vaxenburg:2024] have created a growing need for tools that translate 3D kinematics into joint-angle representations. These joint angles can then be replayed in physics-based simulations to estimate unmeasured physical quantities such as joint torques [@lobato-rios:2022].

Inverse kinematics (IK) is widely used in robotics, biomechanics, and character animation [@aristidou:2018]. In robotics, IK typically computes joint angles to achieve a desired end-effector position while respecting joint constraints of a robot. In contrast, biomechanical IK aims to track multiple body markers simultaneously, a process often referred to as multi-body kinematics optimization and well-established in human biomechanics [@delp:2007; @begon:2018; @pagnon:2022; @werling:2023].

In insect research, however, multi-body kinematics optimization remains relatively underdeveloped. Existing approaches lie at two extremes. Simple geometric methods compute joint angles from dot products between adjacent body segments [@lobato-rios:2022; @karashchuk:2021], often leading to cumulative errors due to the lack of iterative corrections. More advanced gradient-based optimization methods estimate joint angles in a biomechanical model of the fly [@vaxenburg:2024], using simulation-based Jacobians in physics engines like MuJoCo [@todorov:2012]. Although these methods provide higher accuracy, they are often entangled with pose estimation and physics engines, leading to complex dependencies and heavy overhead. Here, we introduce `SeqIKPy` as a lightweight, standalone, and modular solution focused on inverse kinematics.

`SeqIKPy` consists of two main stages: marker registration (aligning measured keypoints to a 3D body template) and inverse kinematics. The latter stage builds on the open-source IKPy library [@Manceron_IKPy] and applies it sequentially along the kinematic chain. The sequential nature of our method constrains the solution locally and mitigates ambiguities arising from joint redundancy. Using 3D visualizations, we show that SeqIKPy reliably reconstructs fly kinematics across a range of behaviors. Previous work has demonstrated that the joint angles computed with SeqIKPy accurately reproduce both walking [@wang:2024] and grooming [@ozdil:2024] behaviors in physics-based simulations. Although our examples mostly focus on the fly, we demonstrate that its modular design allows adaptation to other animals with articulated limbs, including a mouse forelimb.

`SeqIKPy` can be used for animals and robots with arbitrarily configured kinematic chains consisting of rigid bodies connected with rotational joints. However, we have focused on the fruit fly _Drosophila melanogaster_ in our demonstrations. Insects are some of the oldest model organisms in the study of motor control [@delcomyn:2004], and _Drosophila melanogaster_ is particularly prominent in neuroscience research due to its compact yet versatile nervous system. Recent advances have made the fly the most complex organism with a fully mapped central nervous system [e.g., @bates:2025], leading to rapid growth in the field and an increased demand for open-source data-processing tools. With `SeqIKPy`, we aim to address a critical gap in the behavioral analysis pipeline.


# Overview

`SeqIKPy` assumes that the 3D pose estimation has the following orientation (\autoref{fig:pipeline}, left):

* x-axis: anteroposterior axis
* y-axis: mediolateral axis
* z-axis: dorsoventral axis

After setting this orientation, users can use the `AlignPose` class to map body keypoints to a template body model (\autoref{fig:pipeline}, middle). This step is also known as "calibration" in pose estimation literature. Despite being optional for inverse kinematics computation, this step has two benefits:

* It aligns measured kinematics to a standardized body template, facilitating replay of behaviors in body models (\autoref{fig:pipeline}, right).
* It reduces noise and variation in kinematics by standardizing body lengths.


We provide a default body template based on a CT scan of the fly [@lobato-rios:2022]. Users can also define custom templates by importing SDF files using the `from_sdf` functionality, which extracts joint locations directly from the model definition, or by specifying templates manually. Utility functions are included to convert data into the required formats.

Next, the `KinematicChainSeq` class defines a pre-configured kinematic chain for fly legs. Users need a dictionary containing segment lengths and joint bounds (\autoref{fig:pipeline}, middle). Segment lengths can be derived from 3D kinematics, while joint bounds are optional. Using the defined kinematic chain, the `LegInvKin` class calculates joint angles sequentially for each leg. This process supports parallelization to perform IK on multiple legs simultaneously. Additionally, our package contains features for animation and visualization of data in 3D. For more technical details on the implementation, please refer to the methodology section at [https://nely-epfl.github.io/sequential-inverse-kinematics](https://nely-epfl.github.io/sequential-inverse-kinematics).

![Overview of the SeqIKPy pipeline. **(Left)** Pose estimation tools (e.g., DeepFly3D and Anipose) estimates 3D positions of body keypoints. **(Middle)** SeqIKPy aligns these points to a body template, performs sequential inverse kinematics using joint constraints to calculate joint angles. **(Right)** The computed joint angles can be used to replay measured motions in biomechanical models, such as NeuroMechFly.\label{fig:pipeline}](pipeline.png)

# Limitations and mitigation
Like `IKPy`, `SeqIKPy` solves inverse kinematics through per-frame optimization. While this approach generally leads to a more faithful fit between raw data and the inferred kinematics, a main disadvantage is that processing can be slow. For use cases requiring higher throughput but not requiring sequential fitting over the whole kinematic chain, we refer the user to the `ik` module [@aversiveplusplus_ik] from the Aversive++ project. Alternatively, inverse kinematics can be implemented using approaches that do not require explicit optimization loops for every frame. These include methods based on Extended Kalman Filters [e.g., @Bonnet2017; @Fohanno2010; @Ceglia2025], and methods based on deep neural networks [e.g., @Toquica2021; @Wang2021]. These methods can also implicitly solve for the angles of all degrees of freedom on each kinematic chain simultaneously, instead of running a sequential optimization. At the cost of losing explicit per-frame accuracy, these methods are much faster and, in many cases, can run in real-time.

For use cases where sequential inverse kinematics is required, we have implemented parallel processing in `SeqIKPy` in order to improve the throughput. Parallelism is implemented over different kinematic chains and over time, with the size of each atomic task determined adaptively to balance the trade-off between load balancing and overhead. On a typical machine, the overall processing rate scales near-linearly up to the number of physical CPU cores (~90% efficiency) and, to a lesser extent, up to the number of logical threads (~80% efficiency). On an Intel Xeon Platinum 8360Y processor, this translates to ~2ms per frame with 36 processes.

# Acknowledgements

We acknowledge the contributors and maintainers of the open-source software tools that `SeqIKPy` builds upon, including Python, NumPy, SciPy, Matplotlib, and IKPy [@Manceron_IKPy], among others. PGÖ acknowledges support from a Swiss Government Excellence Scholarship for Doctoral Studies and a Google PhD Fellowship. SWC acknowledges support from a Boehringer Ingelheim Fonds PhD fellowship. PR acknowledges support from an SNSF Project Grant (175667) and an SNSF Eccellenza Grant (181239).

# References
