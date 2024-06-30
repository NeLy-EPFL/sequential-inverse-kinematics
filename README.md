<div align="center">


<p align="center">
<img src="https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/e10f700ef9dd925b49cb98858763225e3d64bc7b/docs/images/logo.png" width="95%">
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-red)](https://opensource.org/license/apache-2-0)
[![python](https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12601317.svg)](https://doi.org/10.5281/zenodo.12601317)

</div>


# 🪰 Overview

`SeqIKPy` is a Python package that provides an implementation of inverse kinematics (IK) that is based on the open-source Python package [IKPy](https://github.com/Phylliade/ikpy). In constrast to the current IK approaches that aims to match only the end-effector, `SeqIKPy` is designed to calculate the joint angles of the fly body parts to align the 3D pose of the entire kinematic chain to a desired 3D pose. In particular, you can use `SeqIKPy` in the pipeline shown below.

<p align="center">
<img src="https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/a6ec5560825f4d5570dc2f61c9e07fd6e2fbb8d7/docs/images/pipeline.png" width="95%">
</p>


# 📐 Features

* **Pose alignment:** Align of 3D pose data to a fly biomechanical model, e.g., [NeuroMechFly](https://github.com/NeLy-EPFL/NeuroMechFly).
* **Leg inverse kinematics:** Calculate leg joint angles using sequential inverse kinematics.
* **Head inverse kinematics:** Calculate head and antenna joint angles using the vector dot product method.
* **Visualization and animation:** Visualize and animate the results in 3D.

<!-- # Summary of directories

```
.
├── data: Folder containing the sample data.
├── docs: Documentation for the website.
├── examples: Examples and tutorials on how to use the package.
├── seqikpy: Main package.
└── tests: Tests for the package.
``` -->


# 📚 Documentation

Documentation can be found [here](https://nely-epfl.github.io/sequential-inverse-kinematics/).

# 🛠️ Installation

You can pip install the package by running the following line in the terminal:
```bash
$ pip install seqikpy
```

Or, you can install the newest version of the package manually by running the following line in the terminal:
```bash
$ pip install https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
```

# 🏁 Quick Start

Please see the quick start guide [here](https://nely-epfl.github.io/sequential-inverse-kinematics/).


# 💻 Contributing

We welcome contributions from the community. If you would like to contribute to the project, please refer to the [contribution guidelines](https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/e10f700ef9dd925b49cb98858763225e3d64bc7b/CONTRIBUTING.md). Also, read our [code of conduct](https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/e10f700ef9dd925b49cb98858763225e3d64bc7b/CONDUCT.md). If you have any questions, please feel free to open an issue or contact the developers.

# 📖 License

This project is licensed under the [Apache 2.0 License](https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/e10f700ef9dd925b49cb98858763225e3d64bc7b/LICENSE.md).

# 🐞 Issues
If you encounter any bugs or request a new feature, please open an issue in our [issues page](https://github.com/NeLy-EPFL/sequential-inverse-kinematics/issues).

# 💬 Citing
If you find this package useful in your research, please consider citing it using the following BibTeX entry:

```bibtex
@software{ozdil2024seqikpy,
  author       = {Ozdil, Pembe Gizem and Ijspeert, Auke and Ramdya, Pavan},
  title        = {sequential-inverse-kinematics: v1.0.0},
  month        = jun,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.12601317},
  url          = {https://doi.org/10.5281/zenodo.12601317}
} 
```
