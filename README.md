<div align="center">


<p align="center">
<img src="https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/main/docs/images/logo.png?raw=true" width="95%">
</p>

[![License](https://img.shields.io/badge/License-Apache_2.0-red)](https://opensource.org/license/apache-2-0)
[![python](https://img.shields.io/pypi/pyversions/seqikpy)]()
[![Cov](https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/main/docs/images/coverage.svg?raw=true)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12601317.svg)](https://doi.org/10.5281/zenodo.12601317)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08557/status.svg)](https://doi.org/10.21105/joss.08557)
</div>


# 🪰 Overview

`SeqIKPy` is a Python package that provides an implementation of inverse kinematics (IK) that is based on the open-source Python package [IKPy](https://github.com/Phylliade/ikpy). In constrast to the current IK approaches that aims to match only the end-effector, `SeqIKPy` is designed to calculate the joint angles of the fly body parts to align the 3D pose of the entire kinematic chain to a desired 3D pose. In particular, you can use `SeqIKPy` in the pipeline shown below.

<p align="center">
<img src="https://github.com/NeLy-EPFL/sequential-inverse-kinematics/blob/main/docs/images/pipeline.png?raw=true" width="95%">
</p>


# 📐 Features

* **Pose alignment:** Align of 3D pose data to a fly biomechanical model, e.g., [NeuroMechFly](https://github.com/NeLy-EPFL/NeuroMechFly).
* **Leg inverse kinematics:** Calculate leg joint angles using sequential inverse kinematics.
* **Head inverse kinematics:** Calculate head and antenna joint angles using the vector dot product method.
* **Visualization and animation:** Visualize and animate the results in 3D.

# 🗂️ Summary of directories

```
.
├── data: Folder containing the sample data.
├── docs: Documentation for the website.
├── examples: Examples and tutorials on how to use the package.
├── seqikpy: Main package.
└── tests: Tests for the package.
```


# 📚 Documentation

Documentation can be found [here](https://nely-epfl.github.io/sequential-inverse-kinematics/).

# 🛠️ Installation

If you aim to purely use SeqIKPy as a dependency for your project, you can install the newest version of the package manually by running the following line in the terminal:
```bash
# Optionally create/activate a virtual environment first. Then,
$ pip install seqikpy
```

If you plan to follow the tutorial or help develop SeqIKPy, you should download the `data/` directory, which contains sample data for demonstration and testing. In this case, you should clone this GitHub repository and installing it locally instead of downloading it from PyPI:
```bash
# Optionally create/activate a virtual environment first. Then,
$ git clone git@github.com:NeLy-EPFL/sequential-inverse-kinematics.git
# ... or with HTTP: git clone https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
$ cd sequential-inverse-kinematics/

# Install the package locally (development mode)
$ pip install -e ".[dev]" --config-settings editable_mode=compat
# ... where editable_mode=compat helps static code analyzers in IDEs parse your code better

# Alternatively, use uv for faster dependency resolution (`pip install uv` first)
$ uv pip install -e ".[dev]"
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
@article{ozdil2026seqikpy,
  title={SeqIKPy: a Python package for inverse kinematics in insects},
  author={{\"O}zdil, Pembe Gizem and Wang-Chen, Sibo and Ning, Chuanfang and Ijspeert, Auke and Ramdya, Pavan},
  journal={Journal of Open Source Software},
  volume={11},
  number={117},
  pages={8557},
  year={2026}
}
```
