# Installation

You can pip install the package by running the following line in the terminal:
```bash
pip install seqikpy
```

Or, you can install the newest version of the package manually by running the following line in the terminal:
```bash
pip install https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
```

Note that the IKPy module is added as a submodule. To initialize the submodule, run:
```bash
git submodule add https://github.com/gizemozd/ikpy.git ikpy_submodule
git submodule update --init
```