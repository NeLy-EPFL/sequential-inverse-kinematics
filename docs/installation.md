# Installation

First, create a virtual environment by running the following line in the terminal:
```bash
conda create -n seqikpy python=3.8
```

Note that Python versions 3.8, 3.9, 3.10 are tested and supported. Next, activate the virtual environment by running the following line in the terminal:
```bash
conda activate seqikpy
```

To install `seqikpy`, you can run the following line in the terminal:
```bash
pip install seqikpy
```

Or, you can install the newest version of the package manually by running the following line in the terminal:
```bash
pip install https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
```

You can as well clone the repository and install the package by running the following lines in the terminal:
```bash
git clone https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
cd sequential-inverse-kinematics
pip install -e . (or pip install -e ".[dev]" for development)
```

To check if the installation was successful, you can run the following line in the terminal and see if it throws any errors:
```bash
python -c "import seqikpy"
```

If not, you are ready to use the package!
