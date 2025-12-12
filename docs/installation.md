# Installation

First, create a virtual environment by running the following line in the terminal:
```bash
conda create -n seqikpy python=3.10
```

Note that Python versions from 3.10 to 3.14 are tested and supported. Next, activate the virtual environment by running the following line in the terminal:
```bash
conda activate seqikpy
```

To install `seqikpy`, you can run the following line in the terminal:
```bash
pip install seqikpy
```

You can as well clone the repository and install the package by running the following lines in the terminal:
```bash
git clone https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git
cd sequential-inverse-kinematics
# (OPTIONAL) If you don't have `uv` installed, you can install it with:
pip install uv
# Install the package locally using `uv` (faster than pip):
uv pip install -e . (or uv pip install -e ".[dev]" for development)
```

To check if the installation was successful, you can run the following line in the terminal and see if it throws any errors:
```bash
python -c "import seqikpy"
```

If not, you are ready to use the package!
