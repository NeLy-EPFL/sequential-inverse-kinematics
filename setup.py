#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    author="Pembe Gizem Ozdil",
    author_email="pembe.ozdil@epfl.ch",
    python_requires=">=3.8,<3.11",
    description="Inverse kinematics module for Drosophila",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
        ]
    },
    license="Apache License 2.0",
    package_data={"seqikpy": ["data/*"]},
    include_package_data=True,
    name="seqikpy",
    packages=find_packages(include=["seqikpy", "seqikpy.*"]),
    test_suite="tests",
    url="https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git",
    version="1.0.0",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="inverse kinematics, robotics, Drosophila, motion analysis",
)
