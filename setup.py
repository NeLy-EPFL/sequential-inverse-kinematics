#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages

setup(
    author="Pembe Gizem Ozdil",
    author_email="pembe.ozdil@epfl.ch",
    python_requires=">=3.8,<3.11",
    description="Inverse kinematics module for Drosophila",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "ikpy==3.3.4",
        "opencv-python==4.5.*",
        "numpy<2.0",
        "tqdm",
        "matplotlib",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ImageHash",
        ]
    },
    license="Apache License 2.0",
    package_data={"seqikpy": ["data/*"]},
    include_package_data=True,
    name="seqikpy",
    packages=find_packages(include=["seqikpy", "seqikpy.*"]),
    test_suite="tests",
    url="https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git",
    version="1.0.1",
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
