#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "utils_video @ git+https://github.com/NeLy-EPFL/utils_video@replace-deepfly#egg=utils_video-0.1",
    "matplotlib",
    "nptyping",
    "numpy==1.22",
    "scipy==1.8.0",
    "graphviz",
    "poppy",
    "sympy",
    "tqdm",
    "wcmatch",
    "pandas",
    "seaborn",
    "mycolorpy",
]


setup(
    author="Pembe Gizem Ozdil",
    author_email='pembe.ozdil@epfl.ch',
    python_requires='>=3.6',
    description="Inverse kinematics module for Drosophila",
    entry_points={
        'console_scripts': [
            'seqikpy-cli=seqikpy.cli:main',
        ],
    },
    dependency_links=["https://github.com/gizemozd/ikpy.git"],
    install_requires=requirements,
    license="Apache 2.0 License",
    long_description=readme + '\n\n',
    include_package_data=True,
    name='seqIKPy',
    packages=find_packages(include=['seqikpy', 'seqikpy.*']),
    test_suite='tests',
    url='https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git',
    version='1.0.0',
    zip_safe=False,
)
