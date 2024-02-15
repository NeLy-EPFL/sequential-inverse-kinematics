#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy==1.22",
    "scipy==1.8.0",
    "tqdm",
    "opencv-python",
    "nptyping",
    "pandas",
    "mycolorpy",
    "matplotlib",
    "ikpy @ git+https://github.com/gizemozd/ikpy.git#egg=ikpy",
]


setup(
    author="Pembe Gizem Ozdil",
    author_email='pembe.ozdil@epfl.ch',
    python_requires='>=3.8,<3.12',
    description="Inverse kinematics module for Drosophila",
    install_requires=requirements,
    license="Apache 2.0 License",
    long_description=readme + '\n\n',
    package_data={"seqikpy": ["data/*"]},
    include_package_data=True,
    name='SeqIKPy',
    packages=find_packages(include=['seqikpy', 'seqikpy.*']),
    test_suite='tests',
    url='https://github.com/NeLy-EPFL/sequential-inverse-kinematics.git',
    version='1.0.0',
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
    ],
    keywords='inverse kinematics, robotics, Drosophila, motion analysis',

)
