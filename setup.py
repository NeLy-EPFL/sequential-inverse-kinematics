#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages


setup(
    author="Pembe Gizem Ozdil",
    author_email='pembe.ozdil@epfl.ch',
    python_requires='>=3.8,<3.12',
    description="Inverse kinematics module for Drosophila",
    long_description=open("README.md").read(), long_description_content_type='text/markdown',
    install_requires=[
        "numpy==1.22",
        "scipy==1.8.0",
        "tqdm",
        "opencv-python",
        "nptyping",
        "pandas",
        "mycolorpy",
        "matplotlib",
        "ikpy",
    ],
    license="Apache 2.0 License",
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
