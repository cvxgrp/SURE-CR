#!/usr/bin/python
from setuptools import setup

with open("README.md", "r") as fh:
   long_description = fh.read()

setup(
    name="surecr",
    setup_requires=["setuptools>=18.0"],
    install_requires=[
        "numpy >= 1.17.5",
        "scipy",
        "torch",
        "cvxpy",
        "cvxpylayers",
        "torch-linops",
    ],
    url="https://github.com/cvxgrp/SURE-CR",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Parth Nobel",
    author_email="ptnobel@stanford.edu",
)
