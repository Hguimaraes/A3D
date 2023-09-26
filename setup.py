#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="a3d",
    version="0.1.0",
    description="Audio Adversarial Attacks and Defenses",
    author="Heitor Guimarães",
    url="https://github.com/Hguimaraes/A3D",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/Hguimaraes/A3D/issues",
        "Source Code": "https://github.com/Hguimaraes/A3D",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.8",
    install_requires=[
        "pandas==2.1.1",
        "torch==2.0.1",
        "torchaudio==2.0.2",
        "speechbrain==0.5.15",
        "transformers==4.33.2",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
)
