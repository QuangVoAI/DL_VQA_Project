"""
Setup script for VQA Project

This allows the project to be installed as a package for easier imports.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vqa-project",
    version="0.1.0",
    author="Quang and Thành",
    description="Visual Question Answering system using CNN-LSTM architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QuangVoAI/DL_VQA_Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vqa-train=src.train:main",
        ],
    },
)
