# PyBullet Multi-Robot Coverage
### EK505 Final Project

A simulation framework for multi-robot coverage mapping using the PyBullet physics engine. This project implements autonomous frontier-based exploration with auction-based task allocation and distributed path planning.

---


## Installation

Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### 1. Create the Environment
Create a new Conda environment with all required dependencies (including `scipy` for frontier detection).

```bash
conda create -n pybullet_conda_env -c conda-forge python=3.13 pybullet matplotlib opencv numba scipy pyyaml -y
conda activate pybullet_conda_env