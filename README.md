# PyBullet Multi-Robot Coverage
### EK505 Final Project

A simulation framework for multi-robot coverage algorithms using the PyBullet physics engine.

---

## Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your machine.

## Installation

To set up the project environment, run the following commands in your terminal:

### 1. Create the Environment
This command creates a new Conda environment named `pybullet_conda_env` and installs the required dependencies (`pybullet`, `matplotlib`, `opencv`, `numba`) from the `conda-forge` channel.

```bash
conda create -n pybullet_conda_env -c conda-forge python=3.13 pybullet matplotlib opencv numba -y