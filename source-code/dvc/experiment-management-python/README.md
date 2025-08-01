# DVC experiment management with Python

The DVCLive Python package helps manage experiments in Python scripts by providing
facilities to log parameters, metrics and other artifacts of the experiments.  This
directory contains a minimal example.
this.


## What is it?

1. `environment.yml`: Conda environment description that contains the software
   packages required to run the code in this directory.  It includes DVC,
   a Python interpreter and DVCLive.
1. `requirements.txt`: Python packages required to run the code in this directory.
   It includes DVC, a Python interpreter and DVCLive in case you prefer to use
   venv instead of conda.
1. `src/experiment.py`: Python script that runs a trivial experiment and
   uses DVC to log the parameters and metrics of the experiment.
   

## How to use it?

1. Create a conda environment with the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
1. Activate the environment:
  ```bash
  conda activate mlops_dvc
  ```
1. Create a git repository and initialize DVC:
   ```bash
   mkdir /tmp/ml_repo
   cp -r src/ /tmp/ml_repo
   cd /tmp/ml_repo
   git init
   dvc init
   git add src/
   git commit -m "Initialize DVC experiment management with R"
   ```
1. Run the experiment:
   ```bash
   $ python src/experiment.R
   ```
