# DVC experiment management with R

Although DVC doesn't have a native R interface, you can still use it
effectively with R projects. The code in this directory illustrate how to do
this.


## What is it?

1. `environment.yml`: Conda environment description that contains the R
   packages required to run the code in this directory.  It also includes DVC,
   a Python interpreter and DVCLive.
1. `src/experiment.R`: R script that runs a trivial experiment. It uses the
   `reticulate` package to call Python code that uses DVC to log the parameters
   and metrics of the experiment.
   

## How to use it?

1. Create a conda environment with the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
1. Activate the environment:
  ```bash
  conda activate mlops_dvc_experiments_r
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
1. Ensure that retilucate uses the conda environment, rather than creating its own.
   ```bash
   $ export RETICULATE_PYTHON=$(which python)
   ```
1. Run the experiment:
   ```bash
   $ Rscript src/experiment.R
   ```
