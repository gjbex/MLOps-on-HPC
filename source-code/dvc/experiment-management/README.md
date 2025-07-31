# DVC experiment management

This is an illustration of using DVC Live to manage and visualize experiments.
The experiment is not a traditional machine learning experiment, but rather a
Ising model simulation.  This choice was made to emphasize that experiment
management is not limited to machine learning, but can be applied to any
iterative process where you want to track changes and results over time and as
a function of parameters.


## What is it?

1. `environment.yml`: Conda definition for the software environment.
1. `setup_dvc_repo.sh`: script to set up a git repository with DVC and DVC Live.
1. `requirements.txt`: file that contains the Python dependencies for the simulation.
1. `src`: directory that contains the code for the simulation.
1. `params.yaml`: file that contains the parameters for the simulation.
