# Pipelines and experiments

DVCLive can also be used by stages of a DVC pipeline. This allows you to track
metrics and parameters of each stage in a DVC pipeline, making it easier to
manage experiments.  Using the DVC queue, you can run multiple experiments in
parallel, and easily run parameter sweeps to compare their impact.

This directory contains the source code for a single-stage pipeline that runs
an experiment and tracks the results using DVCLive.


## What is it?

1. `environment.yaml`: Conda environment description for the dependencies
   required to run the experiment, including DVC.
1. `src/experiment.py`: Python script that runs the experiment and logs metrics
   using DVCLive.
1. `params.yaml`: YAML file that contains the parameters for the experiment,
   read by `src/experiment.py`.



## How to use?

Add the experiment as a stage to the pipeline:
```bash
$ dvc stage add --name 'computation' \
    --deps src/experiment.py \
    --params a,t_max \
    --metrics result.tsv \
    python src/experiment.py  --params params.yaml
```

Run the stage:
```bash
$ dvc exp run
```
