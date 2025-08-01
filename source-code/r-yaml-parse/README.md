# Reading YAML files with R

Reading YAML files with R is as trivial as loading the `yaml` library and using
the `yaml.load_file` function.


## What is it?

1. `params.yaml`: YAML file containing parameters to read.
1. `read_yaml.R`: R script that reads the YAML file and prints one of the
   parameters.
1. `environment.yml`: conda environment definition that contains R and the
   `yaml` package.
