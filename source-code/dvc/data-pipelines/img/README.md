# Images

Visualizations of the example pipeline.


## What is it?

1. `dvc_dag.png`: A directed acyclic graph (DAG) showing the dependencies
   between the stages in the DVC pipeline. Each node represents a stage, and
   arrows indicate the flow of data between them.
1. `dvc_dag_data.png`: A directed acyclic graph (DAG) showing the data file
   dependencies.


## How to create them?

Run the following DVC command in the project directory:

```bash
$ dvc dag --dot | dot -Tpng -o img/dvc_dag.png
$ dvc dag --dot --outs | dot -Tpng -o img/dvc_dag_data.png
```
