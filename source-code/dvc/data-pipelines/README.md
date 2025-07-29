# DVC data pipelines

DVC (Data Version Control) can be used to manage data and model files in a
machine learning project. It allows you to track changes, share datasets, and
collaborate with others.  You can view DVC a no-fuss MLOps alternative.

Note that a number of design decisions have been made to keep the project
simple and easy to understand.  This is not a production-ready project, but
rather a learning tool to understand how DVC can be used in a machine learning
project.


## What is it?

* `environment.yml`: Conda environment file to create the environment with the
  required packages and software, including DVC.
* `generate_data.py`: Python script to generate synthetic data.
* `plot_data.py`: Python script to plot the data.
* `data`: directory with the data files.
* `src`: directory with the Python scripts required to execute the workflow.
* `params.yaml`: YAML file to store parameters for the scripts.
* `setup.sh`: Bash script to set up the environment (same functionality as setup.py).
* `requirements.txt`: Python requirements file to install the required packages.
* `init_repository.sh`: shell script to initialize the project directory and
  set up the DVC remote storage.
* `define_workflow.sh`: shell script to define the workflow and add the
  stages to DVC.
* `img`: directory with images to visualize the pipeline in terms of stages and data.


## How to use it?

### Setting up the project

First, create the directory structure for the project.  Note that the directory
you choose for this should not be a git repository.

```bash
$ mkdir ~/dvc_project
```

Then execute the `setup.sh` script to create the directory structure and
copy the source code and data.

```bash
$ bash setup.sh ~/dvc_project
```

Two directories are created in `~/dvc_project`:
* `ml_project`: directory with the source code and data files.
* `dvc_data`: directory that will act as a remote for DVC.

Next, change to the `ml_project` directory:

```bash
$ cd ~/dvc_project/ml_project
```

Initialize it as a git repository:

```bash 
$ git init
```

Add the `src` directory, the `params.yaml`, and the `requirements.txt` files
to the git repository.  Do **not** add the `data` directory since this will
under DVC control.

```bash
$ git add src params.yaml requirements.txt
$ git commit -m 'Initial commit'
```

Create a virtual environment and install the required packages:

```bash
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Add the `env` directory to the `.gitignore` file to avoid committing the
virtual environment to the git repository.  Add and commit the file to the repository.

```bash
$ echo "env/" >> .gitignore
$ git add .gitignore
$ git commit -m 'Add .gitignore file'
```

Initialize DVC so that it can manage the data files and model files in the
project.  This will create a `.dvc/` directory in the root of the project
directory, which will contain the DVC configuration files. It will also create
a `.dvcignore` file to ignore files that should not be tracked by DVC, such as
the virtual environment directory and other temporary files.  Commit the files
to the git repository that were created and added by the `dvc init` command.

```bash 
$ dvc init
$ git commit -m 'Initialize DVC'
```

Add a remote to DVC to store the data and model files. In this examples, you
can use the local directory `dvc_data` as the remote storage.  This will allow you to
store the data and model files in a separate directory, which can be useful for
collaboration and sharing.  Run the following command to add the remote storage.  Since this changes the DVC configuration file, add and commit it to the git repository.

```bash 
$ dvc remote add -d local_storage ../dvc_data
$ git add .dvc/config
$ git commit -m 'Add remote storage for DVC'
```

Everything is now in place, and you can start using DVC to manage the data
and model files in your project.

All these steps can be executed by running the `init_repository.sh` script
in the `ml_project` directory.


### Adding data to DVC

To add the data files to DVC, run the following commands:

```bash
$ dvc add data/data.csv
```

This will create a `.dvc` file in the `data` directory, which contains the
metadata for the data file.  The actual data file will be stored in the remote
storage directory `dvc_data`.  Add the `.dvc` file to the git repository and
commit the changes:

```bash
$ git add data/data.csv.dvc
$ git commit -m 'Add data file to DVC'
```

It is more convenient to let DVC automatically stage these files for you.  You
can do this by configuring DVC to automatically stage the `.dvc` files when you
run `dvc add`.  To do this, run the following command:

```bash
$ dvc config core.autostage true
```

Add the `.dvc/config` file to the git repository and commit the changes:

```bash
$ git add .dvc/config
$ git commit -m 'Configure DVC to automatically stage .dvc files'
```

You can now push the data files to the remote storage by running:

```bash 
$ dvc push
```


### Defining the workflow

Each step in the workflow is called a stage in DVC.  A stage is defined by a
command that produces an output file.  The command can be a Python script, a
shell command, or any other command that can be executed in the terminal.  The
output file can be a data file, a model file, or any other file that is
produced by the command.  The output can also be specified as a directory. The
output file is stored in the remote storage directory, and the metadata for the
stage is stored in a `.dvc` file in the project directory.


The first stage is to split the data into training and test sets.  This is done by
the `split_data.py` script, which takes the data file as input and produces two
output files: `train.csv` and `test.csv`.  The script uses the parameters
`test_size` and `random_state` from the `params.yaml` file to control the
splitting of the data.  The command to add this stage to DVC is as follows:

```bash
dvc stage add --name split_data \
    --deps data/data.csv \
    --deps src/split_data.py --deps src/utils.py \
    --params split_data.test_size,split_data.random_state \
    --outs data/split_data/train.csv \
    --outs data/split_data/test.csv \
    python src/split_data.py \
        --data data/data.csv \
        --params params.yaml \
        --output data/split_data/
```

When the first stage is added, DVC will automatically create a file called
`dvc.yaml` in the root of the project directory.  This file contains the
metadata for the stage, as well as a `.gitignore` file that ignores the
artifacts produced by the stage.  The `dvc.yaml` is staged for commit, so you can
add it to the git repository and commit the changes:

```bash
$ git commit -m 'Add split_data stage to DVC'
```

It is good practice to commit each time you add a new stage.

The second stage is to train a preprocessor on the training data.  This is done by
the `train_preprocessor.py` script, which takes the training data file as input
and produces a preprocessor file `preprocessor.pkl`.  The preprocessor is used to
transform the data before training the model.  The command to add this stage to
DVC is as follows:

```bash
dvc stage add --name train_preprocessor \
    --deps data/split_data/train.csv \
    --deps src/train_preprocessor.py --deps src/utils.py \
    --outs ./preprocessor.pkl \
    python src/train_preprocessor.py \
        --data data/split_data/train.csv \
        --output preprocessor.pkl
```

The third stage is to preprocess the training data using the preprocessor file
`preprocessor.pkl`.  This is done by the `preprocess.py` script, which takes the
training data file and the preprocessor file as input and produces a preprocessed
training data file `train.csv` in the `data/preprocessed/` directory.  The command
to add this stage to DVC is as follows:

```bash
dvc stage add --name preprocess_train \
    --deps data/split_data/train.csv \
    --deps preprocessor.pkl \
    --deps src/preprocess.py --deps src/utils.py \
    --outs data/preprocessed/train.csv \
    python src/preprocess.py \
        --data data/split_data/train.csv \
        --preprocessor preprocessor.pkl \
        --output data/preprocessed/train.csv
```

The fourth stage is to preprocess the test data, similar to the training data.

```bash
dvc stage add --name preprocess_test \
    --deps data/split_data/test.csv \
    --deps preprocessor.pkl \
    --deps src/preprocess.py --deps src/utils.py \
    --outs data/preprocessed/test.csv \
    python src/preprocess.py \
        --data data/split_data/test.csv \
        --preprocessor preprocessor.pkl \
        --output data/preprocessed/test.csv
```

The fifth stage is to train the model using the preprocessed training data.  This
is done by the `train_model.py` script, which takes the preprocessed training data
file and the parameters from the `params.yaml` file as input and produces a model
file `model.pkl`.  The command to add this stage to DVC is as follows:

```bash
dvc stage add --name train_model \
    --deps data/preprocessed/train.csv \
    --deps src/train_model.py --deps src/utils.py \
    --params train_model.penalty,train_model.C,train_model.solver \
    --outs model.pkl \
    python src/train_model.py \
        --data data/preprocessed/train.csv \
        --params params.yaml \
        --output model.pkl
```

The sixth stage is to evaluate the model on the preprocessed training data.  This
is done by the `evaluate_model.py` script, which takes the preprocessed training
data file and the model file as input and produces a metrics file
`metrics/train.yaml`.  The command to add this stage to DVC is as follows:

```bash
dvc stage add --name compute_metrics_train \
    --deps data/preprocessed/train.csv \
    --deps model.pkl \
    --deps src/compute_metrics.py --deps src/utils.py \
    --metrics metrics/train.yaml \
    python src/compute_metrics.py \
        --data data/preprocessed/train.csv \
        --model model.pkl \
        --output metrics/train.yaml
```

The seventh stage is to evaluate the model on the preprocessed test data, similar to
the training data.

```bash
dvc stage add --name compute_metrics_test \
    --deps data/preprocessed/test.csv \
    --deps model.pkl \
    --deps src/compute_metrics.py --deps src/utils.py \
    --metrics metrics/test.yaml \
    python src/compute_metrics.py \
        --data data/preprocessed/test.csv \
        --model model.pkl \
        --output metrics/test.yaml
```

The stages are defined in the shell script `define_workflow.sh` to make it
easier to add them to DVC.  You can run the script to add all the stages to
DVC at once.

All these commands simply define the workflow without executing it.  To
execute the workflow, you can run the following command:

```bash
$ dvc repro
```

This will execute all the stages in the workflow in the correct order, based on
the dependencies defined in the `dvc.yaml` file.  DVC will automatically
download the data files from the remote storage, execute the commands. The
output files will be stored in the remote storage directory `dvc_data`, and the
metadata for the stages will be stored in the `dvc.yaml` file.  You can add the
`-f` flag to force DVC to execute the stages even if the output files already
exist.  This can be useful if you want to re-execute the stages with different
parameters or if you want to re-execute the stages after making changes to the
source code.

When the workflow finishes, you should commit the files that DVC has created
to the git repositoryh, and you can push the results to the remote storage by
running:

```bash
$ git commit -m 'Run workflow'
$ dvc push
```

You can view the status of the DVC repository by running:

```bash
$ dvc dag
```


### Changing things

When you change something to the workflow, such as the parameters in the
`params.yaml` file, you can run the `dvc repro` command again to re-execute the
stages that depend on the changed parameters.  DVC will automatically detect
the changes and re-execute the stages in the correct order.  You can also
force DVC to re-execute all the stages by running the `dvc repro -f` command.

For instance, if you change the `train_model.penalty` parameter in the
`params.yaml` file to `'l2'`, you can run the following command to re-execute the
stages that depend on this parameter:

```bash
$ git commit -a -m 'Change train_model.penalty parameter to l2'
$ dvc repro
$ git commit -m 'Rerun for change train_model.penalty'
$ dvc push
```

This will re-execute the `train_model` stage and all the stages that depend on
it, such as `compute_metrics_train` and `compute_metrics_test`.

You can now easily compare the metrics of the old and new runs by
first finding the commit hash of the old run using `git log` and then
using the `dvc metrics diff` command to compare the metrics of the two runs:

```bash
$ git log
$ dvc metrics diff <old_commit_hash> HEAD
```

Similarly, you can compare the parameters of the two runs by using the
`dvc params diff` command:

```bash
$ dvc params diff <old_commit_hash> HEAD
```

