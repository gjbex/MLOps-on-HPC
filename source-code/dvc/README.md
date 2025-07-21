# DVC

DVC (Data Version Control) can be used to manage data and model files in a
machine learning project. It allows you to track changes, share datasets, and
collaborate with others.  You can view DVC a no-fuss MLOps alternative.

Note that a number of design decisions have been made to keep the project
simple and easy to understand.  This is not a production-ready project, but
rather a learning tool to understand how DVC can be used in a machine learning
project.


## What is it?

* `generate_data.py`: Python script to generate synthetic data.
* `plot_data.py`: Python script to plot the data.
* `data`: directory with the data files.
* `src`: directory with the Python scripts required to execute the workflow.
* `params.yaml`: YAML file to store parameters for the scripts.
* `setup.py`: Python script to set up the environment.
* `requirements.txt`: Python requirements file to install the required packages.



## How to use it?

### Setting up the project

First, create the directory structure for the project.  Note that the directory
you choose for this should not be a git repository.

```bash
$ mkdir ~/dvc_project
```

Then execute the `setup.py` script to create the directory structure and
copy the source code and data.

```bash
$ python setup.py ~/dvc_project
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

