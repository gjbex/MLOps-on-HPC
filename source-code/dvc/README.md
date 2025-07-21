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
* `split_data.py`: Python script to split the generated data into training and
  testing sets.
* `train_preprocessor.py`: Python script to train a preprocessor (e.g.,
  scaling, encoding) on the training data.
* `preprocess.py`: Python script to preprocess a data file using the trained
  preprocessor.
* `train_model.py`: Python script to train a logistic regression model using
  the preprocessed data.
* `compute_metrics.py`: Python script to compute metrics from the trained
  model.
* `predict.py`: Python script to make predictions using the trained model.
* `data`: Directory to store the generated data (CSV).
* `params.yaml`: YAML file to store parameters for the scripts.


## How to use it?

We assume that we have the data sets given, i.e., `data/data.csv`.


### Initializing DVC

To initialize DVC in your project, run the following command in the root
directory of your project (where the `.git/` directory is located) to create a
`.dvc/` directory and a `.dvcignore` file. 

```bash
$ dvc init
```

Set a remote storage for DVC to store the data and model files. This can be a
cloud storage service like AWS S3, Google Cloud Storage, or Azure Blob Storage.
However, it can also be a local directory. For example, to set a local
directory as the remote storage, run the following command:

```bash
$ dvc remote add -d local_storage /path/to/storage
```

### Adding data to DVC

To add the data files to DVC, run the following commands:

```bash
$ dvc add data/data.csv
```

