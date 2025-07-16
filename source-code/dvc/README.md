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
* `train.py`: Python script to train a model using the generated data.
* `compute_metrics.py`: Python script to compute metrics from the trained model.
* `predict.py`: Python script to make predictions using the trained model.
* `data`: Directory to store the generated data (CSV).
* `models`: Directory to store the trained models (pkl).
* `params`: Directory to store hyperparameters for the models (JSON).
* `metrics`: Directory to store computed metrics (JSON).
* `predictions`: Directory to store predictions (CSV).


## How to use it?

We assume that we have the data sets given, i.e.,
* `data/train.csv`: Training data
* `data/test.csv`: Test data
* `data/production.csv`: Production data


### Initializing DVC

To initialize DVC in your project, run the following command in the root
directory of your project (where the `.git/` directory is located) to create a
`.dvc/` directory and a `.dvcignore` file. 

```bash
$ dvc init
```

Set a remote storage for DVC to store the data and model files. This can be a
cloud storage service like AWS S3, Google Cloud Storage, or Azure Blob Storage.
However, it can also be a local directory. For example, to set a local directory as
the remote storage, run the following command:

```bash
$ dvc remote add -d local_storage /path/to/storage
```

### Adding data to DVC

To add the data files to DVC, run the following commands:

```bash
$ dvc add data/train.csv
$ dvc add data/test.csv
$ dvc add data/production.csv
```

### Training a model

To train a model, run the following command:

```bash
$ dvc run -n train_model \
  -d data/train.csv -d params/train_params.json \
  -o models/model.pkl \
  python train.py
```

### Computing metrics

To compute metrics from the trained model, run the following command:

```bash
$ dvc run -n compute_training_metrics \
  -d models/model.pkl -d data/train.csv \
  -o metrics/training_metrics.json \
  python compute_metrics.py
$ dvc run -n compute_test_metrics \
  -d models/model.pkl -d data/test.csv \
  -o metrics/test_metrics.json \
  python compute_metrics.py
```

### Making predictions

To make predictions using the trained model, run the following command:

```bash 
$ dvc run -n make_predictions \
  -d models/model.pkl -d data/production.csv \
  -o predictions/predictions.csv \
  python predict.py
```

### Pushing changes to remote storage

To push the changes to remote storage, run the following command:

```bash
$ dvc push
```

### Pulling changes from remote storage

To pull the changes from remote storage, run the following command:

```bash
$ dvc pull
```
