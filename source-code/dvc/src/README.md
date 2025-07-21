# src

Python scripts that define steps in the workflow.


## What is it?

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
