"""
Utility functions for common operations across scripts.
"""
import pandas as pd
import yaml
import pickle
import pathlib
from sklearn.model_selection import train_test_split

def load_csv(path, dtype=None):
    """Load a CSV file into a pandas DataFrame with optional dtype specification."""
    return pd.read_csv(path, dtype=dtype)

def load_yaml(path):
    """Load YAML configuration from a file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_pickle(path):
    """Load a pickle object from a file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    """Save an object to a pickle file, creating parent directories if needed."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def split_train_test(data, test_size, random_state):
    """Split data into train and test sets."""
    return train_test_split(data, test_size=test_size, random_state=random_state)

def save_train_test(train_data, test_data, output_dir):
    """Save train and test DataFrames to CSV files in the specified directory."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(path / 'train.csv', index=False)
    test_data.to_csv(path / 'test.csv', index=False)

def save_data(data, path):
    """Save DataFrame to CSV file."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
