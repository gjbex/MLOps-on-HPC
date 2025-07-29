#!/usr/bin/env python3

# Python script that trains a logistic regression model on data generated
# by the `generate_data.py` script.
# It takes the following command line arguments:
# --data: Path to the training data file (default: 'data/train.csv')
# --output: Path to save the trained model (default: 'models/model.pkl')
# --params: Path to the file that defines the hyperparameters for the model
# --verbose: If set, prints additional information during training
# The script uses scikit-learn for training the model and pickle for saving the model.

import argparse
from utils import load_csv, load_yaml, save_pickle
from sklearn.linear_model import LogisticRegression

def parse_args():
    parser = argparse.ArgumentParser(description="Train a logistic regression model.")
    parser.add_argument('--data', type=str, default='data/train.csv', help='Path to the training data file')
    parser.add_argument('--output', type=str, default='models/model.pkl', help='Path to save the trained model')
    parser.add_argument('--params', type=str, default='params/params.json', help='Path to the hyperparameters file')
    parser.add_argument('--verbose', action='store_true', help='Print additional information during training')
    return parser.parse_args()


def train_model(X, y, params):
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model


def main():
    args = parse_args()
    
    if args.verbose:
        print(f"Loading data from {args.data}...")
    data = load_csv(args.data, dtype={'A': 'float64', 'B': 'float64', 'R': 'int'})
    
    if args.verbose:
        print("Splitting data into features and target variable...")
    X = data.drop(columns=['R'])
    y = data['R']
    
    if args.verbose:
        print(f"Loading hyperparameters from {args.params}...")
    params = load_yaml(args.params)['train_model']
    
    if args.verbose:
        print("Training the model...")
    model = train_model(X, y, params)
    
    if args.verbose:
        print(f"Saving the model to {args.output}...")
    save_pickle(model, args.output)
    
    if args.verbose:
        print("Training complete.")

if __name__ == "__main__":
    main()
