#!/usr/bin/env python3

# Python script to compute metrics for a logistic regression model
# based on a given dataset.  The results are saved to a JSON file.
# It takes the following command line arguments:
# --data: Path to the test data file (default: 'data/test.csv')
# --model: Path to the trained model file (default: 'models/model.pkl')
# --output: Path to save the computed metrics (default: 'metrics/metrics.yaml')
# --verbose: If set, prints additional information during computation


import argparse
import pathlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import load_csv, load_pickle
import yaml

def parse_args():   
    parser = argparse.ArgumentParser(description="Compute metrics for a logistic regression model.")
    parser.add_argument('--data', type=str, default='data/test.csv',
                        help='Path to the test data file')
    parser.add_argument('--model', type=str, default='models/model.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--output', type=str, default='metrics/metrics.yaml',
                        help='Path to save the computed metrics')
    parser.add_argument('--verbose', action='store_true',
                        help='Print additional information during computation')
    return parser.parse_args()


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0)
    }
    return metrics

def save_metrics(metrics, output_path):
    """Saves the computed metrics to a YAML file."""
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as file:
        yaml.dump(metrics, file, indent=4)

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
        print(f"Loading model from {args.model}...")
    model = load_pickle(args.model)
    
    if args.verbose:
        print("Computing metrics...")
    metrics = compute_metrics(model, X, y)
    
    if args.verbose:
        print(f"Saving metrics to {args.output}...")
    save_metrics(metrics, args.output)
    
    if args.verbose:
        print("Metrics computation completed successfully.")

if __name__ == "__main__":
    main()
