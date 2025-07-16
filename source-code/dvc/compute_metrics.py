#!/usr/bin/env python3

# Python script to compute metrics for a logistic regression model
# based on a given dataset.  The results are saved to a JSON file.
# It takes the following command line arguments:
# --data: Path to the test data file (default: 'data/test.csv')
# --model: Path to the trained model file (default: 'models/model.pkl')
# --output: Path to save the computed metrics (default: 'metrics/metrics.json')
# --verbose: If set, prints additional information during computation


import argparse
import json
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_args():   
    parser = argparse.ArgumentParser(description="Compute metrics for a logistic regression model.")
    parser.add_argument('--data', type=str, default='data/test.csv', help='Path to the test data file')
    parser.add_argument('--model', type=str, default='models/model.pkl', help='Path to the trained model file')
    parser.add_argument('--output', type=str, default='metrics/metrics.json', help='Path to save the computed metrics')
    parser.add_argument('--verbose', action='store_true', help='Print additional information during computation')
    return parser.parse_args()

def load_data(data_path):
    return pd.read_csv(data_path, dtype={
        'A': 'float64',
        'B': 'float64',
        'R': 'int'
    })

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

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
    with open(output_path, 'w') as file:
        json.dump(metrics, file, indent=4)

def main():
    args = parse_args()
    
    if args.verbose:
        print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    
    if args.verbose:
        print("Splitting data into features and target variable...")
    X = data.drop(columns=['R'])
    y = data['R']
    
    if args.verbose:
        print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
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
