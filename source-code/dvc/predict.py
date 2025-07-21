#!/usr/bin/env python3

# Python script that predicts the target variable using a trained logistic regression model.
# It takes the following command line arguments:
# --data: Path to the production data file (default: 'data/production.csv')
# --model: Path to the trained model file (default: 'models/model.pkl')
# --output: Path to save the predictions (default: 'predictions/predictions.csv')
# --verbose: If set, prints additional information during prediction
# The prediction  is saved in a CSV file with the same structure as the input data,
# with an additional column 'R' for the predicted values.

import argparse
from utils import load_csv, load_pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Predict using a logistic regression model.")
    parser.add_argument('--data', type=str, default='data/production.csv', help='Path to the production data file')
    parser.add_argument('--model', type=str, default='models/model.pkl', help='Path to the trained model file')
    parser.add_argument('--output', type=str, default='predictions/predictions.csv', help='Path to save the predictions')
    parser.add_argument('--verbose', action='store_true', help='Print additional information during prediction')
    return parser.parse_args()


def predict(model, X):
    return model.predict(X)

def save_predictions(X, predictions, output_path):
    X['R'] = predictions
    X.to_csv(output_path, index=False)

def main():
    args = parse_args()
    
    if args.verbose:
        print(f"Loading data from {args.data}...")
    data = load_csv(args.data, dtype={'A': 'float64', 'B': 'float64'})
    
    if args.verbose:
        print("Loading model from {args.model}...")
    model = load_pickle(args.model)
    
    if args.verbose:
        print("Making predictions...")
    predictions = predict(model, data)
    
    if args.verbose:
        print(f"Saving predictions to {args.output}...")
    save_predictions(data, predictions, args.output)
    
    if args.verbose:
        print("Prediction completed successfully.")

if __name__ == "__main__":
    main()
