#!/usr/bin/env python3

# Python script to preprocess data for training a model using
# a trained preprocessor.
# The script takes the following command line arguments:
# - `--data`: Path to the data file to preprocess (default: 'data/data.csv')
# - `--preprocessor`: Path to the preprocessor file (default: 'preprocessor.pkl')
# - `--output`: Output file path for the preprocessed data (default: 'preprocessed_data.csv')

import argparse
import pandas as pd
from utils import load_csv, load_yaml, load_pickle, save_data

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for training a model using a trained preprocessor.")
    parser.add_argument('--data', type=str, default='data/data.csv',
                        help='Path to the data file to preprocess')
    parser.add_argument('--preprocessor', type=str, default='preprocessor.pkl',
                        help='Path to the preprocessor file')
    parser.add_argument('--params', type=str, default='params.yaml',
                        help='Path to the parameter file')
    parser.add_argument('--output', type=str, default='data/preprocessed/data.csv',
                        help='Output file path for the preprocessed data')
    return parser.parse_args()

def preprocess_data(data, scaler):
    X = data[['A', 'B']]
    X_scaled = scaler.transform(X)
    preprocessed_data = pd.DataFrame(X_scaled, columns=['A', 'B'])
    if 'R' in data.columns:
        preprocessed_data['R'] = data['R'].values
    return preprocessed_data

def main():
    args = parse_args()
    
    data = load_csv(args.data, dtype={'A': 'float64', 'B': 'float64', 'R': 'int'})
    scaler = load_pickle(args.preprocessor)
    params = load_yaml(args.params)

    preprocessed = preprocess_data(data, scaler)
    save_data(preprocessed, args.output)
    print(f"Preprocessed data saved to {args.output}")

if __name__ == "__main__":
    main()
