#!/usr/bin/env python3

# Python script to train the preprocessor for the model.
# The input for the data should be scaled to [0, 1] range.
# This is done using MinMaxScaler from scikit-learn.
# The script takes the following command line arguments:
# - `--data`: Path to the training data file (default: 'data/data.csv')
# - `--output`: Output file path for the preprocessor (default: 'models/preprocessor.pkl')

import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Train a preprocessor for the model.")
    parser.add_argument('--data', type=str, default='data/data.csv', help='Path to the training data file')
    parser.add_argument('--output', type=str, default='models/preprocessor.pkl', help='Output file path for the preprocessor')
    return parser.parse_args()

def load_data(data_path):
    return pd.read_csv(data_path, dtype={
        'A': 'float64',
        'B': 'float64',
        'R': 'int'
    })

def train_preprocessor(data):
    scaler = MinMaxScaler()
    X = data[['A', 'B']]
    scaler.fit(X)
    return scaler

def save_preprocessor(scaler, output_path):
    with open(output_path, 'wb') as file:
        pickle.dump(scaler, file)

def main():
    args = parse_args()
    
    # Load the data
    data = load_data(args.data)
    
    # Train the preprocessor
    scaler = train_preprocessor(data)
    
    # Save the preprocessor
    save_preprocessor(scaler, args.output)
    
    print(f"Preprocessor trained and saved to {args.output}")

if __name__ == "__main__":
    main()
