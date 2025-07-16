#!/usr/bin/env python3

# Python script to preprocess data for training a model using
# a trained preprocessor.
# The script takes the following command line arguments:
# - `--data`: Path to the data file to preprocess (default: 'data/data.csv')
# - `--preprocessor`: Path to the preprocessor file (default: 'models/preprocessor.pkl')
# - `--output`: Output file path for the preprocessed data (default: 'preprocessed/preprocessed_data.csv')

import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for training a model using a trained preprocessor.")
    parser.add_argument('--data', type=str, default='data/data.csv', help='Path to the data file to preprocess')
    parser.add_argument('--preprocessor', type=str, default='models/preprocessor.pkl', help='Path to the preprocessor file')
    parser.add_argument('--output', type=str, default='preprocessed/preprocessed_data.csv', help='Output file path for the preprocessed data')
    return parser.parse_args()

def load_data(data_path):
    return pd.read_csv(data_path, dtype={
        'A': 'float64',
        'B': 'float64',
        'R': 'int'
    })

def load_preprocessor(preprocessor_path):
    with open(preprocessor_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def preprocess_data(data, scaler):
    X = data[['A', 'B']]
    X_scaled = scaler.transform(X)
    preprocessed_data = pd.DataFrame(X_scaled, columns=['A', 'B'])
    if 'R' in data.columns:
        preprocessed_data['R'] = data['R'].values
    return preprocessed_data

def save_preprocessed_data(data, output_path):
    data.to_csv(output_path, index=False)

def main():
    args = parse_args()
    
    # Load the data
    data = load_data(args.data)
    
    # Load the preprocessor
    scaler = load_preprocessor(args.preprocessor)
    
    # Preprocess the data
    preprocessed_data = preprocess_data(data, scaler)
    
    # Save the preprocessed data
    save_preprocessed_data(preprocessed_data, args.output)
    
    print(f"Preprocessed data saved to {args.output}")

if __name__ == "__main__":
    main()
