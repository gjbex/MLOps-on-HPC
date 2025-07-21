#!/usr/bin/env python3

# Python script to split data into training and testing sets.
# # The script takes the following command line arguments:
# - `--data`: Path to the data file to split (default: 'data/data.csv')
# - `--output`: Output directory for the split data (default: 'split_data/')

import argparse
from utils import load_csv, load_yaml, split_train_test, save_train_test

def parse_args():
    parser = argparse.ArgumentParser(description="Split data into training and testing sets.")
    parser.add_argument('--data', type=str, default='data/data.csv', help='Path to the data file to split')
    parser.add_argument('--params', type=str, default='params.yaml', help='Path to the parameter file')
    parser.add_argument('--output', type=str, default='split_data/', help='Output directory for the split data')
    return parser.parse_args()


def main():
    args = parse_args()
    
    data = load_csv(args.data, dtype={'A': 'float64', 'B': 'float64', 'R': 'int'})
    params = load_yaml(args.params)

    train_data, test_data = split_train_test(
        data,
        params['split_data']['test_size'],
        params['split_data']['random_state']
    )
    save_train_test(train_data, test_data, args.output)

if __name__ == "__main__":
    main()
