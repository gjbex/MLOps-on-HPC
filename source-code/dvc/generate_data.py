#!/usr/bin/env python3

# Python script to generate synthetic data for training a model.  The
# data consists of two values 'A' and 'B', where 'A' and 'B' are
# random numbers between 0 and 1, uniformly distributed.
# The data further consists of a value 'R' which is either 0 or 1.
# The value of 'R' is in part determined by the values of 'A' and 'B':
# - If A + B < 1, then R = 0
# - If A + B >= 1, then R = 1
# However, to introduce some noise, we randomly flip the value of 'R'
# with a probability that depends on (A + B - 1)^2.  The smaller this value,
# the more likely we are to flip 'R'.  A normal distribution is used with
# standard deviation given as a parameter to the script.
# The script takes the following command line arguments:
# - `--num_samples`: Number of samples to generate (default: 100)
# - `--std_dev`: Standard deviation for the noise in flipping R (default: 0.1)
# - `--seed`: Random seed for reproducibility (default: 42)
# - `--output`: Output file path (default: 'data.csv')

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data for training a model.")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--std_dev', type=float, default=0.1, help='Standard deviation for noise in flipping R')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_target', action='store_true', help='If set, do not generate target variable R')
    parser.add_argument('--output', type=str, default='data.csv', help='Output file path')
    return parser.parse_args()

def generate_data(num_samples, std_dev, seed, no_target=False):
    np.random.seed(seed)
    A = np.random.uniform(0, 1, num_samples)
    B = np.random.uniform(0, 1, num_samples)

    if no_target:
        return A, B, None

    # Calculate R based on A and B
    R = (A + B >= 1).astype(int)

    # Calculate the probability of flipping R based on the distance from 1
    flip_prob = np.clip((A + B - 1) ** 2, 0, 1)
    flip_mask = np.random.normal(0, std_dev, num_samples) < flip_prob

    # Flip R based on the calculated mask
    R[flip_mask] = 1 - R[flip_mask]

    return A, B, R      

def save_data(A, B, R, output_file):
    data = np.column_stack((A, B, R) if R is not None else (A, B))
    np.savetxt(output_file, data, delimiter=',', header='A,B,R' if R is not None else 'A,B', comments='')

def main():
    args = parse_args()
    A, B, R = generate_data(args.num_samples, args.std_dev, args.seed, args.no_target)
    save_data(A, B, R, args.output)
    print(f"Data generated and saved to {args.output}")

if __name__ == "__main__":
    main()
