#!/usr/bin/env python3

import argparse
import dvclive


def parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")
    parser.add_argument("--a", type=float, default=1.0, help="Parameter a")
    parser.add_argument("--t_max", type=int, default=10, help="Maximum number of time steps")
    args = parser.parse_args()
    return args

def function(t, a):
    """A simple function to simulate some behavior."""
    return a * t**2

def main():
    args = parse_args()
    with dvclive.Live(report='html') as live:
        live.log_param("a", args.a)
        live.log_param("t_max", args.t_max)
        for t in range(args.t_max + 1):
            result = function(t, args.a)
            live.log_metric("result", result)
            live.next_step()

if __name__ == "__main__":
    main()
