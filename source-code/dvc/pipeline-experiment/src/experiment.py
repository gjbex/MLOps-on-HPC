#!/usr/bin/env python3

import argparse
import yaml
import dvclive


def parse_args():
    parser = argparse.ArgumentParser(description="Run an experiment with specified parameters.")
    parser.add_argument("--a", type=float, default=1.0, help="Parameter a")
    parser.add_argument("--t_max", type=int, default=10, help="Maximum number of time steps")
    parser.add_argument("--params", type=str, help="Path to parameter file")
    parser.add_argument("--not_live", action="store_true", help="Disable DVC Live reporting")
    args = parser.parse_args()
    return args

def function(t, a):
    """A simple function to simulate some behavior."""
    return a * t**2

class DummyLive:
    """No-op substitute for dvclive.Live when reporting is disabled."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # propagate exceptions
        return False

    def __getattr__(self, name):
        # only noop methods that exist on dvclive.Live; otherwise error
        if hasattr(dvclive.Live, name):
            def _noop(*args, **kwargs):
                pass
            return _noop
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

def main():
    args = parse_args()
    if args.params:
        with open(args.params, 'r') as file:
            params = yaml.safe_load(file)
            args.a = params.get('a', args.a)
            args.t_max = params.get('t_max', args.t_max)

    live_ctx = DummyLive() if args.not_live else dvclive.Live(report='html')

    with live_ctx as live:
        live.log_param("a", args.a)
        live.log_param("t_max", args.t_max)
        with open('result.tsv', 'w') as result_file:
            for t in range(args.t_max + 1):
                result = function(t, args.a)
                print(t, result, file=result_file)
                live.log_metric("result", result)
                live.next_step()

if __name__ == "__main__":
    main()
