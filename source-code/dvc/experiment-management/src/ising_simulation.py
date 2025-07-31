#!/usr/bin/env python3

import argparse
from dvclive import Live
import pydantic
import sys
import yaml
from convergence import IsMeasureStable
from dynamics import GlauberStepper, MetropolisHastingsStepper
from model import IsingSystem
from measures import Magnetization, Energy
from parameters import load_params, log_params
from simulation import Simulation


def parse_args():
    '''Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
      parsed command line arguments
    '''
    parser = argparse.ArgumentParser(description='Run an Ising simulation.')
    parser.add_argument('params', type=str, help='YAML file with simulation parameters')
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        params = load_params(args.params)
    except FileNotFoundError:
        print(f"Error: The file '{args.params}' was not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the YAML file '{args.params}'. {e}")
        sys.exit(1)
    except pydantic.ValidationError as e:
        print('Error: Invalid parameters')
        print(f'{e}')
        sys.exit(1)

    with Live(report='html') as live:
        # Log parameters
        log_params(live, params)

        # Initialize Ising system
        ising = IsingSystem(
            nr_rows=params.ising.nr_rows,
            nr_cols=params.ising.nr_cols,
            J=params.ising.J,
            h=params.ising.h,
            seed=params.ising.seed,
        )

        # Initialize dynamics stepper
        if params.dynamics.type == 'glauber':
            stepper = GlauberStepper(temperature=params.dynamics.temperature, ising=ising)
        elif params.dynamics.type == 'metropolis_hastings':
            stepper = MetropolisHastingsStepper(temperature=params.dynamics.temperature)
        else:
            raise ValueError(f"Unknown dynamics type: {params.dynamics.type}")

        # Initialize convergence criterion
        measures = {
            'magnetization': Magnetization(),
            'energy': Energy(),
        }
        if params.convergence.measure not in measures:
            raise ValueError(f"Unknown convergence type: {params.convergence.measure}")
        is_converged = IsMeasureStable(
            measure=measures[params.convergence.measure],
            nr_measurement_steps=params.convergence.nr_measurement_steps,
            delta=params.convergence.delta
        )
        # Remove the measure used for the convergence check from the measures since
        # it will be added automatically to the simulation
        del measures[params.convergence.measure]

        # Create simulation instance
        simulation = Simulation(ising=ising, stepper=stepper, is_converged=is_converged)

        # Add measures
        for measure in measures.values():
            simulation.add_measures(measure)

        # Run the simulation
        simulation.run(
            max_steps=params.simulation.max_steps,
            measure_interval=params.simulation.measure_interval,
            live=live,
        )
        live.make_report()


if __name__ == '__main__':
    main()
