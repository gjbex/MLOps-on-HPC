#!/usr/bin/env python3

import argparse
from convergence import IsMeasureStable
from dynamics import GlauberStepper, MetropolisHastingsStepper
from ising_system import IsingSystem
from measures import Magnetization, Energy
from simulation import Simulation
import yaml


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

def load_configuration(filename):
    '''Load simulation parameters from a YAML file.
    
    Parameters
    ----------
    filename: str
      path to the YAML file with simulation parameters
    
    Returns
    -------
    dict
      dictionary with simulation parameters
    '''
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def main():
    args = parse_args()
    config = load_configuration(args.params)

    # Initialize Ising system
    ising = IsingSystem(
        nr_rows=config['ising']['nr_rows'],
        nr_cols=config['ising']['nr_cols'],
        J=config['ising']['J'],
        h=config['ising']['h'],
        seed=config['ising']['seed'],
    )

    # Initialize dynamics stepper
    if config['dynamics']['type'] == 'glauber':
        stepper = GlauberStepper(temperature=config['dynamics']['temperature'], ising=ising)
    elif config['dynamics']['type'] == 'metropolis_hastings':
        stepper = MetropolisHastingsStepper(temperature=config['dynamics']['temperature'])
    else:
        raise ValueError(f"Unknown dynamics type: {config['dynamics']['type']}")

    # Initialize convergence criterion
    measures = {
        'magnetization': Magnetization(),
        'energy': Energy(),
    }
    if config['convergence']['measure'] not in measures:
        raise ValueError(f"Unknown convergence type: {config['convergence']['measure']}")
    is_converged = IsMeasureStable(
        measure=measures[config['convergence']['measure']],
        nr_measurement_steps=config['convergence']['nr_measurement_steps'],
        delta=config['convergence']['delta']
    )
    # Remove the measure used for the convergence check from the measures since
    # it will be added automatically to the simulation
    del measures[config['convergence']['measure']]

    # Create simulation instance
    simulation = Simulation(ising=ising, stepper=stepper, is_converged=is_converged)

    # Add measures
    for measure in measures.values():
        simulation.add_measures(measure)

    # Run the simulation
    simulation.run(
        max_steps=config['simulation']['max_steps'],
        measure_interval=config['simulation']['measure_interval']
    )


if __name__ == '__main__':
    main()
