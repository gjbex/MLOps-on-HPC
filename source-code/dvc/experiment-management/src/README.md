# Source code

Source code for the simulator of Ising models.


## What is it?

1. `model.py`: defines the `IsingSystem` class, which represents the Ising
   model system.
1. `measures.py`: defines the `AbstractMeasure` class, which serves as the
   base class for all measures that can be computed from the Ising model system.
   It implements two derived classes: `Energy` and `Magnetization`.
1. `dynamics.py`: defines the `AbstractStepper` class, as well as two derived
   classes: `MetropolisHastingsStepper` and `GlauberStepper`.
1. `convergence.py`: defines the `AbstractIsConverged` class, which serves as
   the base class for all convergence criteria. It implements a derived class:
   `IsMeasureStable`, which checks if a measure is stable over a number of
   steps in the simulation.
1. `simulation.py`: defines the `Simulation` class, which orchestrates the
   simulation and provides a `run` method to execute the simulation.
1. `ising_simulation.py`: Python script to read the parameters from a
   configuration file and run the simulation.


## How to run it?

Simply run:
```bash
$ python ising_simulation.py params.yaml
```
