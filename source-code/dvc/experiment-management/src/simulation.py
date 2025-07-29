from convergence import AbstractIsConverged
import copy
from dynamics import AbstractStepper
from model import IsingSystem
from measures import AbstractMeasure
import numpy as np


class Simulation:
    '''Class to run simulations with a given initial Ising system, dynamics
    and stop criterion.
    '''

    def __init__(self, *, ising, stepper, is_converged, sep=' '):
        '''Initializes the simulation.

        Parameters
        ----------
        ising: IsingSystem
          instance of an initialized Ising system
        stepper: AbstractStepper
          stepper implementation to update the Ising system
        is_converged: Callable
          callable that returns `True` when the dynamics has converged, `False`
          ottherwise
        sep: str
          separator to use for output, defaults to ' '
        '''
        self._ising = ising
        self._stepper = stepper
        self._is_converged = is_converged
        self._sep = sep
        self._measures = []
        self._measure_steps = []
        self.add_measures(self._is_converged.measure)

    def add_measures(self, *measures):
        '''Add measures to the simulation.
        
        Parameters
        ----------
        measures: *AbstractMeasure
          one or more measures to add to the simulation
        '''
        # ensure the separator is propagated to each measure, this only
        # matters for non-scalar measures
        for measure in measures:
            measure.sep = self._sep
        self._measures.extend(measures)

    def _compute_measures(self, step_nr):
        '''Computes the measures for the simulation.

        Parameters
        ----------
        step_nr: int
          current step number
        '''
        self._measure_steps.append(step_nr)
        values = [str(step_nr)]
        for measure in self._measures:
            measure(self._ising)
            values.append(measure.current_value)
        print(self._sep.join(value for value in values))

    @property
    def measures(self):
        '''Returns an iterable over the measures of the simulation.  The
        actual values are deep copies of the original measures.

        Returns
        -------
        Iterable[AbstractMeasure]
          iterable to deep copies of the measures
        '''
        return (copy.deepcopy(measure) for measure in self._measures)

    @property
    def measure_steps(self):
        '''Returns the step numbers at which measurements where computed during the run
        of the simulation.

        Returns
        -------
        list[int]
          deep copy of the list of steps at which measures were computed
        '''
        return copy.deepcopy(self._measure_steps)

    def run(self, *, max_steps, measure_interval=1):
        ''' Simulates to convergence, or a maximum number of steps.

        Parameters
        ----------
        max_steps: int
          maximum number of simulation steps to perform
        measure_interval: int
          number of steps between the computation and display of measurements
        '''
        print('step' + self._sep + self._sep.join(measure.headers for measure in self._measures))    
        for step_nr in range(max_steps + 1):
            if step_nr % measure_interval == 0:
                self._compute_measures(step_nr)
                if self._is_converged():
                    break
            self._stepper.update(self._ising)
        else:
            self._compute_measures(step_nr)

