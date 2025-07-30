import abc
import numpy as np
from measures import AbstractMeasure


class AbstractIsConverged(abc.ABC):

    def __init__(self, measure):
        self._measure = measure

    @property
    def measure(self):
        '''Returns the measure used in the convergence criterion.
        
        Returns
        -------
        AbstractMeasure
          measure used in this convergence criterion
          
        Note
        ----
        The measure returned is *not* a copy, it is the actual object.
        '''
        return self._measure

    @abc.abstractmethod
    def is_converged(self):
        '''Returns `True` if the simulation has converged, `False` otherwise, should
        be implemented by derived classes.
        
        Returns
        -------
        bool
          `True` if the simulation has converged, `False` otherwise
        '''
        ...

    def __call__(self):
        '''Returns `True` if the simulation has converged, `False` otherwise.
        
        Returns
        -------
        bool
          `True` if the simulation has converged, `False` otherwise
        '''
        return self.is_converged()


class IsMeasureStable(AbstractIsConverged):
    '''Convergence criterion that will stop the simulation if the measure is
    constant to within an absolute error for a given number of steps.'''

    def __init__(self, *, measure, nr_measurement_steps, delta):
        '''Initialize the criterion.
        
        Parameters
        ----------
        measure: AbstractMeasure
          measure that is used in the simulation
        nr_measurement_steps: int
          number of measurement steps for which the measure should be constant
        delta: float
          absolute error to consider the measure to be constant within

        Note
        ----
        This class is only designed to work for scalar measures.
        '''
        self._measure = measure
        self._nr_measurement_steps = nr_measurement_steps
        self._delta = delta

    def is_converged(self):
        '''Returns `True` if the measure remained approximately constant, `False`
        otherwise.

        Returns
        -------
        bool
          `True` if the measure was approximately constant for the specified number
          of measurement steps, `False` otherwise
        '''
        if len(self._measure) < self._nr_measurement_steps:
            return False
        values = self._measure.values[-self._nr_measurement_steps:]
        mean = np.mean(values)
        return max(np.abs(value - mean) for value in values) < self._delta
