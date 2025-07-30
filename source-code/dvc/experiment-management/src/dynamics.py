import abc
import itertools
import numpy as np
from model import IsingSystem


class AbstractStepper(abc.ABC):
    '''Abstract base class for steppers.  Derived classes should
    implement the `update` method.
    '''

    def __init__(self, temperature):
        '''Initializes the stepper.
        
        Parameters
        ----------
        temperature: float
          temperature to use in the dynamics
        '''
        self._temperature = temperature

    @property
    def T(self):
        '''Returns the temperature for the dynamics

        Returns
        -------
        float
          temperature of the dynamics
        '''
        return self._temperature

    @staticmethod
    def _compute_delta_H(ising, i, j):
        '''Computes the energy difference of the Hamiltonian if a spin
        were flipped (without actually flipping it).

        Parameters
        ----------
        ising: IsingSystem
          Ising system to compute the difference for
        i: int
          candiate spin's row index
        j: int
          candiate spin's column index

        Returns
        -------
        float
          difference for the Hamiltonian value if the given spin were flipped
        '''
        return 2*ising[i, j]*(
            ising.J*(
                ising[i - 1, j] + ising[i, j + 1] + ising[i + 1, j] + ising[i, j - 1]
            ) + ising.h
        )

    @abc.abstractmethod
    def update(self, ising, nr_steps=1):
        '''Abstract method that updates the Ising system according to the dynamics
        specified by the derived classes.

        Parameters
        ----------
        ising: IsingSystem
          Ising system to update
        nr_steps: int
          number of update steps to take, defaults to 1
        '''
        ...


class GlauberStepper(AbstractStepper):
    '''Class that implements a stepper for the Glauber dynamics.
    '''

    def __init__(self, temperature, ising):
        '''Initializes the stepper.
        
        Parameters
        ----------
        temperature: float
          temperature to use in the dynamics
        ising: IsingSystem
          Ising system to update
        '''
        super().__init__(temperature)
        self._row_indices = np.arange(0, ising.nr_rows)
        self._col_indices = np.arange(0, ising.nr_cols)

    def update(self, ising, nr_steps=None):
        '''Updates the Ising system according to the Glauber dynamics.

        Parameters
        ----------
        ising: IsingSystem
          Ising system to update
        nr_steps: int
          number of update steps to take, defaults to the number of spins
          in the system
        '''
        if nr_steps is None:
            nr_steps = ising.nr_rows*ising.nr_cols
        for _ in range(nr_steps):
            i = np.random.choice(self._row_indices)
            j = np.random.choice(self._col_indices)
            delta_E = self.__class__._compute_delta_H(ising, i, j)
            if np.random.uniform() < 1.0/(1.0 + np.exp(delta_E/self._temperature)):
                ising[i, j] = -ising[i, j]


class MetropolisHastingsStepper(AbstractStepper):
    '''Class that implements a stepper for the Metropolis-Hastings dynamics.
    '''

    def __init__(self, temperature):
        '''Initializes the stepper.
        
        Parameters
        ----------
        temperature: float
          temperature to use in the dynamics
        '''
        super().__init__(temperature)

    def update(self, ising, nr_steps=None):
        '''Updates the Ising system according to the Metropolis-Hastings dynamics.

        Parameters
        ----------
        ising: IsingSystem
          Ising system to update
        nr_steps: int
          number of update steps to take, defaults to the number of spins
          in the system
        '''
        if nr_steps is None:
            nr_steps = 1
        for _ in range(nr_steps):
            for i, j in itertools.product(range(ising.nr_rows), range(ising.nr_cols)):
                delta_E = self.__class__._compute_delta_H(ising, i, j)
                if delta_E <= 0.0 or np.random.uniform() < np.exp(-delta_E/self._temperature):
                    ising[i, j] = -ising[i, j]
