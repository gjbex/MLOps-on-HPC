import numpy as np


class IsingSystem:
    '''Class to represent 2-dimensional Ising systems of `nr_rows` by `nr_cols`
    spins.  The interaction between the spins is characterized by `J`, that of
    the spins with an external magnetic field by `h`.
    
    A seed initializes the random number generator for reproducibility.
    '''

    def __init__(self, *, nr_rows, nr_cols, J, h, seed):
        '''Initializes an Ising spin system.
        
        Parameters
        ----------
        nr_rows: int
          number of spin rows
        nr_cols: int
          number of spin columns
        J: float
          strength of the interaction between neighboring spins
        h: float
          strength of the interaction between a spin and an external magnetic field
        seed: int
          seed for the random number generator that initializes the spin values
        '''
        np.random.seed(seed)
        self._spins = np.random.choice([-1, 1], size=(nr_rows, nr_cols))
        self._J = J
        self._h = h
        self._seed = seed

    @property
    def nr_rows(self):
        '''Returns the number of spin rows in the system.
        
        Returns
        -------
        int
          number of spin rows in the system
        '''
        return self._spins.shape[0]

    @property
    def nr_cols(self):
        '''Returns the number of spin columns in the system.
        
        Returns
        -------
        int
          number of spin columns in the system
        '''
        return self._spins.shape[1]

    @property
    def N(self):
        '''Returns the number of spins in the Ising system, i.e., the number of
        rows times the number of columns.
        
        Returns
        -------
        int
          number of spins in the system
        '''
        return self.nr_rows*self.nr_cols

    def __getitem__(self, item):
        '''Accessor to get a spin value.
        
        Parameters
        ----------
        i: int
          row index of the spin
        j: int
          column index of the spin
          
        Returns
        -------
        int:
          value of the spin at row i, column j
        '''
        return self._spins[item[0] % self.nr_rows, item[1] % self.nr_cols]

    def __setitem__(self, item, value):
        '''Accessor to set a spin value.
        
        Parameters
        ----------
        i: int
          row index of the spin
        j: int
          column index of the spin
        value: int
          value to set the spin to
        '''
        self._spins[item[0] % self.nr_rows, item[1] % self.nr_cols] = value

    @property
    def J(self):
        '''Returns `J`, the strength of the interaction between neighboring spins.
        
        Returns
        -------
        float
          strength of the interaction between neighboring spins
        '''
        return self._J

    @property
    def h(self):
        '''Returns `J`, the strength of the interaction between a spin and an
        external magnetic field
        
        Returns
        -------
        float
          strength of the interaction between a spin and an external magnetic
          field
        '''
        return self._h

    def __repr__(self):
        '''Returns a representation of an Ising system that allows to recreate
        its original state for reproducibility.

        Returns
        -------
        str
          string representation of the initial state of the Ising system

        Note
        ----
        This representation *can not* be used for checkpointing/serialization.
        '''
        return f"""{{
    'nr_rows': {self.nr_rows},
    'nr_cols': {self.nr_cols},
    'J': {self.J},
    'h': {self.h},
    'seed': {self._seed},
}}"""

    def __str__(self):
        '''Returns a human readable representation of an Ising system for debugging
        purposes.  Note that -1 values are rendered as 0 to improve visual layout.

        Returns
        -------
        str
          string representation of the initial state of the Ising system

        Note
        ----
        This representation *should not* be used for checkpointing/serialization as
        -1 values are rendered as 0.
        '''
        return '\n'.join(
            (''.join('1' if s > 0 else '0' for s in self._spins[i, :])
             for i in range(self.nr_rows))
        )
