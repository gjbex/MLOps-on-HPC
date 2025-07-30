import abc
import collections
import copy
import itertools


class AbstractMeasure(abc.ABC):
    '''abstract class that represents a measure of the system such as the
    magnetization or the energy.  It is however more general and can be used
    for non-scalar measures as well.
    '''

    def __init__(self, name, /, *, headers=None):
        '''Base initialization method

        Parameters
        ----------
        name: str
          the measure's name
        headers: list[str] | tuple[str] | None
          the headers for the measure, if `None`, the measure is assumed to be scalar
          and the name is the sole header
        '''
        self._name = name
        self._headers = (self._name, ) if headers is None else tuple(headers)
        self._sep = ' '
        self._values = []

    @property
    def name(self):
        '''Returs the measure's name

        Returns
        -------
        str
          the measure's name
        '''
        return self._name

    @property
    def headers(self):
        '''Returns a string representation of the headers for the measure, separated
        by the `sep` value passed on initialization.

        Returns
        -------
        str
            column headers for this measure
        '''
        return self._sep.join(self._headers)

    @property
    def sep(self):
        '''Returns the separator for textual output.
        
        Returns
        -------
        str
          separator for output
        '''

    @sep.setter
    def sep(self, value):
        '''Sets the separator to use for output.
        
        Parameters
        ----------
        sep: str
          separator to use for output
        '''
        self._sep = value

    @property
    def values(self):
        '''Returns the accumulated values of this measure, i.e., all values measured
        during the lifetime of the measure up to the call of this method.

        Returns
        -------
        list
          values measured up to now (note: a deep copy is returned)
        '''
        return copy.deepcopy(self._values)

    def __len__(self):
        '''Returns the number of values measured so far.
        
        Returns
        -------
        int
          number of values measured so far
        '''
        return len(self._values)

    @property
    def current_value(self):
        '''Returns a string representation of the most recently measured value, if non-scalar,
        components are separated by the `sep` value passed during initialization.

        Returns
        -------
        tuple
          the most recent value that was measured
        '''
        value = self._values[-1]
        if isinstance(value, collections.abc.Iterable):
            return tuple(value)
        else:
            return (value, )

    @abc.abstractmethod
    def compute_value(self, system):
        '''Abstract method that has to be implemented to compute the specific measure
        that is derived from this class.

        Parameters
        ----------
        system: Any
          system to compute the measure on

        Returns
        -------
        Any
          the value the measure computes
        '''
        ...

    def __call__(self, system):
        '''Computes and stores the value of this measure.  This makes the objects
        callable, so a measure `A` on a system `s` can be computed as `A(s)`.

        Parameters
        ----------
        system: Any
          system to compute the measure on

        Returns
        -------
        Any
          the value the measure computes
        '''
        value = self.compute_value(system)
        self._values.append(value)
        return value


class Magnetization(AbstractMeasure):
    '''Computes the magnetization of an Ising system.
    '''

    def __init__(self):
        '''Initializes the measure.
        '''
        super().__init__('magnetization')

    def compute_value(self, ising):
        '''Computes the value of the magnetization for the given Ising system.

        Parameters
        ----------
        ising: IsingSystem
          instance of the `IsingSystem` class

        Returns
        -------
        float
          magnetization of the given Ising system
        '''
        magnetization = 0.0
        for i, j in itertools.product(range(ising.nr_rows), range(ising.nr_cols)):
            magnetization += ising[i, j]
        return magnetization/ising.N


class Energy(AbstractMeasure):
    '''Class to compute the energy of an Ising system.
    '''

    def __init__(self):
        '''Initializes the measure.
        '''
        super().__init__('energy')

    def compute_value(self, ising):
        '''Computes the value of the energy for the given Ising system.

        Parameters
        ----------
        ising: IsingSystem
          instance of the `IsingSystem` class

        Returns
        -------
        float
          energy of the given Ising system
        '''
        J, h = ising.J, ising.h
        energy = 0.0
        for i, j in itertools.product(range(ising.nr_rows), range(ising.nr_cols)):
            energy -= J*ising[i, j]*(ising[i, j + 1] + ising[i + 1, j]) + h*ising[i, j]
        return energy/ising.N
