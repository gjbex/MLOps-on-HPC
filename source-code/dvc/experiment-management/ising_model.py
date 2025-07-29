#!/usr/bin/env python
# coding: utf-8

# # Requirements

# In[1]:


import abc
import collections
import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np


# # Ising system

# An Ising system is defined as an $N$-dimensional grid of spins.  Each spin interacts with its nearest neighboours, and is subject to an external magnetic field.  Here, you will only consider a 2-dimensional system that is defined as $s_{kl}$ for $0 \le k \lt K$ and $0 \le l \lt L$, so $K \cdot L$ spins in total.  Conventially, spins are denoted by $s_i$ for $0 \le i < N = K \cdot L$.
# 
# The Hamiltonian of the system is given by
# $$
#     H = -J \sum_{\langle i, j \rangle} s_i s_j - h \sum_{j=0}^{N-1} s_j
# $$
# The sum index $\langle i, j \rangle$ denotes the sum over pairs nearest neighbor spins, where each pair is counted only once.  Periodic boundary conditions are applied, so, e.g., the downward neighbor of a spin on the last row will be the corresponding spin on the upper row, and similar for the other edges.
# 
# This Hamiltonian will be used to define the system's dynamics (see later).

# The implementation of a class to represent Ising systems is straightforward.  Note that this implementation is *not* efficient.

# In[2]:


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


# Note that since the `__init__` method uses the seed provided as an argument to seed the numpy random number generator, and the representation provides that string, an instance of an Ising system can always be reproduced from its representation, without having to contain the individual spins at all.  Incidentally, initializing the system randomly may lead to slow convergence near the critical temperature.
# 
# Also note that to make the string representation visually more pleasing, the $-1$ spins are represented as $0$.
# 
# The `IsingSystem` class is essentially a wrapper for a numpy array.  This allows to later optimize the implementation without having to modify the API.

# You can test the class by
# 
#   * creating an instance,
#   * checking its representation,
#   * checking its string representation,
#   * get the value of a spin,
#   * set the value of a sping, and
#   * show the string representation again.

# In[3]:


ising = IsingSystem(nr_rows=10, nr_cols=10, J=1.0, h=1.0, seed=1234)


# In[4]:


ising


# In[5]:


print(ising)


# In[6]:


ising[1, 4]


# In[7]:


ising [1, 4] = -1


# In[8]:


print(ising)


# # Measures

# Several measures can be computed to characterize an Ising system, e.g.,
# 
#   * the magnetization,
#   * the energy.

# To test the measures you define, it is a good idea to use a very small Ising system, so that it is easy to check the results by hand.

# In[9]:


mini_ising = IsingSystem(nr_rows=4, nr_cols=4, J=0.5, h=2.0, seed=1234)


# Since many measures share implementation details, it is convenient to define an abstract base class `Measure` that encapsulates the common functionality.

# In[10]:


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
        if headers is None:
            self._headers = (self._name, )
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
        str
          string representation of the most recent value that was measured
        '''
        value = self._values[-1]
        if isinstance(value, collections.abc.Iterable):
            return self._sep.join(str(x) for x in value)
        else:
            return str(value)

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


# ## Magnetization

# The magnetization of the system is defined by
# $$
#     M = \frac{\sum_{i=0}^{N-1} s_i}{N}
# $$

# The concrete class `Magnetization` now simply has to implement the `compute_value` method.

# In[11]:


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


# The implementation can now be tested using `mini_ising`.

# In[12]:


M = Magnetization()


# In[13]:


M(mini_ising)


# In[14]:


print(mini_ising)


# The system has 9 spins up ($s_i = 1$) and 7 spins down ($s_i = -1$) while $N = 16$, so the magnetization should be $(9 - 7)/16 = 1/8 = 0.125$ as computed.

# ## Energy

# The energy of the system is defined as
# $$
#   E = H/N
# $$

# The implementation of the `Energy` class is very similar.

# In[15]:


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


# In[16]:


E = Energy()


# In[17]:


E(mini_ising)


# In[18]:


print(mini_ising)


# In[19]:


repr(mini_ising)


# Again, this checks out, so the implementation could be correct.

# # Dynamics

# There are several ways to define the dymamics of an Ising system, and you can consider
# 
#   * Glauber dynamics and
#   * Metropolis-Hastings dynamics.
# 
# Both are based on the same Hamiltonian, but differ in the selection of spins to update and transition probabilities.

# Since both methods share the same Hamiltonian, the energy difference between the original state and a new state where a single spin $i$ is split can be computed by the same function.
# $$
#     \Delta E_{i} = 2 s_i \left( J \sum_{\langle i j \rangle} s_j + h \right)
# $$
# Again, the sum index $\langle i, j \rangle$ runs over *all* neirest neighbors.
# 
# The dynamics in both cases is considered at an absolute temperature $T$.

# In[20]:


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
    def _compute_ΔH(ising, i, j):
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

    @abc.abstractclassmethod
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


# ## Glauber dynamics

# A step in the Glauber dynamics is defined as follows:
# 
#   1. pick a spin at random,
#   2. compute the energy difference $\Delta E$ when it would be flipped,
#   3. with probability $\frac{1}{1 + e^{\Delta E/T}}$, flip the spin.

# In[21]:


class GlauberStepper(AbstractStepper):
    '''Class that implements a stepper for the Glauber dynamics.
    '''

    def __init__(self, temperature):
        '''Initializes the stepper.
        
        Parameters
        ----------
        temperature: float
          temperature to use in the dynamics
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
            ΔE = self.__class__._compute_ΔH(ising, i, j)
            if np.random.uniform() < 1.0/(1.0 + np.exp(ΔE/self._temperature)):
                ising[i, j] = -ising[i, j]


# The process is repeated for as many steps to reach convergence, i.e., thermal equilibrium.  Note that the default number of steps is $N$, the number of spins.  Choosing this default makes it easier to compare the Metropolis-Hastings algorithm that (potentially) updates all spins in a single step.

# ## Metropolis-Hasting dynamics

# A step in the Metropolos-Hastings dynamics is defined as follows:
# 
# For each spin (in order):
#   1. compute the energy difference $\Delta E$ when it would be flipped,
#   1. with probability $e^{-\Delta E/T}$, flip the spin.

# In[22]:


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
                ΔE = self.__class__._compute_ΔH(ising, i, j)
                if ΔE <= 0.0 or np.random.uniform() < np.exp(-ΔE/self._temperature):
                    ising[i, j] = -ising[i, j]


# # Convergence criterion

# Many convergence criterions can be considered, for instance, the magnetization doesn't change for the last $n$ steps.

# In[88]:


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


# In[89]:


class IsMeasureStable(AbstractIsConverged):
    '''Convergence criterion that will stop the simulation if the meaure is
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


# # Simulation

# A simulation consists of
# 
#   1. initializing an Ising system,
#   2. creating a stepper for the desired dynamics and temperature $T$,
#   3. updating the system for multiple time steps while printing various measures,
#   4. stop when either a convergence criterion is met, or a maximum number of steps is reached.

# Again, it is convenient to create a class to encapsulate the scaffolding.

# In[90]:


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
            self._stepper.update(ising)
        else:
            self._compute_measures(step_nr)


# Note that the measure used in the convergence criterion is added automatically.

# # Simulation run

# ## Glauber dynamics

# ### Ferromagnetic phase

# First, you can run for $T < T_c$, i.e., the system should be ferromagnetic.  At equilibrium, the magnetization should be very close to 1 if $h > 0$ or -1 if $h < 0$.

# First you can set up a system of $100 \times 100$ spins with $J = 1$ and $h = 1$.

# In[91]:


ising = IsingSystem(nr_rows=100, nr_cols=100, J=1.0, h=1.0, seed=1234)


# Since you want to use Glauber dynamics, you can create such a stepper with temperature $T = 2$, below the critical temperature $T \approx 2.27$.

# In[92]:


stepper = GlauberStepper(temperature=2.0)


# The convergence criterion is that the magnetization remains constant to within $\delta = 0.001$ for 5 measurement steps.

# In[93]:


is_converged = IsMeasureStable(
    measure=Magnetization(),
    nr_measurement_steps=5,
    delta=0.001,
)


# Now you can create the simulation based on the Ising system, the dynamics, i.e., the stepper and the convergence criterion.  After that, you can add any additional measures you like.

# In[ ]:


simulation = Simulation(
    ising=ising,
    stepper=stepper,
    is_converged=is_converged
)
simulation.add_measures(Energy())


# Now you can run the simulation for at most 500 steps, computing the measures every 10th step.

# In[96]:


simulation.run(max_steps=500, measure_interval=10)


# Since the temperature $T < T_{\mathrm{crit.}}$, the system is in the ferromagnetic phase.

# ### Paramagnetic phase

# Next, you can do a run for $T > Tc$, i.e., the system should be paramagnetic.  At equilibrium, the magnetization should be significantly different from 1 or -1.  For $N \to \infty$, it should be zero.

# First you can set up a system of $100 \times 100$ spins with $J = 1$ and $h = 1$.

# In[108]:


ising = IsingSystem(nr_rows=100, nr_cols=100, J=1.0, h=1.0, seed=1234)


# Since you want to use Glauber dynamics, you can create such a stepper with temperature $T = 2.5$, above the critical temperature $T_c \approx 2.27$.

# In[109]:


stepper = GlauberStepper(temperature=5.0)


# The convergence criterion is that the magnetization remains constant to within $\delta = 0.001$ for 5 measurement steps.

# In[110]:


is_converged = IsMeasureStable(
    measure=Magnetization(),
    nr_measurement_steps=5,
    delta=0.001,
)


# Now you can create the simulation based on the Ising system, the dynamics, i.e., the stepper and the convergence criterion.  After that, you can add any additional measures you like.

# In[111]:


simulation = Simulation(
    ising=ising,
    stepper=stepper,
    is_converged=is_converged
)
simulation.add_measures(Energy())


# Now you can run the simulation for at most 500 steps, computing the measures every 10th step.

# In[112]:


simulation.run(max_steps=500, measure_interval=10)


# Note that the for this temperature, variations in the magnetization are higher, and don't converge to within $\delta = 0.001$.  However, it is clear that $M \approx 0.48 < 1$.

# ## Metropolis-Hastings dynamics

# You can redo the same simulation, but now with the Metropolis-Hastings dynamics.

# ### Ferromagnetic phase

# First, you can run for $T < T_c$, i.e., the system should be ferromagnetic.  At equilibrium, the magnetization should be very close to 1 if $h > 0$ or -1 if $h < 0$.

# First you can set up a system of $100 \times 100$ spins with $J = 1$ and $h = 1$.

# In[113]:


ising = IsingSystem(nr_rows=100, nr_cols=100, J=1.0, h=1.0, seed=1234)


# Since you want to use Glauber dynamics, you can create such a stepper with temperature $T = 2$, below the critical temperature $T \approx 2.27$.

# In[114]:


stepper = MetropolisHastingsStepper(temperature=2.0)


# The convergence criterion is that the magnetization remains constant to within $\delta = 0.001$ for 5 measurement steps.

# In[115]:


is_converged = IsMeasureStable(
    measure=Magnetization(),
    nr_measurement_steps=5,
    delta=0.001,
)


# Now you can create the simulation based on the Ising system, the dynamics, i.e., the stepper and the convergence criterion.  After that, you can add any additional measures you like.

# In[116]:


simulation = Simulation(
    ising=ising,
    stepper=stepper,
    is_converged=is_converged
)
simulation.add_measures(Energy())


# Now you can run the simulation for at most 500 steps, computing the measures every 10th step.

# In[117]:


simulation.run(max_steps=500, measure_interval=10)


# The Metropolis-Hastings dynamics shows more variation between measurement steps, and doesn't converge to within $\delta = 0.001$ withing 500 steps.  It is however also clear that the time taken by an update by the stpper is less than for Glauber dynamics.
# 
# Although the failure to converge seems to be a drawback at first glance, it will also help to escape local minima, so for large spin systems, the accuracy should be better when compared to analytic results.

# ### Paramagnetic phase

# Next, you can do a run for $T > Tc$, i.e., the system should be paramagnetic.  At equilibrium, the magnetization should be significantly different from 1 or -1.  For $N \to \infty$, it should be zero.

# First you can set up a system of $100 \times 100$ spins with $J = 1$ and $h = 1$.

# In[118]:


ising = IsingSystem(nr_rows=100, nr_cols=100, J=1.0, h=1.0, seed=1234)


# Since you want to use Glauber dynamics, you can create such a stepper with temperature $T = 2.5$, above the critical temperature $T_c \approx 2.27$.

# In[119]:


stepper = MetropolisHastingsStepper(temperature=5.0)


# The convergence criterion is that the magnetization remains constant to within $\delta = 0.001$ for 5 measurement steps.

# In[120]:


is_converged = IsMeasureStable(
    measure=Magnetization(),
    nr_measurement_steps=5,
    delta=0.001,
)


# Now you can create the simulation based on the Ising system, the dynamics, i.e., the stepper and the convergence criterion.  After that, you can add any additional measures you like.

# In[121]:


simulation = Simulation(
    ising=ising,
    stepper=stepper,
    is_converged=is_converged
)
simulation.add_measures(Energy())


# Now you can run the simulation for at most 500 steps, computing the measures every 10th step.

# In[122]:


simulation.run(max_steps=500, measure_interval=10)


# As for the ferromagnetic phase, the variation is higher for the Metropolis-Hastings dynamics than for the Glauber dynamics.
