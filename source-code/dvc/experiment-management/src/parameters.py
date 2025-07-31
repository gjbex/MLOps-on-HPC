from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import yaml


class IsingConfig(BaseModel):
    nr_rows: int
    nr_cols: int
    J: float
    h: float
    seed: int


class DynamicsConfig(BaseModel):
    temperature: float
    type: Literal['glauber', 'metropolis_hastings']


class ConvergenceConfig(BaseModel):
    measure: Literal['magnetization', 'energy']
    nr_measurement_steps: int
    delta: float


class SimulationConfig(BaseModel):
    max_steps: int
    measure_interval: int


class Params(BaseModel):
    ising: IsingConfig
    dynamics: DynamicsConfig
    convergence: ConvergenceConfig
    simulation: SimulationConfig


def load_params(file_name: str) -> Params:
    '''Load simulation parameters from a YAML file.
    
    Parameters
    ----------
    file_name: str
      path to the YAML file with simulation parameters
    
    Returns
    -------
    Params
      pydantic object with simulation parameters
    Raises
    ------
    FileNotFoundError
      if the YAML file does not exist
    yaml.YAMLError
      if there is an error parsing the YAML file
    ValidationError
      if the YAML file does not conform to the expected schema
    '''
    with open(file_name, "r") as f:
        raw_params = yaml.safe_load(f)
    return Params(**raw_params)

def walk_model(model: BaseModel, prefix=''):
    for field_name, field in model.model_fields.items():
        value = getattr(model, field_name)
        full_name = f'{prefix}.{field_name}' if prefix else field_name

        if isinstance(value, BaseModel):
            yield from walk_model(value, prefix=full_name)
        else:
            yield (full_name, value)

def log_params(live, params):
    '''Log simulation parameters using dvclive.
    
    Parameters
    ----------
    live: dvclive.Live
      instance of dvclive to log parameters
    params: Params
      Pydantic object with simulation parameters
    '''
    for name, value in walk_model(params):
        live.log_param(name, value)
