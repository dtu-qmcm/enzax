# Enzax

[![Tests](https://github.com/dtu-qmcm/enzax/actions/workflows/run_tests.yml/badge.svg)](https://github.com/dtu-qmcm/enzax/actions/workflows/run_tests.yml)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Supported Python versions: 3.12 and newer](https://img.shields.io/badge/python->=3.12-blue.svg)](https://www.python.org/)

Enzax is a library of automatically differentiable equations and solvers for modelling networks of enzyme-catalysed reactions, written in [JAX](https://jax.readthedocs.io/en/latest/).

Enzax provides straightforward, fast and interoperable access to the gradients of realistic metabolic network models, allowing you to incorporate these models in your MCMC and machine learning algorithms when you want to, for example, predict the effect of down-regulating an enzyme on the yield of a fermentation experiment.

## Installation

```sh
pip install enzax
```

## Usage

### Find a kinetic model's steady state

```python
from enzax.examples import methionine
from enzax.steady_state import solve_steady_state
from jax import numpy as jnp

guess = jnp.full((5,) 0.01)

steady_state = solve_steady_state(
    methionine.parameters, methionine.unparameterised_model, guess
)
```

### Find a steady state's Jacobian with respect to all parameters

```python
import jax
from enzax.examples import methionine
from enzax.steady_state import solve_steady_state
from jax import numpy as jnp

guess = jnp.full((5,) 0.01)

jacobian = jax.jacrev(solve_steady_state)(
    methionine.parameters, methionine.unparameterised_model, guess
)
```
