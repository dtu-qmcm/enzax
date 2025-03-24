# Enzax

[![Tests](https://github.com/dtu-qmcm/enzax/actions/workflows/run_tests.yml/badge.svg)](https://github.com/dtu-qmcm/enzax/actions/workflows/run_tests.yml)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Supported Python versions: 3.12 and newer](https://img.shields.io/badge/python->=3.12-blue.svg)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/enzax/badge/?version=latest)](https://enzax.readthedocs.io/en/latest/?badge=latest)

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
from enzax.steady_state import get_kinetic_model_steady_state
from jax import numpy as jnp

guess = jnp.full((5,) 0.01)

steady_state = get_steady_state(methionine.model, guess, methionine.parameters)
```

### Find a steady state's Jacobian with respect to all parameters

```python
import jax
from enzax.examples.methionine import model, parameters
from enzax.steady_state import get_steady_state
from jax import numpy as jnp

guess = jnp.full((5,), 0.01)

jacobian = jax.jacrev(get_steady_state, argnums=2)(model, guess, parameters)
jacobian["log_kcat"]["GNMT1"]
```
```
Array([-3.83561770e-07, -9.66801636e-06,  3.38183140e-10,  3.15564928e-09,
        5.28588273e-08], dtype=float64, weak_type=True)
```

### Load a kinetic model from an sbml file

```python
from enzax.sbml import load_libsbml_model_from_url, sbml_to_enzax

url = "https://raw.githubusercontent.com/dtu-qmcm/enzax/refs/heads/main/tests/data/exampleode.xml"

libsbml_model = load_libsbml_model_from_url(url)
# or to load an sbml file from your computer:
# libsbml_model = load_libsbml_model_from_file(path_to_file)

model, parameters = sbml_to_enzax(libsbml_model)
```

> [!NOTE]
> The parameters in the sbml file have to have unique identifiers.
> In CopasiUI it is possible to make Global Quantities as assignments and odes. Enzax currently does not support this.
