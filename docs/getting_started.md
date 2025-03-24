# Getting started

## How to install enzax

```sh
pip install enzax
```

To install the latest version of enzax from GitHub:

```
$ pip install git+https://github.com/dtu-qmcm/enzax.git@main
```

## Make your own kinetic model

Enzax provides building blocks for you to construct a wide range of differentiable kinetic models using pre-written and tested rate laws.

Here we write a model describing a simple linear pathway with two state variables, two boundary species and three reactions.

First we import some enzax classes, as well as [equinox](https://github.com/patrick-kidger/equinox) and both JAX and standard versions of numpy:

```python
import equinox as eqx

import numpy as np

from enzax.kinetic_model import RateEquationModel
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

```

Next we start specifying our model's structure by providing stoichiometric coefficients for its reactions and saying which species represent ODE state variables (aka which ones are "balanced").

```python
stoichiometry = {
    "r1": {"m1e": -1.0, "m1c": 1.0},
    "r2": {"m1c": -1.0, "m2c": 1.0},
    "r3": {"m2c": -1.0, "m2e": 1.0},
}
reactions = ["r1", "r2", "r3"]
species = ["m1e", "m1c", "m2c", "m2e"]
balanced_species = ["m1c", "m2c"]
```

Next we specify the model's rate equations. Note that the order of the equations should match our `reactions` list and that the indexes that refer to species, like `ix_allosteric_activators` and `ix_ki_species` should match the order of the `species` list.

```python
rate_equations = [
    AllostericReversibleMichaelisMenten(
        ix_allosteric_activators=np.array([2]), subunits=1
    ),
    AllostericReversibleMichaelisMenten(
        ix_allosteric_inhibitors=np.array([1]), ix_ki_species=np.array([1])
    ),
    ReversibleMichaelisMenten(water_stoichiometry=0.0),
]
```

Now we can make a RateEquationModel object

```python
model = RateEquationModel(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    rate_equations=rate_equations,
)

```

Next we specify a set of kinetic parameters as a dictionary:

```python
from jax import numpy as jnp

parameters = {
    "log_substrate_km": {
        "r1": jnp.array([0.1]),
        "r2": jnp.array([0.5]),
        "r3": jnp.array([-1.0]),
    },
    "log_product_km": {
        "r1": jnp.array([-0.2]),
        "r2": jnp.array([0.0]),
        "r3": jnp.array([0.5]),
    },
    "log_kcat": {"r1": jnp.array(-0.1), "r2": jnp.array(0.0), "r3": jnp.array(0.1)},
    "dgf": jnp.array([-3.0, -1.0]),
    "log_ki": {"r1": jnp.array([]), "r2": jnp.array([1.0]), "r3": jnp.array([])},
    "temperature": jnp.array(310.0),
    "log_enzyme": {
        "r1": jnp.log(jnp.array(0.3)),
        "r2": jnp.log(jnp.array(0.2)),
        "r3": jnp.log(jnp.array(0.1)),
    },
    "log_conc_unbalanced": jnp.log(jnp.array([0.5, 0.1])),
    "log_tc": {"r1": jnp.array(-0.2), "r2": jnp.array(0.3)},
    "log_dc_activator": {"r1": jnp.array([-0.1]), "r2": jnp.array([])},
    "log_dc_inhibitor": {"r1": jnp.array([]), "r2": jnp.array([0.2])},
}
```

Note that the parameters use `jnp` whereas the structure uses `np`. This is because we want JAX to trace the parameters, whereas the structure should be static. Read more about this [here](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#static-vs-traced-operations).

To test out the model, we can see if it returns some fluxes and state variable rates when provided a set of balanced species concentrations:

```python
conc = jnp.array([0.43658744, 0.12695706])
flux = model.flux(conc, parameters)
flux
```

```python
dcdt = model.dcdt(conc, parameters)
dcdt
```

## Load a kinetic model from SBML

Enzax supports loading kinetic models from SBML files, either locally:

```python
from pathlib import Path
from enzax.sbml import load_libsbml_model_from_file, sbml_to_enzax

path = Path("path") / "to" / "sbml_file.xml"
libsbml_model = load_libsbml_model_from_file(path)
model = sbml_to_enzax(libsbml_model)
```


or from a url:

```python
from enzax.sbml import load_libsbml_model_from_url, sbml_to_enzax

url = "https://raw.githubusercontent.com/dtu-qmcm/enzax/refs/heads/main/tests/data/exampleode.xml"
libsbml_model = load_libsbml_model_from_url(url)
model = sbml_to_enzax(libsbml_model)
```

!!! note

    The parameters in the sbml file have to have unique identifiers.
    In CopasiUI it is possible to make Global Quantities as assignments and odes. Enzax currently does not support this.

## Find a kinetic model's steady state

Enzax provides a few example kinetic models, including [`methionine`](https://github.com/dtu-qmcm/enzax/blob/main/src/enzax/examples/methionine.py), a model of the mammalian methionine cycle.

Here is how to find this model's steady state (and its parameter gradients) using enzax's `get_kinetic_model_steady_state` function:

```python
from enzax.examples.methionine import model, parameters
from enzax.steady_state import get_steady_state
from jax import numpy as jnp

guess = jnp.full((5,) 0.01)

steady_state = get_steady_state(model, guess, parameters)
```

To access the Jacobian of this steady state with respect to the model's parameters, we can use JAX's [`jacrev`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html) function:

```python
jacobian = jax.jacrev(get_steady_state, argnums=2)(model, guess, parameters)
jacobian
```
