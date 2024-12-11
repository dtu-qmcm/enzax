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

from jax import numpy as jnp
import numpy as np

from enzax.kinetic_model import (
    KineticModelStructure,
    RateEquationModel,
)
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

```

Next we specify our model's structure by providing a stoichiometric matrix and saying which of its rows represent state variables (aka "balanced species") and which reactions have which rate equations.

```python
stoichiometry = {
    "r1": {"m1e": -1, "m1c": 1},
    "r2": {"m1c": -1, "m2c": 1},
    "r3": {"m2c": -1, "m2e": 1},
}
reactions = ["r1", "r2", "r3"]
species = ["m1e", "m1c", "m2c", "m2e"]
balanced_species = ["m1c", "m2c"]
rate_equations = [
    AllostericReversibleMichaelisMenten(
        ix_allosteric_activators=np.array([2]), subunits=1
    ),
    AllostericReversibleMichaelisMenten(
        ix_allosteric_inhibitors=np.array([1]), ix_ki_species=np.array([1])
    ),
    ReversibleMichaelisMenten(water_stoichiometry=0.0),
]
structure = KineticModelStructure(
    stoichiometry=stoichiometry,
    species=species,
    balanced_species=balanced_species,
    rate_equations=rate_equations,
)
```

Next we define what a set of kinetic parameters looks like for our problem, and provide a set of parameters matching this definition:

```python
class ParameterDefinition(eqx.Module):
    log_substrate_km: dict[int, Array]
    log_product_km: dict[int, Array]
    log_kcat: dict[int, Scalar]
    log_enzyme: dict[int, Array]
    log_ki: dict[int, Array]
    dgf: Array
    temperature: Scalar
    log_conc_unbalanced: Array
    log_dc_inhibitor: dict[int, Array]
    log_dc_activator: dict[int, Array]
    log_tc: dict[int, Array]

parameters = ParameterDefinition(
    log_substrate_km={
        "r1": jnp.array([0.1]),
        "r2": jnp.array([0.5]),
        "r3": jnp.array([-1.0]),
    },
    log_product_km={
        "r1": jnp.array([-0.2]),
        "r2": jnp.array([0.0]),
        "r3": jnp.array([0.5]),
    },
    log_kcat={"r1": jnp.array(-0.1), "r2": jnp.array(0.0), "r3": jnp.array(0.1)},
    dgf=jnp.array([-3.0, -1.0]),
    log_ki={"r1": jnp.array([]), "r2": jnp.array([1.0]), "r3": jnp.array([])},
    temperature=jnp.array(310.0),
    log_enzyme={
        "r1": jnp.log(jnp.array(0.3)),
        "r2": jnp.log(jnp.array(0.2)),
        "r3": jnp.log(jnp.array(0.1)),
    },
    log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.1])),
    log_tc={"r1": jnp.array(-0.2), "r2": jnp.array(0.3)},
    log_dc_activator={"r1": jnp.array([-0.1]), "r2": jnp.array([])},
    log_dc_inhibitor={"r1": jnp.array([]), "r2": jnp.array([0.2])},
)
```
Note that the parameters use `jnp` whereas the structure uses `np`. This is because we want JAX to trace the parameters, whereas the structure should be static. Read more about this [here](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#static-vs-traced-operations).

Now we can declare our model:

```python
model = RateEquationModel(structure, parameters)
```

To test out the model, we can see if it returns some fluxes and state variable rates when provided a set of balanced species concentrations:

```python
conc = jnp.array([0.43658744, 0.12695706])
flux = model.flux(conc)
flux
```

```python
dcdt = model.dcdt(conc)
dcdt
```

## Find a kinetic model's steady state

Enzax provides a few example kinetic models, including [`methionine`](https://github.com/dtu-qmcm/enzax/blob/main/src/enzax/examples/methionine.py), a model of the mammalian methionine cycle.

Here is how to find this model's steady state (and its parameter gradients) using enzax's `get_kinetic_model_steady_state` function:

```python
from enzax.examples import methionine
from enzax.steady_state import get_kinetic_model_steady_state
from jax import numpy as jnp

guess = jnp.full((5,) 0.01)

steady_state = get_kinetic_model_steady_state(methionine.model, guess)
```

To access the Jacobian of this steady state with respect to the model's parameters, we can wrap `get_kinetic_model_steady_state` in a function that has a set of parameters as its only argument, then use JAX's [`jacrev`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html) function:

```python
import jax
from jaxtyping import PyTree

guess = jnp.full((5,) 0.01)
model = methionine.model

def get_steady_state_from_params(parameters: PyTree):
    """Get the steady state with a one-argument non-pure function."""
    _model = RateEquationModel(parameters, model.structure)
    return get_kinetic_model_steady_state(_model, guess)

jacobian = jax.jacrev(get_steady_state_from_params)(model.parameters)
```
