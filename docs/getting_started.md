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

from jax import numpy as jnp

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

Next we specify the model's rate equations. The order of the equations should match our `reactions` list. Species are referred to by id, so `dc_activator=["m2c"]` says that `r1` is allosterically activated by `m2c`.

```python
rate_equations = [
    AllostericReversibleMichaelisMenten(dc_activator=["m2c"], subunits=1),
    AllostericReversibleMichaelisMenten(dc_inhibitor=["m1c"], ki=["m1c"]),
    ReversibleMichaelisMenten(water_stoichiometry=0.0),
]
```

Now we can make a RateEquationModel object. Note the `species_to_dgf_ix` argument, which says that `m1e` and `m1c` share a formation energy, as do `m2c` and `m2e`: the model has four species but only two formation energies.

```python
model = RateEquationModel(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    species_to_dgf_ix=np.array([0, 0, 1, 1]),
    rate_equations=rate_equations,
)

```

### Parameters and the parameter layout

Parameters live in flat arrays, one per kind, and each parameter has a name. Building the model works out which names exist, from the rate equations and from the model's structure, and records them in a `ParameterLayout`:

```python
model.parameter_layout.names["log_k"]
```

```
('km|r1|m1e', 'km|r1|m1c', 'dc|r1|m2c', 'km|r2|m1c', 'km|r2|m2c',
 'ki|r2|m1c', 'dc|r2|m1c', 'km|r3|m2c', 'km|r3|m2e')
```

`log_k` holds every dissociation constant, whatever its role. The prefix says which role a constant plays: `km` for a Michaelis constant, `ki` for a competitive inhibition constant and `dc` for an allosteric one. Keeping them in one array is what lets two reactions share a constant, or a reaction reuse one of its own Michaelis constants as an allosteric constant --- in both cases the two uses simply name the same slot.

We build a parameter set by giving a value for every name, and let the layout pack them into arrays:

```python
parameters = model.parameter_layout.pack(
    {
        "log_k": {
            "km|r1|m1e": 0.1,
            "km|r1|m1c": -0.2,
            "dc|r1|m2c": -0.1,
            "km|r2|m1c": 0.5,
            "km|r2|m2c": 0.0,
            "ki|r2|m1c": 1.0,
            "dc|r2|m1c": 0.2,
            "km|r3|m2c": -1.0,
            "km|r3|m2e": 0.5,
        },
        "log_kcat": {"r1": -0.1, "r2": 0.0, "r3": 0.1},
        "log_enzyme": {
            "r1": jnp.log(0.3),
            "r2": jnp.log(0.2),
            "r3": jnp.log(0.1),
        },
        "log_tc": {"r1": -0.2, "r2": 0.3},
        "dgf": {"m1e": -3.0, "m2c": -1.0},
        "log_conc_unbalanced": {"m1e": jnp.log(0.5), "m2e": jnp.log(0.1)},
        "temperature": 310.0,
    }
)
```

`pack` complains about a name it does not recognise and about a name you leave out, so a typo or an omission is an error when you build the parameters rather than a wrong number later. It applies no transform: a `log_` key wants a value on the log scale, which is why the enzyme concentrations above are wrapped in `jnp.log`.

Going the other way, `model.parameter_layout.unpack(parameters)` gives back a dictionary of named values, which is handy when a traceback shows you something like `parameters["log_k"][6]` and you want to know which constant that is.

Note that the parameters use `jnp` whereas the structure uses `np`. This is because we want JAX to trace the parameters, whereas the structure should be static. Read more about this [here](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#static-vs-traced-operations).

### Sharing a parameter between reactions

Two rate equations that use the same name use the same parameter --- one array slot, one leaf, one thing to infer. Say `r1` and `r3` were catalysed by the same enzyme and had the same Michaelis constant for their substrates:

```python
shared_rate_equations = [
    AllostericReversibleMichaelisMenten(
        dc_activator=["m2c"],
        subunits=1,
        enzyme="E1",
        k={"m1e": "km|E1|substrate"},
    ),
    AllostericReversibleMichaelisMenten(dc_inhibitor=["m1c"], ki=["m1c"]),
    ReversibleMichaelisMenten(
        water_stoichiometry=0.0,
        enzyme="E1",
        k={"m2c": "km|E1|substrate"},
    ),
]
```

Now `log_enzyme` has one entry named `E1` instead of two, and `km|E1|substrate` is one slot that both reactions gather from. Gradients with respect to it accumulate contributions from both reactions, as they should.

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

guess = jnp.full((5,), 0.01)

steady_state = get_steady_state(model, guess, parameters)
```

To access the Jacobian of this steady state with respect to the model's parameters, we can use JAX's [`jacrev`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html) function:

```python
jacobian = jax.jacrev(get_steady_state, argnums=2)(model, guess, parameters)
jacobian
```

Because each parameter kind is one flat array, each entry of the Jacobian is a dense matrix whose columns are that kind's parameters, in `model.parameter_layout.names` order. To pick out a single parameter's column, ask the layout where it lives:

```python
layout = model.parameter_layout
jacobian["log_kcat"][:, layout.index("log_kcat", "GNMT1")]
```
