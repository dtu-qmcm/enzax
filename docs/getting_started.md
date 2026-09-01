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

First we import some enzax classes, as well as [equinox](https://github.com/patrick-kidger/equinox) and JAX's version of numpy:

```python
import equinox as eqx

from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters, unpack_parameters
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
balanced_species = ["m1c", "m2c"]
```

The model works out its own reactions and species from this: the reactions are the stoichiometry's keys, in order, and the species are what they consume and produce, in the order they first appear. A species that takes part in no reaction, like an allosteric effector, joins them when a rate equation names it.

Next we specify the model's rate equations, one per reaction id. Species are referred to by id, so `allosteric_activators=["m2c"]` says that `r1` is allosterically activated by `m2c`.

```python
rate_equations = {
    "r1": AllostericReversibleMichaelisMenten(allosteric_activators=["m2c"], subunits=1),
    "r2": AllostericReversibleMichaelisMenten(allosteric_inhibitors=["m1c"], competitive_inhibitors=["m1c"]),
    "r3": ReversibleMichaelisMenten(water_stoichiometry=0.0),
}
```

A reaction with no rate equation, or a rate equation whose key is not one of the stoichiometry's reactions, is an error when the model is built.

Now we can make a RateEquationModel object. Note the `compound_to_species` argument, which says that `m1e` and `m1c` are the same compound `m1`, as are `m2c` and `m2e`: the model has four species but only two compounds, and formation energies belong to compounds. It is partial, so only compounds with more than one species need mentioning; any species no compound claims is a compound of its own, labelled by its own id.

```python
model = RateEquationModel(
    stoichiometry=stoichiometry,
    balanced_species=balanced_species,
    compound_to_species={"m1": ["m1e", "m1c"], "m2": ["m2c", "m2e"]},
    rate_equations=rate_equations,
)

```

### Parameters and their labels

Each parameter is one flat array, and each value in an array has a label. Building the model works out which labels exist, from the rate equations and from the model's structure, and records them in `model.parameter_labelling`:

```python
model.parameter_labelling["log_saturation_constant"]
```

```
('km|r1|m1e', 'km|r1|m1c', 'dc|r1|m2c', 'km|r2|m1c', 'km|r2|m2c',
 'ki|r2|m1c', 'dc|r2|m1c', 'km|r3|m2c', 'km|r3|m2e')
```

`log_saturation_constant` holds every constant that a concentration is divided by, whatever it does. The prefix says which kind it is: `km` for a Michaelis constant, `ki` for a competitive inhibition constant and `dc` for an allosteric dissociation constant. Keeping them in one array is what lets two reactions share a constant, or a reaction reuse one of its own Michaelis constants as an allosteric constant --- in both cases the two uses simply give the same label.

We build a parameter set by giving a value for every label:

```python
parameters = pack_parameters(
    model.parameter_labelling,
    {
        "log_saturation_constant": {
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
        "dgf": {"m1": -3.0, "m2": -1.0},
        "log_conc_unbalanced": {"m1e": jnp.log(0.5), "m2e": jnp.log(0.1)},
        "temperature": 310.0,
    }
)
```

`pack_parameters` complains about a parameter or a label it does not recognise and about a parameter or a label you leave out, so a typo or an omission is an error when you build the parameters rather than a wrong number later. It applies no transform: a `log_` key wants a value on the log scale, which is why the enzyme concentrations above are wrapped in `jnp.log`.

`temperature` is the exception: it has no labels at all, because its whole array is one parameter, so it takes a value directly rather than a mapping.

Going the other way, `unpack_parameters(model.parameter_labelling, parameters)`
gives back a dictionary of labelled values, which is handy when a traceback shows you something like `parameters["log_saturation_constant"][6]` and you want to know which constant that is.

Note that the parameters use `jnp` whereas the structure uses `np`. This is because we want JAX to trace the parameters, whereas the structure should be static. Read more about this [here](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#static-vs-traced-operations).

### Sharing a parameter between reactions

Two rate equations that use the same label use the same value --- one position in one array, one thing to infer. Say `r1` and `r3` were catalysed by the same enzyme and had the same Michaelis constant for their substrates:

```python
shared_rate_equations = {
    "r1": AllostericReversibleMichaelisMenten(
        allosteric_activators=["m2c"],
        subunits=1,
        enzyme="E1",
        michaelis_constants={"m1e": "km|E1|substrate"},
    ),
    "r2": AllostericReversibleMichaelisMenten(allosteric_inhibitors=["m1c"], competitive_inhibitors=["m1c"]),
    "r3": ReversibleMichaelisMenten(
        water_stoichiometry=0.0,
        enzyme="E1",
        michaelis_constants={"m2c": "km|E1|substrate"},
    ),
}
```

Now `log_enzyme` has one entry labelled `E1` instead of two, and `km|E1|substrate` is one position that both reactions gather from. Gradients with respect to it accumulate contributions from both reactions, as they should.

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
from enzax.parameters import get_parameter_position
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

Because each parameter is one flat array, each entry of the Jacobian is a dense matrix whose columns are that parameter's values, in `model.parameter_labelling` order. To pick out a single value's column, ask the labels where it lives:

```python
labelling = model.parameter_labelling
jacobian["log_kcat"][
    :, get_parameter_position(labelling, "log_kcat", "GNMT1")
]
```
