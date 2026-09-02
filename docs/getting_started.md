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

Enzax's main function is to help you describe the dynamics of an enzyme-catalysed reaction network. The first step towards doing this is to define a "kinetic model", that is, a parameterised function that says how to the network's fluxes, and its states' dynamics, depend on its parameters and the present values of its states. Enzax provides an abstract base class `KineticModel` class for this, as well as handy subclasses like `RateEquationModel`.

Here is a simple example of how to specify a `RateEquationModel` describing a simple linear pathway with two state variables, two boundary species and three reactions.

```python

from enzax.kinetic_model import RateEquationModel
from enzax.rate_equations import MichaelisMenten

my_model = RateEquationModel(
    stoichiometry={
        "r1": {"m1e": -1.0, "m1c": 1.0},
        "r2": {"m1c": -1.0, "m2c": 1.0},
        "r3": {"m2c": -1.0, "m2e": 1.0},
    },
    balanced_species=["m1c", "m2c"],
    compound_to_species={"m1": ["m1e", "m1c"], "m2": ["m2c", "m2e"]},
    rate_equations = {
        "r1": MichaelisMenten(allosteric_activators=["m2c"]),
        "r2": MichaelisMenten(allosteric_inhibitors=["m1c"], competitive_inhibitors=["m1c"]),
        "r3": MichaelisMenten(water_stoichiometry=0.0),
    },
)

```
The first statement imports `RateEquationModel`.

The second statement imports the rate equaiton class `MichaelisMenten`. Instances of this class define rate equations that determine the flux of a reaction.

The third statement initialises a `RateEquationModel` instance. Let's go through the arguments:
- the `stoichiometry` argument specifies, for every reaction, the stoichiometric coefficient of each of its reactants.
- the `balanced_species` argument indicates which of the species mentioned in `stoichiometry` (i.e. `"m1e"`, `"m1c"`, `"m2c"` and `"m2e"`) are assumed to have potentially-changing abundances, i.e. are "balanced". The model assumes that other unbalanced species have constant concentrations, helping to determine the system's boundary conditions.
- `compound_to_species` maps ids of compounds to ids of their species: for example `"m1e"` and `"m1c"` belong to the compound `"m1"`. This is important for correctly representing the thermodynamics of single-compound reactions like `"r1"` and `"r2"`. Note that not every compound has to appear here: an unmentioned species is assumed to be its own singleton compound.
- `rate_equations` maps reaction ids to rate equation instances. For example, reaction `"r2"` obeys allosteric Michaelis-Menten kinetics, allosterically and competitively inhibited by `"m1c"`.

### Parameters and their labels

To do something with `my_model` we need some parameters. In enzax a set of parameters is a mapping whose keys are parameter names and whose values are flat, labelled JAX arrays. Building a model determines which labels exist; you can find a model's parameter names and the shape and labels of each parameter in the `parameter_labelling` attribute:

```python
model.parameter_labelling
```

```
{'log_kcat': ('r1', 'r2', 'r3'), 'log_enzyme': ('r1', 'r2', 'r3'), 'log_saturation_constant': ('km|r1|m1e', 'km|r1|m1c', 'dc|r1|m2c', 'km|r2|m1c', 'km|r2|m2c', 'ki|r2|m1c', 'dc|r2|m1c', 'km|r3|m2c', 'km|r3|m2e'), 'log_tc': ('r1', 'r2'), 'dgf': ('m1', 'm2'), 'log_conc_unbalanced': ('m1e', 'm2e'), 'temperature': ()}
```

`log_saturation_constant` holds every constant that a concentration is divided by, whatever it does. The prefix says which kind it is: `km` for a Michaelis constant, `ki` for a competitive inhibition constant and `dc` for an allosteric dissociation constant. Keeping them in one array lets reactions share constants and allows a reaction to reuse the same constant in two different roles if required.

We build a parameter set by giving a value for every label:

```python
from jax import numpy as jnp

from enzax.parameters import pack_parameters

my_parameters = pack_parameters(
    my_model.parameter_labelling,
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
my_parameters
```
```
{'log_saturation_constant': Array([ 0.1, -0.2, -0.1,  0.5,  0. ,  1. ,  0.2, -1. ,  0.5], dtype=float64), 'log_kcat': Array([-0.1,  0. ,  0.1], dtype=float64), 'log_enzyme': Array([-1.2039728 , -1.60943791, -2.30258509], dtype=float64), 'log_tc': Array([-0.2,  0.3], dtype=float64), 'dgf': Array([-3., -1.], dtype=float64), 'log_conc_unbalanced': Array([-0.69314718, -2.30258509], dtype=float64), 'temperature': Array(310., dtype=float64, weak_type=True)}
```

Note that the `temperature` parameter is different from the others: it has no labels at all, because its whole array is one parameter, so it takes a value directly rather than a mapping.

Going the other way, `enzax.parameters.unpack_parameters(my_model.parameter_labelling, my_parameters)`
gives back a dictionary of labelled values, which is handy when a traceback shows you something like `parameters["log_saturation_constant"][6]` and you want to know which constant that is.

Note that the parameters use `jnp` whereas the structure uses `np`. This is because we want JAX to trace the parameters, whereas the structure should be static. Read more about this [here](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#static-vs-traced-operations).

### Doing things with a kinetic model

To test out our model, we can see if it returns some fluxes and state variable rates when provided a set of balanced species concentrations:

```python
conc = jnp.array([0.43658744, 0.12695706])
flux = my_model.flux(conc, my_parameters)
flux
```
```
Array([0.00576084, 0.00576084, 0.00576084], dtype=float64)
```

```python
dcdt = my_model.dcdt(conc, my_parameters)
dcdt
```

```
Array([-4.04273105e-10,  1.12297637e-09], dtype=float64)
```

These methods work with standard JAX transformations!

```python
import jax

jax.jacobian(my_model.dcdt)(conc, my_parameters)
Array([[-0.12574924,  0.08105433],
       [ 0.03229972, -0.28213634]], dtype=float64)
```

## Find a kinetic model's steady state

To find a kinetic model's steady state, i.e. balanced species concentrations that do not change under the model's own dynamics, you can use enzax's `get_steady_state` function:

```python
from enzax.steady_state import get_steady_state
from jax import numpy as jnp

my_guess = jnp.full((len(my_model.balanced_species),), 0.01)

steady_state = get_steady_state(my_model, my_guess, my_parameters)
```

To access the Jacobian of this steady state with respect to the model's parameters, we can use JAX's [`jacrev`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html) function:

```python
jacobian = jax.jacrev(get_steady_state, argnums=2)(my_model, my_guess, my_parameters)
jacobian
```

```
{'dgf': Array([[-0.02347218,  0.02347218],
[ 0.01090831, -0.01090831]], dtype=float64), 'log_conc_unbalanced': Array([[0.38172522, 0.05332598],
[0.04370092, 0.08273094]], dtype=float64), 'log_enzyme': Array([[ 0.04946205, -0.03525221, -0.01420984],
[ 0.00566255,  0.01638288, -0.02204542]], dtype=float64), 'log_kcat': Array([[ 0.04946205, -0.03525221, -0.01420984],
[ 0.00566255,  0.01638288, -0.02204542]], dtype=float64), 'log_saturation_constant': Array([[-0.04118528,  0.00975552, -0.00161623,  0.03249426, -0.00132227,
 -0.00167278, -0.0050253 ,  0.01072141, -0.0006131 ],
[-0.004715  ,  0.00111684, -0.00018503, -0.01510117,  0.0006145 ,
  0.0007774 ,  0.00233543,  0.01663341, -0.00095118]],      dtype=float64), 'log_tc': Array([[-0.01313526,  0.01908416],
[-0.00150376, -0.00886905]], dtype=float64), 'temperature': Array([-1.51433444e-04,  7.03761766e-05], dtype=float64, weak_type=True)}
```

Because each parameter is a flat array, each entry of the Jacobian is a dense matrix whose columns are that parameter's values, in `model.parameter_labelling` order. To pick out a single value's column, we can ask the labels where it lives:

```python
from enzax.parameters import get_parameter_position

position = get_parameter_position(model.parameter_labelling, "log_kcat", "r2")
jacobian["log_kcat"][:,  position]
```

### Sharing a parameter between reactions

Two rate equations that use the same label use the same value --- one position in one array, one thing to infer. Say `r1` and `r3` were catalysed by the same enzyme and had the same Michaelis constant for their substrates:

```python
shared_rate_equations = {
    "r1": MichaelisMenten(
        allosteric_activators=["m2c"],
        subunits=1,
        enzyme="E1",
        michaelis_constants={"m1e": "km|E1|substrate"},
    ),
    "r2": MichaelisMenten(allosteric_inhibitors=["m1c"], competitive_inhibitors=["m1c"]),
    "r3": MichaelisMenten(
        water_stoichiometry=0.0,
        enzyme="E1",
        michaelis_constants={"m2c": "km|E1|substrate"},
    ),
}
```

Now `log_enzyme` has one entry labelled `E1` instead of two, and `km|E1|substrate` is one position that both reactions gather from. Gradients with respect to it accumulate contributions from both reactions, as they should.

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
