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

```python
from enzax.kinetic_model import (
    KineticModel,
    KineticModelParameters,
    KineticModelStructure,
    UnparameterisedKineticModel,
)
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

```

```python
parameters = KineticModelParameters(
    log_kcat=jnp.array([-0.1, 0.0, 0.1]),
    log_enzyme=jnp.log(jnp.array([0.3, 0.2, 0.1])),
    dgf=jnp.array([-3, -1.0]),
    log_km=jnp.array([0.1, -0.2, 0.5, 0.0, -1.0, 0.5]),
    log_ki=jnp.array([1.0]),
    log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.1])),
    temperature=jnp.array(310.0),
    log_transfer_constant=jnp.array([-0.2, 0.3]),
    log_dissociation_constant=jnp.array([-0.1, 0.2]),
    log_drain=jnp.array([]),
)
```

```python
structure = KineticModelStructure(
    S=jnp.array(
        [[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]], dtype=jnp.float64
    ),
    balanced_species=jnp.array([1, 2]),
    unbalanced_species=jnp.array([0, 3]),
)
```

Now we can make some rate laws

```python
r0 = AllostericReversibleMichaelisMenten(
    kcat_ix=0,
    enzyme_ix=0,
    km_ix=jnp.array([0, 1], dtype=jnp.int16),
    ki_ix=jnp.array([], dtype=jnp.int16),
    reactant_stoichiometry=jnp.array([-1, 1], dtype=jnp.int16),
    reactant_to_dgf=jnp.array([0, 0], dtype=jnp.int16),
    ix_ki_species=jnp.array([], dtype=jnp.int16),
    substrate_km_positions=jnp.array([0], dtype=jnp.int16),
    substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
    ix_substrate=jnp.array([0], dtype=jnp.int16),
    ix_product=jnp.array([1], dtype=jnp.int16),
    ix_reactants=jnp.array([0, 1], dtype=jnp.int16),
    product_reactant_positions=jnp.array([1], dtype=jnp.int16),
    product_km_positions=jnp.array([1], dtype=jnp.int16),
    water_stoichiometry=jnp.array(0.0),
    tc_ix=0,
    ix_dc_inhibition=jnp.array([], dtype=jnp.int16),
    ix_dc_activation=jnp.array([0], dtype=jnp.int16),
    species_activation=jnp.array([2], dtype=jnp.int16),
    species_inhibition=jnp.array([], dtype=jnp.int16),
    subunits=1,
)
r1 = AllostericReversibleMichaelisMenten(
    kcat_ix=1,
    enzyme_ix=1,
    km_ix=jnp.array([2, 3], dtype=jnp.int16),
    ki_ix=jnp.array([0]),
    reactant_stoichiometry=jnp.array([-1, 1], dtype=jnp.int16),
    reactant_to_dgf=jnp.array([0, 1], dtype=jnp.int16),
    ix_ki_species=jnp.array([1]),
    substrate_km_positions=jnp.array([0], dtype=jnp.int16),
    substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
    ix_substrate=jnp.array([1], dtype=jnp.int16),
    ix_product=jnp.array([2], dtype=jnp.int16),
    ix_reactants=jnp.array([1, 2], dtype=jnp.int16),
    product_reactant_positions=jnp.array([1], dtype=jnp.int16),
    product_km_positions=jnp.array([1], dtype=jnp.int16),
    water_stoichiometry=jnp.array(0.0),
    tc_ix=1,
    ix_dc_inhibition=jnp.array([1], dtype=jnp.int16),
    ix_dc_activation=jnp.array([], dtype=jnp.int16),
    species_activation=jnp.array([], dtype=jnp.int16),
    species_inhibition=jnp.array([1], dtype=jnp.int16),
    subunits=1,
)
r2 = ReversibleMichaelisMenten(
    kcat_ix=2,
    enzyme_ix=2,
    km_ix=jnp.array([4, 5], dtype=jnp.int16),
    ki_ix=jnp.array([], dtype=jnp.int16),
    ix_substrate=jnp.array([2], dtype=jnp.int16),
    ix_product=jnp.array([3], dtype=jnp.int16),
    ix_reactants=jnp.array([2, 3], dtype=jnp.int16),
    reactant_to_dgf=jnp.array([1, 1], dtype=jnp.int16),
    reactant_stoichiometry=jnp.array([-1, 1], dtype=jnp.int16),
    ix_ki_species=jnp.array([], dtype=jnp.int16),
    substrate_km_positions=jnp.array([0], dtype=jnp.int16),
    substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
    product_reactant_positions=jnp.array([1], dtype=jnp.int16),
    product_km_positions=jnp.array([1], dtype=jnp.int16),
    water_stoichiometry=jnp.array(0.0),
)
```

Next an unparameterised kinetic model

```python
unparameterised_model = UnparameterisedKineticModel(structure, [r0, r1, r2])
```

Finally a parameterised model:

```python
model = KineticModel(parameters, unparameterised_model)
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

Enzax provides a few example kinetic models, including [`methionine`](https://github.com/dtu-qmcm/enzax/blob/main/src/enzax/examples/methionine.py), a model of the mammallian methionine cycle.

Here is how to find this model's steady state (and its parameter gradients) using enzax's `solve_steady_state` function:

```python
from enzax.examples import methionine
from enzax.steady_state import solve_steady_state
from jax import numpy as jnp

guess = jnp.full((5,) 0.01)

steady_state = solve_steady_state(
    methionine.parameters, methionine.unparameterised_model, guess
)
```

To find the jacobian of this steady state with respect to the model's parameters, we can wrap `solve_steady_state` in JAX's [`jacrev`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jacrev.html) function:

```python
import jax

jacobian = jax.jacrev(solve_steady_state)(
    methionine.parameters, methionine.unparameterised_model, guess
)
```
