"""A simple linear kinetic model."""

import numpy as np
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


stoichiometry = {
    "r1": {"m1e": -1.0, "m1c": 1.0},
    "r2": {"m1c": -1.0, "m2c": 1.0},
    "r3": {"m2c": -1.0, "m2e": 1.0},
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
model = RateEquationModel(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    species_to_dgf_ix=np.array([0, 0, 1, 1]),
    rate_equations=rate_equations,
)
parameters = dict(
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
    log_kcat={
        "r1": jnp.array(-0.1),
        "r2": jnp.array(0.0),
        "r3": jnp.array(0.1),
    },
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
steady_state = jnp.array([0.43658744, 0.12695706])
