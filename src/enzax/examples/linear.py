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
    AllostericReversibleMichaelisMenten(dc_activator=["m2c"], subunits=1),
    AllostericReversibleMichaelisMenten(dc_inhibitor=["m1c"], ki=["m1c"]),
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
parameters = model.parameter_layout.pack(
    {
        "log_k": {
            "km|r1|m1e": 0.1,  # r1's substrate
            "km|r1|m1c": -0.2,  # r1's product
            "dc|r1|m2c": -0.1,  # r1's allosteric activator
            "km|r2|m1c": 0.5,  # r2's substrate
            "km|r2|m2c": 0.0,  # r2's product
            "ki|r2|m1c": 1.0,  # r2's competitive inhibitor
            "dc|r2|m1c": 0.2,  # r2's allosteric inhibitor
            "km|r3|m2c": -1.0,  # r3's substrate
            "km|r3|m2e": 0.5,  # r3's product
        },
        "log_kcat": {"r1": -0.1, "r2": 0.0, "r3": 0.1},
        "log_enzyme": {
            "r1": jnp.log(0.3),
            "r2": jnp.log(0.2),
            "r3": jnp.log(0.1),
        },
        "log_tc": {"r1": -0.2, "r2": 0.3},
        # m1e and m1c share a formation energy, as do m2c and m2e: each group
        # is named after the first species that uses it.
        "dgf": {"m1e": -3.0, "m2c": -1.0},
        "log_conc_unbalanced": {"m1e": jnp.log(0.5), "m2e": jnp.log(0.1)},
        "temperature": 310.0,
    }
)
steady_state = jnp.array([0.43658744, 0.12695706])
