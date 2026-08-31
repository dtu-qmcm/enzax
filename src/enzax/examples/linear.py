"""A simple linear kinetic model."""

from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

stoichiometry = {
    "r1": {"m1e": -1.0, "m1c": 1.0},
    "r2": {"m1c": -1.0, "m2c": 1.0},
    "r3": {"m2c": -1.0, "m2e": 1.0},
}
balanced_species = ["m1c", "m2c"]
rate_equations = [
    AllostericReversibleMichaelisMenten(
        allosteric_activators=["m2c"], subunits=1
    ),
    AllostericReversibleMichaelisMenten(
        allosteric_inhibitors=["m1c"], competitive_inhibitors=["m1c"]
    ),
    ReversibleMichaelisMenten(water_stoichiometry=0.0),
]
model = RateEquationModel(
    stoichiometry=stoichiometry,
    balanced_species=balanced_species,
    compound_to_species={"m1": ["m1e", "m1c"], "m2": ["m2c", "m2e"]},
    rate_equations=rate_equations,
)
parameters = pack_parameters(
    model.parameter_labelling,
    {
        "log_saturation_constant": {
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
        # m1e and m1c are the same compound m1, as are m2c and m2e, so
        # there are two formation energies rather than four.
        "dgf": {"m1": -3.0, "m2": -1.0},
        "log_conc_unbalanced": {"m1e": jnp.log(0.5), "m2e": jnp.log(0.1)},
        "temperature": 310.0,
    },
)
steady_state = jnp.array([0.43658744, 0.12695706])
