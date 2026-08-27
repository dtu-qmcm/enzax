"""A small kinetic model with a conserved moiety.

The model represents the following made-up metabolic network:

    transA:  A_e <-> A_c
    r1:      A_c <-> B_c                 (competitively inhibited by D_c)
    r2:      A_c <-> C_c                 (isoenzymes r2A and r2B)
    r3:      B_c + X1_c <-> D_c + X2_c
    r4:      C_c <-> D_c
    transD:  D_c <-> D_e
    regX:    Z_c + X2_c <-> X1_c

Two features of the model:

- Reaction `r2` is catalysed by two isoenzymes, `r2A` and `r2B`, which have
  different kinetics: `r2A` is allosterically activated by C_c, whereas
  `r2B` is allosterically inhibited by it. Enzax gives every rate equation its
  own column of the stoichiometric matrix, so the isoenzymes appear here as
  two reactions `r2A` and `r2B` with the same stoichiometry.

- The cofactors X1_c and X2_c form a conserved moiety `X`: reactions r3
  and regX interconvert them, so their total is constant. X2_c is therefore
  declared a dependent species, leaving five independent species to solve for,
  and the total X1_c + X2_c is the parameter `conserved_pools`.

"""

import numpy as np
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

stoichiometry = {
    "transA": {"A_e": -1.0, "A_c": 1.0},
    "r1": {"A_c": -1.0, "B_c": 1.0},
    "r2A": {"A_c": -1.0, "C_c": 1.0},
    "r2B": {"A_c": -1.0, "C_c": 1.0},
    "r3": {"B_c": -1.0, "X1_c": -1.0, "D_c": 1.0, "X2_c": 1.0},
    "r4": {"C_c": -1.0, "D_c": 1.0},
    "transD": {"D_c": -1.0, "D_e": 1.0},
    "regX": {"Z_c": -1.0, "X2_c": -1.0, "X1_c": 1.0},
}
reactions = ["transA", "r1", "r2A", "r2B", "r3", "r4", "transD", "regX"]
species = ["A_c", "A_e", "B_c", "C_c", "D_c", "D_e", "X1_c", "X2_c", "Z_c"]
balanced_species = ["A_c", "B_c", "C_c", "D_c", "X1_c", "X2_c"]
dependent_species = ["X2_c"]
# A and D each live in two compartments, so they share a formation energy.
species_to_dgf_ix = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6], dtype=np.int16)
rate_equations = [
    ReversibleMichaelisMenten(),  # transA
    ReversibleMichaelisMenten(ki=["D_c"]),  # r1, inhibited competitively
    AllostericReversibleMichaelisMenten(  # r2A, activated by C_c
        dc_activator=["C_c"],
        subunits=1,
    ),
    AllostericReversibleMichaelisMenten(  # r2B, inhibited by C_c
        dc_inhibitor=["C_c"],
        subunits=1,
    ),
    ReversibleMichaelisMenten(),  # r3
    ReversibleMichaelisMenten(),  # r4
    ReversibleMichaelisMenten(),  # transD
    ReversibleMichaelisMenten(),  # regX
]
model = RateEquationModel(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    dependent_species=dependent_species,
    species_to_dgf_ix=species_to_dgf_ix,
    rate_equations=rate_equations,
)
parameters = pack_parameters(
    model.parameter_labelling,
    {
        "log_k": {
            "km|transA|A_e": jnp.log(0.3),
            "km|transA|A_c": jnp.log(0.3),
            "km|r1|A_c": jnp.log(0.2),
            "km|r1|B_c": jnp.log(0.4),
            "ki|r1|D_c": jnp.log(0.3),
            "km|r2A|A_c": jnp.log(0.25),
            "km|r2A|C_c": jnp.log(0.5),
            "dc|r2A|C_c": jnp.log(0.1),
            "km|r2B|A_c": jnp.log(0.4),
            "km|r2B|C_c": jnp.log(0.3),
            "dc|r2B|C_c": jnp.log(0.1),
            "km|r3|B_c": jnp.log(0.15),
            "km|r3|X1_c": jnp.log(0.5),
            "km|r3|D_c": jnp.log(0.3),
            "km|r3|X2_c": jnp.log(0.4),
            "km|r4|C_c": jnp.log(0.2),
            "km|r4|D_c": jnp.log(0.35),
            "km|transD|D_c": jnp.log(0.25),
            "km|transD|D_e": jnp.log(0.25),
            "km|regX|X2_c": jnp.log(0.4),
            "km|regX|Z_c": jnp.log(0.1),
            "km|regX|X1_c": jnp.log(0.5),
        },
        "log_kcat": {
            "transA": jnp.log(2.0),
            "r1": jnp.log(1.0),
            "r2A": jnp.log(0.5),
            "r2B": jnp.log(0.5),
            "r3": jnp.log(1.5),
            "r4": jnp.log(1.0),
            "transD": jnp.log(2.0),
            "regX": jnp.log(3.0),
        },
        "log_enzyme": {
            "transA": jnp.log(0.2),
            "r1": jnp.log(0.3),
            "r2A": jnp.log(0.15),
            "r2B": jnp.log(0.15),
            "r3": jnp.log(0.25),
            "r4": jnp.log(0.2),
            "transD": jnp.log(0.2),
            "regX": jnp.log(0.3),
        },
        "log_tc": {"r2A": jnp.log(0.5), "r2B": jnp.log(2.0)},
        # Each formation energy is named after the first species that uses it,
        # so A_c stands for A in both compartments and D_c for D in both.
        "dgf": {
            "A_c": 0.0,
            "B_c": -5.0,
            "C_c": -5.0,
            "D_c": -15.0,
            "X1_c": 0.0,
            "X2_c": 5.0,
            "Z_c": 5.0,
        },
        "log_conc_unbalanced": {
            "A_e": jnp.log(0.5),
            "D_e": jnp.log(0.05),
            "Z_c": jnp.log(0.2),
        },
        "conserved_pools": {"X2_c": 1.0},  # X1_c + X2_c
        "temperature": 298.15,
    },
)
# Concentrations of the independent balanced species at steady state. X2_c's
# concentration follows from X1_c's and the conserved pool total.
steady_state = jnp.array(
    [
        0.19887131,  # A_c
        0.15278682,  # B_c
        0.06575086,  # C_c
        0.20105146,  # D_c
        0.81022648,  # X1_c
    ]
)
