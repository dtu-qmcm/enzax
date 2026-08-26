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
    ReversibleMichaelisMenten(  # r1, inhibited competitively by D_c
        ix_ki_species=np.array([4], dtype=np.int16),
    ),
    AllostericReversibleMichaelisMenten(  # r2A, activated by C_c
        ix_allosteric_activators=np.array([3], dtype=np.int16),
        subunits=1,
    ),
    AllostericReversibleMichaelisMenten(  # r2B, inhibited by C_c
        ix_allosteric_inhibitors=np.array([3], dtype=np.int16),
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
parameters = dict(
    log_kcat={
        "transA": jnp.log(jnp.array(2.0)),
        "r1": jnp.log(jnp.array(1.0)),
        "r2A": jnp.log(jnp.array(0.5)),
        "r2B": jnp.log(jnp.array(0.5)),
        "r3": jnp.log(jnp.array(1.5)),
        "r4": jnp.log(jnp.array(1.0)),
        "transD": jnp.log(jnp.array(2.0)),
        "regX": jnp.log(jnp.array(3.0)),
    },
    log_enzyme={
        "transA": jnp.log(jnp.array(0.2)),
        "r1": jnp.log(jnp.array(0.3)),
        "r2A": jnp.log(jnp.array(0.15)),
        "r2B": jnp.log(jnp.array(0.15)),
        "r3": jnp.log(jnp.array(0.25)),
        "r4": jnp.log(jnp.array(0.2)),
        "transD": jnp.log(jnp.array(0.2)),
        "regX": jnp.log(jnp.array(0.3)),
    },
    log_substrate_km={
        "transA": jnp.log(jnp.array([0.3])),  # A_e
        "r1": jnp.log(jnp.array([0.2])),  # A_c
        "r2A": jnp.log(jnp.array([0.25])),  # A_c
        "r2B": jnp.log(jnp.array([0.4])),  # A_c
        "r3": jnp.log(jnp.array([0.15, 0.5])),  # B_c, X1_c
        "r4": jnp.log(jnp.array([0.2])),  # C_c
        "transD": jnp.log(jnp.array([0.25])),  # D_c
        "regX": jnp.log(jnp.array([0.4, 0.1])),  # X2_c, Z_c
    },
    log_product_km={
        "transA": jnp.log(jnp.array([0.3])),  # A_c
        "r1": jnp.log(jnp.array([0.4])),  # B_c
        "r2A": jnp.log(jnp.array([0.5])),  # C_c
        "r2B": jnp.log(jnp.array([0.3])),  # C_c
        "r3": jnp.log(jnp.array([0.3, 0.4])),  # D_c, X2_c
        "r4": jnp.log(jnp.array([0.35])),  # D_c
        "transD": jnp.log(jnp.array([0.25])),  # D_e
        "regX": jnp.log(jnp.array([0.5])),  # X1_c
    },
    log_ki={
        "transA": jnp.array([]),
        "r1": jnp.log(jnp.array([0.3])),  # D_c
        "r2A": jnp.array([]),
        "r2B": jnp.array([]),
        "r3": jnp.array([]),
        "r4": jnp.array([]),
        "transD": jnp.array([]),
        "regX": jnp.array([]),
    },
    log_tc={
        "r2A": jnp.log(jnp.array(0.5)),
        "r2B": jnp.log(jnp.array(2.0)),
    },
    log_dc_activator={
        "r2A": jnp.log(jnp.array([0.1])),  # C_c
        "r2B": jnp.array([]),
    },
    log_dc_inhibitor={
        "r2A": jnp.array([]),
        "r2B": jnp.log(jnp.array([0.1])),  # C_c
    },
    dgf=jnp.array(
        [
            0.0,  # A
            -5.0,  # B
            -5.0,  # C
            -15.0,  # D
            0.0,  # X1
            5.0,  # X2
            5.0,  # Z
        ]
    ),
    temperature=jnp.array(298.15),
    log_conc_unbalanced=jnp.log(
        jnp.array(
            [
                0.5,  # A_e
                0.05,  # D_e
                0.2,  # Z_c
            ]
        )
    ),
    conserved_pools=jnp.array([1.0]),  # X1_c + X2_c
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
