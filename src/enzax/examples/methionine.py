"""A simple linear kinetic model."""

from enzax.kinetic_model import (
    KineticModelParameters,
    KineticModelStructure,
    UnparameterisedKineticModel,
    KineticModel,
)
from enzax.rate_equations import (
    Drain,
    AllostericIrreversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)
from jax import numpy as jnp

parameters = KineticModelParameters(
    log_kcat=jnp.array([-0.1, 0.0, 0.1]),
    log_enzyme=jnp.log(jnp.array([0.3, 0.2, 0.1])),
    log_drain=jnp.array([-1]),
    dgf=jnp.array([-3, -1.0]),
    log_km=jnp.array([0.1, -0.2, 0.5, 0.0, -1.0, 0.5]),
    log_ki=jnp.array([1.0]),
    log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.1])),
    temperature=jnp.array(310.0),
    log_transfer_constant=jnp.array([-0.2, 0.3]),
    log_dissociation_constant=jnp.array([-0.1, 0.2]),
)
structure = KineticModelStructure(
    S=jnp.array(
        [
            [1, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1],
            [0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
    ),
    water_stoichiometry=jnp.array([0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0]),
    balanced_species=jnp.array([0, 4, 5, 8, 11]),
    rate_to_km_ixs=[
        [],
        [0, 1],
        [2, 3],
        [4],
        [5, 6],
        [7, 8, 9],
        [10, 11],
        [12, 13],
        [14, 15],
        [16, 17],
        [18],
    ],
    species_to_metabolite_ix=jnp.arange(19),
    rate_to_subunits=jnp.array([1, 2, 1, 4, 1, 1, 1, 2, 2, 1]),
    rate_to_tc_ix=[[], [], [0], [], [1], [], [], [], [2], [3], []],
    rate_to_dc_ixs_activation=[[], [], [0, 1], [], [2], [], [], [], [4], [6], []],
    rate_to_drain_ix=[[0], [], [], [], [], [], [], [], [], []],
    drain_sign=jnp.array([1.0]),
    rate_to_dc_ixs_inhibition=[[], [], [], [], [3], [], [], [], [], [5], []],
    dc_to_species_ix=jnp.array([4, 0, 4, 12, 4, 4, 5]),
    ki_to_species_ix=jnp.array([5, 6, 6]),
    rate_to_ki_ixs=[[], [0], [], [2], [3], [], [], [], [], []],
)
unparameterised_model = UnparameterisedKineticModel(
    structure,
    [
        Drain,  # met-L source
        IrreversibleMichaelisMenten,  # MAT1 METAT
        AllostericIrreversibleMichaelisMenten,  # MAT3 METAT
        IrreversibleMichaelisMenten,  # METH Gen
        AllostericIrreversibleMichaelisMenten,  # GNMT1
        ReversibleMichaelisMenten,  # AHC
        IrreversibleMichaelisMenten,  # MS
        IrreversibleMichaelisMenten,  # BHMT
        AllostericIrreversibleMichaelisMenten,  # CBS
        AllostericIrreversibleMichaelisMenten,  # MTHFR
        IrreversibleMichaelisMenten,  # PROT
    ],
)
model = KineticModel(parameters, unparameterised_model)
