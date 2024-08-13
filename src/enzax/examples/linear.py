"""A simple linear kinetic model."""

from jax import config
from jax import numpy as jnp

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

config.update("jax_enable_x64", True)

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
structure = KineticModelStructure(
    S=jnp.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]]),
    water_stoichiometry=jnp.array([0, 0, 0]),
    balanced_species=jnp.array([1, 2]),
    rate_to_km_ixs=jnp.array([[0, 1], [2, 3], [4, 5]]),
    species_to_metabolite_ix=jnp.array([0, 0, 1, 1]),
    rate_to_subunits=jnp.array([1, 1, 1]),
    rate_to_tc_ix=[[0], [1], []],
    rate_to_dc_ixs_activation=[[0], [], []],
    rate_to_dc_ixs_inhibition=[[], [1], []],
    rate_to_drain_ix=[[], [], []],
    drain_sign=[],
    dc_to_species_ix=jnp.array([2, 1]),
    ki_to_species_ix=jnp.array([1]),
    rate_to_ki_ixs=[[], [0], []],
)
unparameterised_model = UnparameterisedKineticModel(
    structure,
    [
        AllostericReversibleMichaelisMenten,
        AllostericReversibleMichaelisMenten,
        ReversibleMichaelisMenten,
    ],
)
model = KineticModel(parameters, unparameterised_model)
