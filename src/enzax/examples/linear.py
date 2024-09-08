"""A simple linear kinetic model."""

from jax import numpy as jnp

from enzax.kinetic_model import (
    RateEquationModel,
    KineticModelStructure,
)
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)
from enzax.parameters import AllostericMichaelisMentenParameterSet

parameters = AllostericMichaelisMentenParameterSet(
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
    S=jnp.array(
        [[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]], dtype=jnp.float64
    ),
    balanced_species=jnp.array([1, 2]),
    unbalanced_species=jnp.array([0, 3]),
)
rate_equations = [
    AllostericReversibleMichaelisMenten(
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
    ),
    AllostericReversibleMichaelisMenten(
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
    ),
    ReversibleMichaelisMenten(
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
    ),
]
steady_state = jnp.array([0.43658744, 0.12695706])
model = RateEquationModel(parameters, structure, rate_equations)
