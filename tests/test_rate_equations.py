"""Unit tests for rate equations."""

from jax import numpy as jnp
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    AllostericReversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)
from enzax.parameters import AllostericMichaelisMentenParameters

EXAMPLE_CONC = jnp.array([0.5, 0.2, 0.1, 0.3])
EXAMPLE_PARAMETERS = AllostericMichaelisMentenParameters(
    log_kcat=jnp.array([-0.1]),
    log_enzyme=jnp.log(jnp.array([0.3])),
    dgf=jnp.array([-3, -1.0]),
    log_km=jnp.array([0.1, -0.2]),
    log_ki=jnp.array([1.0]),
    log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.3])),
    temperature=jnp.array(310.0),
    log_transfer_constant=jnp.array([-0.2, 0.3]),
    log_dissociation_constant=jnp.array([-0.1, 0.2]),
    log_drain=jnp.array([]),
)


def test_irreversible_michaelis_menten():
    expected_rate = 0.08455524
    f = IrreversibleMichaelisMenten(
        kcat_ix=0,
        enzyme_ix=0,
        km_ix=jnp.array([0, 1], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([-1, 1], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
        ix_substrate=jnp.array([0], dtype=jnp.int16),
    )
    rate = f(EXAMPLE_CONC, EXAMPLE_PARAMETERS)
    assert jnp.isclose(rate, expected_rate)


def test_reversible_michaelis_menten():
    expected_rate = 0.04342889
    f = ReversibleMichaelisMenten(
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
    )
    rate = f(EXAMPLE_CONC, EXAMPLE_PARAMETERS)
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_irreversible_michaelis_menten():
    expected_rate = 0.05608589
    f = AllostericIrreversibleMichaelisMenten(
        kcat_ix=0,
        enzyme_ix=0,
        km_ix=jnp.array([0, 1], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([-1, 1], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
        ix_substrate=jnp.array([0], dtype=jnp.int16),
        tc_ix=0,
        ix_dc_inhibition=jnp.array([], dtype=jnp.int16),
        ix_dc_activation=jnp.array([0], dtype=jnp.int16),
        species_activation=jnp.array([2], dtype=jnp.int16),
        species_inhibition=jnp.array([], dtype=jnp.int16),
        subunits=1,
    )
    rate = f(EXAMPLE_CONC, EXAMPLE_PARAMETERS)
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_reversible_michaelis_menten():
    expected_rate = 0.03027414
    f = AllostericReversibleMichaelisMenten(
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
    rate = f(EXAMPLE_CONC, EXAMPLE_PARAMETERS)
    assert jnp.isclose(rate, expected_rate)
