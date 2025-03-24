"""Unit tests for rate equations."""

import numpy as np
from jax import numpy as jnp
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    AllostericReversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


EXAMPLE_S = np.array([[-1], [1], [0]], dtype=np.float64)
EXAMPLE_CONC = jnp.array([0.5, 0.2, 0.1])
EXAMPLE_PARAMETERS = dict(
    log_substrate_km={"r1": jnp.array([0.1])},
    log_product_km={"r1": jnp.array([-0.2])},
    log_kcat={"r1": jnp.array(-0.1)},
    dgf=jnp.array([-3.0, 1.0]),
    log_ki={"r1": jnp.array([1.0])},
    temperature=jnp.array(310.0),
    log_enzyme={"r1": jnp.log(jnp.array(0.3))},
    log_conc_unbalanced=jnp.array([]),
    log_tc={"r1": jnp.array(-0.2)},
    log_dc_activator={"r1": jnp.array([-0.1])},
    log_dc_inhibitor={"r1": jnp.array([0.2])},
)
EXAMPLE_SPECIES_TO_DGF_IX = np.array([0, 0, 1, 1])


def test_irreversible_michaelis_menten():
    expected_rate = 0.08455524
    f = IrreversibleMichaelisMenten()
    f_input = f.get_input(
        parameters=EXAMPLE_PARAMETERS,
        reaction_id="r1",
        reaction_stoichiometry=EXAMPLE_S[:, 0],
        species_to_dgf_ix=EXAMPLE_SPECIES_TO_DGF_IX,
    )
    rate = f(EXAMPLE_CONC, f_input)
    assert jnp.isclose(rate, expected_rate)


def test_reversible_michaelis_menten():
    expected_rate = 0.04342889
    f = ReversibleMichaelisMenten(
        ix_ki_species=np.array([], dtype=np.int16),
        water_stoichiometry=0.0,
    )
    f_input = f.get_input(
        parameters=EXAMPLE_PARAMETERS,
        reaction_id="r1",
        reaction_stoichiometry=EXAMPLE_S[:, 0],
        species_to_dgf_ix=EXAMPLE_SPECIES_TO_DGF_IX,
    )
    rate = f(EXAMPLE_CONC, f_input)
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_irreversible_michaelis_menten():
    expected_rate = 0.05608589
    f = AllostericIrreversibleMichaelisMenten(
        ix_ki_species=np.array([], dtype=np.int16),
        ix_allosteric_activators=np.array([2], dtype=np.int16),
        subunits=1,
    )
    f_input = f.get_input(
        parameters=EXAMPLE_PARAMETERS,
        reaction_id="r1",
        reaction_stoichiometry=EXAMPLE_S[:, 0],
        species_to_dgf_ix=EXAMPLE_SPECIES_TO_DGF_IX,
    )
    rate = f(EXAMPLE_CONC, f_input)
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_reversible_michaelis_menten():
    expected_rate = 0.03027414
    f = AllostericReversibleMichaelisMenten(
        ix_ki_species=np.array([], dtype=np.int16),
        ix_allosteric_activators=np.array([2], dtype=np.int16),
        subunits=1,
    )
    f_input = f.get_input(
        parameters=EXAMPLE_PARAMETERS,
        reaction_id="r1",
        reaction_stoichiometry=EXAMPLE_S[:, 0],
        species_to_dgf_ix=EXAMPLE_SPECIES_TO_DGF_IX,
    )
    rate = f(EXAMPLE_CONC, f_input)
    assert jnp.isclose(rate, expected_rate)
