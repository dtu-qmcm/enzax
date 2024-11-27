"""Unit tests for rate equations."""

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Scalar
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    AllostericReversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


class ExampleParameterSet(eqx.Module):
    log_km: dict[int, Array]
    log_kcat: dict[int, Scalar]
    log_enzyme: dict[int, Array]
    log_ki: dict[int, Array]
    dgf: Array
    temperature: Scalar
    log_conc_unbalanced: Array
    log_dc_inhibitor: dict[int, Array]
    log_dc_activator: dict[int, Array]
    log_tc: dict[int, Array]


EXAMPLE_S = np.array(
    [[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]], dtype=np.float64
)
EXAMPLE_CONC = jnp.array([0.5, 0.2, 0.1, 0.3])
EXAMPLE_PARAMETERS = ExampleParameterSet(
    log_km={
        0: jnp.array([[0.1], [-0.2]]),
        1: jnp.array([[0.5], [0.0]]),
        2: jnp.array([[-1.0], [0.5]]),
    },
    log_kcat={0: jnp.array(-0.1), 1: jnp.array(0.0), 2: jnp.array(0.1)},
    dgf=jnp.array([-3.0, 1.0]),
    log_ki={0: jnp.array([1.0]), 1: jnp.array([1.0]), 2: jnp.array([])},
    temperature=jnp.array(310.0),
    log_enzyme={0: jnp.array(0.3), 1: jnp.array(0.2), 2: jnp.array(0.1)},
    log_conc_unbalanced=jnp.array([0.5, 0.3]),
    log_tc={0: jnp.array(-0.2), 1: jnp.array(0.3)},
    log_dc_activator={0: jnp.array([-0.1]), 1: jnp.array([])},
    log_dc_inhibitor={0: jnp.array([]), 1: jnp.array([0.2])},
)
EXAMPLE_SPECIES_TO_DGF_IX = np.array([0, 0, 1, 1])


def test_irreversible_michaelis_menten():
    expected_rate = 0.08455524
    f = IrreversibleMichaelisMenten()
    f_input = f.get_input(
        parameters=EXAMPLE_PARAMETERS,
        rxn_ix=0,
        S=EXAMPLE_S,
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
        rxn_ix=0,
        S=EXAMPLE_S,
        species_to_dgf_ix=EXAMPLE_SPECIES_TO_DGF_IX,
    )
    rate = f(EXAMPLE_CONC, f_input)
    assert jnp.isclose(rate, expected_rate)


def test_allosteric_irreversible_michaelis_menten():
    expected_rate = 0.05608589
    f = AllostericIrreversibleMichaelisMenten(
        ix_ki_species=np.array([], dtype=np.int16),
        ix_activators=np.array([2], dtype=np.int16),
        subunits=1,
    )
    f_input = f.get_input(
        parameters=EXAMPLE_PARAMETERS,
        rxn_ix=0,
        S=EXAMPLE_S,
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
        rxn_ix=0,
        S=EXAMPLE_S,
        species_to_dgf_ix=EXAMPLE_SPECIES_TO_DGF_IX,
    )
    rate = f(EXAMPLE_CONC, f_input)
    assert jnp.isclose(rate, expected_rate)
