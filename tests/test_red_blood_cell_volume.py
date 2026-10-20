import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.examples import red_blood_cell_volume as rbcv
from enzax.steady_state import get_steady_state_dae, get_steady_state_hybrid

jax.config.update("jax_enable_x64", True)


def test_the_variables_are_the_volume_and_the_donnan_ratio():
    assert rbcv.faithful.algebraic_variables == ["volume|c", "r"]
    assert rbcv.faithful.volume_variables == {"volume|c": "c"}


def test_the_leak_reads_the_donnan_ratio_as_a_variable():
    labels = rbcv.faithful.parameter_labelling["log_custom"]
    assert "cu|leak_Na|r" not in labels
    assert "cu|leak_Na|k_perm" in labels


def test_the_faithful_constraints_at_the_reference_composition():
    volume, donnan_ratio = rbcv.faithful.get_algebraic_variables(
        rbcv.faithful_initial_state,
        rbcv.faithful_parameters,
    )
    assert np.isclose(volume, 0.94522, atol=1e-5)
    assert np.isclose(donnan_ratio, 0.61500, atol=1e-5)


@pytest.mark.slow
def test_the_faithful_steady_state():
    pytest.importorskip("diffrax_bdf")
    model, parameters = rbcv.faithful, rbcv.faithful_parameters
    by_hybrid = get_steady_state_hybrid(
        model,
        rbcv.faithful_initial_state,
        parameters,
    )
    by_dae = get_steady_state_dae(
        model,
        rbcv.faithful_initial_state,
        parameters,
    )
    assert np.allclose(by_dae, by_hybrid, rtol=1e-6)
    assert jnp.max(jnp.abs(model.dndt(by_hybrid, parameters))) < 1e-9
    volume, donnan_ratio = model.get_algebraic_variables(by_hybrid, parameters)
    assert 0.9 < volume < 1.1
    assert 0.55 < donnan_ratio < 0.70


def test_the_improved_volume_is_one_at_the_reference_composition():
    volume, _ = rbcv.improved.get_algebraic_variables(
        rbcv.improved_initial_state,
        rbcv.improved_parameters,
    )
    assert np.isclose(volume, 1.0, atol=1e-8)


def test_the_improved_osmotic_sum_counts_every_intracellular_species():
    osmotic = rbcv.improved.algebraic_constraints["osmotic"]
    in_cell = rbcv.improved.compartments["c"].species
    assert set(osmotic.species) == set(in_cell) - {"k"}


@pytest.mark.slow
def test_the_improved_steady_state():
    pytest.importorskip("diffrax_bdf")
    model, parameters = rbcv.improved, rbcv.improved_parameters
    by_hybrid = get_steady_state_hybrid(
        model,
        rbcv.improved_initial_state,
        parameters,
    )
    by_dae = get_steady_state_dae(
        model,
        rbcv.improved_initial_state,
        parameters,
    )
    assert np.allclose(by_dae, by_hybrid, rtol=1e-6)
    assert jnp.max(jnp.abs(model.dndt(by_hybrid, parameters))) < 1e-9
