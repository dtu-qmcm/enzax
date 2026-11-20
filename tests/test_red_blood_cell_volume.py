"""Tests of the red blood cell examples with a variable cell volume.

As in `test_red_blood_cell.py`, the steady states are the stored ones in
`data/red_blood_cell_steady_states.json`.
"""

import json
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

from enzax.examples import red_blood_cell_volume as rbcv
from enzax.thermodynamics import MEMBRANE_POTENTIAL

jax.config.update("jax_enable_x64", True)

STATES = {
    name: jnp.array(state)
    for name, state in json.loads(
        (
            Path(__file__).parent / "data" / "red_blood_cell_steady_states.json"
        ).read_text(),
    ).items()
}


def get_volume_and_donnan_ratio(model, state, parameters):
    volume, membrane_potential = model.get_algebraic_variables(
        state,
        parameters,
    )
    return volume, rbcv.get_donnan_ratio(
        membrane_potential,
        parameters["temperature"],
    )


def test_the_variables_are_the_volume_and_the_membrane_potential():
    assert rbcv.faithful.algebraic_variables == [
        "volume|c",
        MEMBRANE_POTENTIAL,
    ]
    assert rbcv.faithful.volume_variables == {"volume|c": "c"}


def test_the_membrane_potential_is_not_a_parameter():
    labelling = rbcv.faithful.parameter_labelling
    assert MEMBRANE_POTENTIAL not in labelling
    assert "cu|leak_Na|r" not in labelling["log_custom"]
    assert "cu|leak_Na|k_perm" in labelling["log_custom"]


def test_the_intracellular_anions_follow_the_donnan_ratio():
    model, parameters = rbcv.faithful, rbcv.faithful_parameters
    _, donnan_ratio = get_volume_and_donnan_ratio(
        model,
        rbcv.faithful_initial_state,
        parameters,
    )
    conc = model.get_balanced_conc(rbcv.faithful_initial_state, parameters)
    for anion, plasma in rbcv.PLASMA_ANIONS.items():
        position = model.balanced_species.index(anion.removesuffix("_e"))
        assert np.isclose(conc[position], donnan_ratio * plasma, rtol=1e-10)


def test_the_faithful_constraints_at_the_reference_composition():
    volume, donnan_ratio = get_volume_and_donnan_ratio(
        rbcv.faithful,
        rbcv.faithful_initial_state,
        rbcv.faithful_parameters,
    )
    assert np.isclose(volume, 0.94522, atol=1e-5)
    assert np.isclose(donnan_ratio, 0.61500, atol=1e-5)


def test_the_stored_faithful_steady_state():
    model, parameters = rbcv.faithful, rbcv.faithful_parameters
    steady = STATES["faithful"]
    assert jnp.max(jnp.abs(model.dndt(steady, parameters))) < 1e-9
    volume, donnan_ratio = get_volume_and_donnan_ratio(
        model,
        steady,
        parameters,
    )
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


def test_the_stored_improved_steady_state():
    model, parameters = rbcv.improved, rbcv.improved_parameters
    steady = STATES["improved"]
    assert jnp.max(jnp.abs(model.dndt(steady, parameters))) < 1e-9
