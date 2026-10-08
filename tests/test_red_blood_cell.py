"""Tests of the Joshi and Palsson red blood cell example."""

import numpy as np
import pytest
from jax import numpy as jnp

from enzax.examples import red_blood_cell as rbc
from enzax.steady_state import get_steady_state_dae, get_steady_state_hybrid

CALIBRATED = [
    "HK", "PFK", "PK", "TPI", "TKI", "TKII", "TA", "PRM", "PNPase",
]  # fmt: skip


def get_fluxes(state):
    conc = rbc.model.get_balanced_conc(state, rbc.parameters)
    fluxes = rbc.model.flux(conc, rbc.parameters)
    return dict(zip(rbc.model.reaction_ids, fluxes.tolist()))


def test_the_ode_state_is_joshi_and_palssons_minus_potassium():
    assert len(rbc.model.ode_state_species) == 32
    assert "k" in rbc.model.unbalanced_species


def test_mg_binding_leaves_about_a_third_of_a_millimolar_free():
    conc = rbc.model.get_balanced_conc(rbc.steady_state, rbc.parameters)
    free_mg = conc[rbc.model.balanced_species.index("mg")]
    assert np.isclose(free_mg, 0.330, atol=1e-3)


def test_calibrated_fluxes_match_part_iv_at_its_steady_state():
    fluxes = get_fluxes(rbc.steady_state)
    for reaction_id in CALIBRATED:
        assert np.isclose(
            fluxes[reaction_id],
            rbc.STEADY_STATE_FLUXES[reaction_id],
        )
    assert np.isclose(fluxes["GSSGR"], fluxes["GSHox"] / 2, rtol=1e-3)
    assert np.isclose(fluxes["LAC_ex"], 2.16, rtol=1e-3)
    assert np.isclose(fluxes["PYR_ex"], 0.0, atol=1e-3)


@pytest.mark.slow
def test_the_steady_state_is_close_to_part_iv():
    steady = get_steady_state_hybrid(
        rbc.model,
        rbc.steady_state,
        rbc.parameters,
    )
    assert jnp.max(jnp.abs(rbc.model.dcdt(steady, rbc.parameters))) < 1e-9
    assert np.allclose(steady, rbc.steady_state, rtol=0.08)
    fluxes = get_fluxes(steady)
    for reaction_id, expected in rbc.STEADY_STATE_FLUXES.items():
        if reaction_id == "ApK":
            continue
        assert np.isclose(fluxes[reaction_id], expected, rtol=0.06)


def test_rapid_equilibria_shrink_the_ode_state():
    assert len(rbc.rapid_equilibrium_model.ode_state_species) == 22


def test_every_fast_moiety_has_non_negative_coefficients():
    matrix = rbc.rapid_equilibrium_model.fast_moieties.fast_moiety_matrix
    assert np.all(matrix >= 0)


@pytest.mark.slow
def test_the_stiff_formulation_converges_to_the_rapid_equilibrium_one():
    model, parameters = (
        rbc.rapid_equilibrium_model,
        rbc.rapid_equilibrium_parameters,
    )
    steady = get_steady_state_hybrid(
        model,
        rbc.rapid_equilibrium_steady_state,
        parameters,
    )
    expected = model.get_balanced_conc(steady, parameters)
    part_iv = rbc.get_steady_state_conc()
    assert np.allclose(
        expected,
        [part_iv[s] for s in model.balanced_species],
        rtol=0.09,
    )
    ratios = np.array([1e3, 1e4])
    errors = []
    for ratio in ratios:
        stiff_parameters = rbc.get_parameters(rbc.model, ratio)
        stiff = get_steady_state_hybrid(
            rbc.model,
            rbc.steady_state,
            stiff_parameters,
            steady_state_rtol=1e-9,
            steady_state_atol=1e-9,
        )
        conc = rbc.model.get_balanced_conc(stiff, stiff_parameters)
        errors.append(float(jnp.max(jnp.abs(conc / expected - 1.0))))
    errors = np.array(errors)
    assert np.allclose(errors * ratios, errors[0] * ratios[0], rtol=0.05)
    assert errors[-1] < 1e-3


@pytest.mark.slow
def test_the_dae_steady_state_matches_the_hybrid_one_from_a_poor_start():
    pytest.importorskip("diffrax_bdf")
    model, parameters = (
        rbc.rapid_equilibrium_model,
        rbc.rapid_equilibrium_parameters,
    )
    hybrid = get_steady_state_hybrid(
        model,
        rbc.rapid_equilibrium_steady_state,
        parameters,
    )
    dae = get_steady_state_dae(
        model,
        0.8 * rbc.rapid_equilibrium_steady_state,
        parameters,
    )
    assert np.allclose(
        model.get_balanced_conc(dae, parameters),
        model.get_balanced_conc(hybrid, parameters),
        rtol=1e-6,
        atol=0.0,
    )
