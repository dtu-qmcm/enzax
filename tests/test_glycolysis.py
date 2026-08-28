"""Tests that the glycolysis example reproduces the model it came from.

`data/expected_glycolysis_flux.json` holds the initial concentrations of
`mammalian_glycolysis.xml` and the flux its own rate laws give there,
evaluated from the SBML function definitions rather than through enzax. Every
reaction is expected to agree, apart from the two the example's docstring
names.

Those two are also where the julia implementation of the same model
(`cho_steady_state_fluxes.json` came from it) sides with one of us: it
normalises HEX1's glucose by glucose's own constant, as we do, and takes ENO's
standard free energy change from 3-phosphoglycerate, as the SBML does. Against
julia's rate laws at the same parameter values, 24 of our 26 reactions agree to
better than 1e-12; the exceptions are ENO at 5e-4 and glucose transport, which
julia holds at zero.
"""

import json
from pathlib import Path

import jax
import pytest
from jax import numpy as jnp

from enzax.examples import glycolysis
from enzax.parameters import pack_parameters

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
expected_flux_file = HERE / "data" / "expected_glycolysis_flux.json"

# The one reaction enzax does not reproduce, and by how much: HEX1 divides
# glucose by its own Michaelis constant rather than by ATP's, which the model's
# write-up calls a correction.
KNOWN_DIFFERENT = {"HEX1": 0.0568}


def get_expected():
    with open(expected_flux_file, "r") as f:
        return json.load(f)


def get_julia(filename: str):
    """Get the CHO-S wild type block of one of the julia model's outputs."""
    with open(Path(glycolysis.__file__).parent / filename, "r") as f:
        return json.load(f)["lines"]["CHO-S wt"]


def get_sbml_parameters():
    """Pack the SBML file's own parameter values, which the example does not.

    The example carries a fit of this model to CHO data instead, so the values
    the file itself ships live here, next to the fluxes they produce.
    """
    with open(HERE / "data" / "sbml_glycolysis_parameters.json", "r") as f:
        return pack_parameters(
            glycolysis.model.parameter_labelling, json.load(f)
        )


def get_initial_conc(expected) -> jnp.ndarray:
    """Get the SBML file's initial concentrations, in the model's order."""
    conc = expected["initial_concentration"]
    return jnp.array([conc[species] for species in glycolysis.species])


def get_flux_at_initial_conc(expected) -> jnp.ndarray:
    model = glycolysis.model
    conc = get_initial_conc(expected)
    return model.flux(conc[model.balanced_species_ix], get_sbml_parameters())


def test_the_model_has_the_shape_of_the_sbml_file():
    model = glycolysis.model
    assert len(model.species) == 31
    # Cytosolic glucose is balanced in the SBML file and fixed here, as it is
    # in the fitted julia version of the same model.
    assert len(model.balanced_species) == 18
    assert len(model.unbalanced_species) == 13
    assert len(model.reactions) == 26
    # Glucose and lactate are the two compounds in two compartments each.
    assert len(model.parameter_labelling["dgf"]) == 29


def test_transketolase_is_one_enzyme_with_two_turnover_numbers():
    labelling = glycolysis.model.parameter_labelling
    assert labelling["log_enzyme"].count("TKT") == 1
    assert "TKT1" not in labelling["log_enzyme"]
    assert "TKT1" in labelling["log_kcat"]
    assert "TKT2" in labelling["log_kcat"]
    # Both reactions read the same Michaelis constant for xylulose-5-phosphate.
    assert labelling["log_k"].count("km|TKT|xu5p_c") == 1


def test_transport_reactions_have_no_standard_free_energy_change():
    """Glucose is one compound, so `dgf_glc - dgf_glc` cancels by itself."""
    ix = glycolysis.model.rate_equation_ix
    for reaction in ["GLUT4", "lac_transport"]:
        position = glycolysis.reactions.index(reaction)
        dgf_positions = ix[position].ix_dgf
        assert len(set(dgf_positions.tolist())) == 1


@pytest.mark.parametrize("reaction", [r for r in glycolysis.reactions])
def test_flux_matches_the_sbml_rate_laws(reaction):
    expected = get_expected()
    flux = get_flux_at_initial_conc(expected)
    position = glycolysis.reactions.index(reaction)
    from_sbml = expected["flux"][reaction]
    relative_difference = abs(float(flux[position]) - from_sbml) / abs(
        from_sbml
    )
    if reaction in KNOWN_DIFFERENT:
        assert relative_difference == pytest.approx(
            KNOWN_DIFFERENT[reaction], rel=0.05
        )
    else:
        assert relative_difference < 1e-9


def test_the_steady_state_carries_glycolytic_flux():
    """The pathway runs forwards, drains included."""
    model = glycolysis.model
    moiety_totals = model.get_moiety_totals(glycolysis.parameters)
    conc = model.get_balanced_conc(glycolysis.steady_state, moiety_totals)
    flux = model.flux(conc, glycolysis.parameters)
    forwards = ["GLUT4", "PGI", "GAPD", "ENO", "LDHA", "G6PDH"]
    for reaction in forwards:
        assert flux[glycolysis.reactions.index(reaction)] > 0.0


def test_the_stoichiometry_balances_an_independent_implementation():
    """The julia version's steady-state fluxes are steady under our `S` too.

    `cho_steady_state_fluxes.json` comes from a separately written version of
    this model, at parameter values fitted to CHO data rather than the ones in
    the SBML file. The fluxes therefore do not match ours, but they must
    satisfy the same mass balance, which makes them a check on the
    stoichiometry that owes nothing to any parameter value. Glucose is left
    out because that model holds it fixed: its glucose transport flux is zero
    and cytosolic glucose is not one of its 18 states.
    """
    here = Path(glycolysis.__file__).parent
    flux_file = here / "cho_steady_state_fluxes.json"
    state_file = here / "cho_steady_state.json"
    with open(flux_file, "r") as f:
        flux = json.load(f)["lines"]["CHO-S wt"]["flux"]
    with open(state_file, "r") as f:
        balanced = json.load(f)["lines"]["CHO-S wt"]["concentration"]
    v = jnp.array([flux[reaction] for reaction in glycolysis.reactions])
    dcdt = glycolysis.model.S @ v
    for species, rate in zip(glycolysis.species, dcdt):
        if species in balanced:
            assert abs(rate) < 1e-9 * jnp.abs(v).max()


def test_the_fitted_model_reproduces_julias_fluxes():
    """At the fitted parameters, our fluxes are the julia version's.

    Glucose transport is the one reaction left out: the julia model holds it
    at exactly zero, and here it is computed and inert, since both glucose
    pools are fixed.
    """
    expected = get_julia("cho_steady_state_fluxes.json")["flux"]
    flux = glycolysis.model.flux(glycolysis.steady_state, glycolysis.parameters)
    for position, reaction in enumerate(glycolysis.reactions):
        if reaction == "GLUT4":
            continue
        assert float(flux[position]) == pytest.approx(
            expected[reaction], rel=1e-9, abs=1e-30
        )


def test_julias_steady_state_is_ours():
    """Their steady state is the example's, and is steady for us too."""
    state = get_julia("cho_steady_state.json")["concentration"]
    conc = glycolysis.steady_state
    assert jnp.array_equal(
        conc,
        jnp.array([state[s] for s in glycolysis.model.independent_species]),
    )
    dcdt = glycolysis.model.dcdt(conc, glycolysis.parameters)
    assert (jnp.abs(dcdt / conc) < 1e-9).all()
