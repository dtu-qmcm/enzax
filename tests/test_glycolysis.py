"""Tests that the glycolysis example reproduces the model it came from.

`data/expected_glycolysis_flux.json` holds the initial concentrations of
`mammalian_glycolysis.xml` and the flux its own rate laws give there,
evaluated from the SBML function definitions rather than through enzax. Every
reaction is expected to agree, apart from the three the example's docstring
names.
"""

import json
from pathlib import Path

import jax
import pytest
from jax import numpy as jnp

from enzax.examples import glycolysis

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
expected_flux_file = HERE / "data" / "expected_glycolysis_flux.json"

# The three reactions enzax deliberately does not reproduce, and by how much.
# HEX1 divides glucose by its own Michaelis constant rather than by ATP's; ENO
# gets its standard free energy change from its stoichiometry, so it uses
# phosphoenolpyruvate's formation energy where the SBML uses
# 3-phosphoglycerate's; and PGL feels enzax's hardcoded formation energy of
# water, which is -150.9 against the SBML's -154.4.
KNOWN_DIFFERENT = {"HEX1": 0.0568, "ENO": 1.2077, "PGL": 0.019}


def get_expected():
    with open(expected_flux_file, "r") as f:
        return json.load(f)


def get_initial_conc(expected) -> jnp.ndarray:
    """Get the SBML file's initial concentrations, in the model's order."""
    conc = expected["initial_concentration"]
    return jnp.array([conc[species] for species in glycolysis.species])


def get_flux_at_initial_conc(expected) -> jnp.ndarray:
    model = glycolysis.model
    conc = get_initial_conc(expected)
    return model.flux(conc[model.balanced_species_ix], glycolysis.parameters)


def test_the_model_has_the_shape_of_the_sbml_file():
    model = glycolysis.model
    assert len(model.species) == 31
    assert len(model.balanced_species) == 19
    assert len(model.unbalanced_species) == 12
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
