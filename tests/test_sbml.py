import importlib.resources
import json

import jax
import libsbml
import pytest
from jax import numpy as jnp

from enzax import examples, sbml
from enzax.steady_state import get_steady_state
from tests import data

jax.config.update("jax_enable_x64", True)

exampleode_file = importlib.resources.files(data) / "exampleode.xml"


@pytest.mark.parametrize(
    "file_path",
    [
        exampleode_file,
    ],
)
def test_load_libsbml_model(file_path):
    sbml.load_libsbml_model_from_file(file_path)


def load_example(name):
    return sbml.load_libsbml_model_from_file(
        importlib.resources.files(examples) / name,
    )


def test_converted_glycolysis_reproduces_the_file_fluxes():
    model, parameters = sbml.sbml_to_enzax(
        load_example("mammalian_glycolysis.xml"),
    )
    with open(
        importlib.resources.files(data) / "expected_glycolysis_flux.json",
    ) as f:
        expected = json.load(f)
    conc = jnp.array(
        [expected["initial_concentration"][s] for s in model.balanced_species],
    )
    flux = model.flux(conc, parameters)
    for position, reaction in enumerate(model.reaction_ids):
        assert jnp.isclose(
            flux[position],
            expected["flux"][reaction],
            rtol=1e-10,
            atol=0.0,
        )


def test_converted_exampleode_has_the_expected_steady_state():
    libsbml_model = sbml.load_libsbml_model_from_file(exampleode_file)
    model, parameters = sbml.sbml_to_enzax(libsbml_model)
    steady_state = get_steady_state(model, jnp.array([0.01, 0.01]), parameters)
    assert jnp.allclose(steady_state, jnp.array([0.3230166, 3.02209784]))


def test_converted_smallbone_matches_the_retired_kinetic_model_sbml():
    model, parameters = sbml.sbml_to_enzax(
        load_example("smallbone2013_model18_modified.xml"),
    )
    with open(importlib.resources.files(data) / "smallbone_fluxes.json") as f:
        expected = json.load(f)
    conc = jnp.array(
        [expected["balanced_concentration"][s] for s in model.balanced_species],
    )
    unbalanced = model.parameter_labelling["log_conc_unbalanced"]
    parameters = parameters | {
        "log_conc_unbalanced": jnp.log(
            jnp.array(
                [expected["unbalanced_concentration"][s] for s in unbalanced],
            ),
        ),
    }
    flux = model.flux(conc, parameters)
    for position, reaction in enumerate(model.reaction_ids):
        assert jnp.isclose(
            flux[position],
            expected["flux"][reaction],
            rtol=1e-10,
        )


def test_parameter_kinds_can_be_overridden():
    libsbml_model = sbml.load_libsbml_model_from_file(exampleode_file)
    model, parameters = sbml.sbml_to_enzax(libsbml_model)
    custom_model, custom_parameters = sbml.sbml_to_enzax(
        libsbml_model,
        parameter_kinds={"cu|r1|Kcat_r1": "custom"},
    )
    assert "cu|r1|Kcat_r1" in custom_model.parameter_labelling["custom"]
    conc = jnp.array([0.2, 0.3])
    assert jnp.allclose(
        custom_model.flux(conc, custom_parameters),
        model.flux(conc, parameters),
    )


def test_a_non_positive_value_cannot_be_log_custom():
    libsbml_model = sbml.load_libsbml_model_from_file(exampleode_file)
    libsbml_model = libsbml_model.clone()
    reaction = libsbml_model.getReaction("r1")
    reaction.getKineticLaw().getParameter("Kcat_r1").setValue(-1.0)
    with pytest.raises(ValueError, match="cannot be log_custom"):
        sbml.sbml_to_enzax(
            libsbml_model,
            parameter_kinds={"cu|r1|Kcat_r1": "log_custom"},
        )


def test_unsupported_sbml_is_rejected():
    libsbml_model = sbml.load_libsbml_model_from_file(exampleode_file).clone()
    libsbml_model.getCompartment("Cytosol").setSize(2.0)
    rule = libsbml_model.createRateRule()
    rule.setVariable("B")
    rule.setMath(libsbml.parseL3Formula("1"))
    with pytest.raises(ValueError, match="rate rules.*compartment 'Cytosol'"):
        sbml.sbml_to_enzax(libsbml_model)
