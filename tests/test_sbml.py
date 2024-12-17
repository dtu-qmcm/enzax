from jax import numpy as jnp
import pytest
import importlib.resources
from tests import data
from enzax import sbml
from enzax.steady_state import get_kinetic_model_steady_state

brusselator_file = importlib.resources.files(data) / "brusselator.xml"
exampleode_file = importlib.resources.files(data) / "exampleode.xml"


@pytest.mark.parametrize(
    "file_path",
    [
        brusselator_file,
        exampleode_file,
    ],
)
def test_load_libsbml_model(file_path):
    sbml.load_libsbml_model(file_path)


@pytest.mark.parametrize(
    ["path", "expected", "guess"],
    [
        (
            exampleode_file,
            jnp.array([0.78276539, 3.65098512]),
            jnp.array([0.01, 0.01]),
        ),
    ],
)
def test_sbml_to_enzax(path, expected, guess):
    libsbml_model = sbml.load_libsbml_model(path)
    model = sbml.sbml_to_enzax(libsbml_model)
    steady_state = get_kinetic_model_steady_state(model, guess)
    assert jnp.isclose(steady_state, expected).all()
