import pytest
import importlib.resources
from tests import data
from enzax import sbml

brusselator_file = importlib.resources.files(data) / "brusselator.xml"
exampleode_file = importlib.resources.files(data) / "exampleode.xml"


@pytest.mark.parametrize(
    "file_path",
    [
        brusselator_file,
        exampleode_file,
    ],
)
def test_load_sbml(file_path):
    sbml.load_sbml(file_path)


@pytest.mark.parametrize(
    "model",
    [
        sbml.load_sbml(brusselator_file),
        sbml.load_sbml(exampleode_file),
    ],
)
def test_sbml_to_sympy(model):
    sbml.sbml_to_sympy(model)


def test_sympy_to_enzax(): ...
