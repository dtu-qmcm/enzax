"""Unit tests for kinetic models."""

import numpy as np
import pytest

from enzax.kinetic_model import RateEquationModel, validate_kinetic_model


def get_model(stoichiometry, species, balanced_species, dependent_species):
    """Make a model with no rate equations, for testing structure only."""
    return RateEquationModel(
        stoichiometry=stoichiometry,
        species=species,
        reactions=list(stoichiometry.keys()),
        balanced_species=balanced_species,
        dependent_species=dependent_species,
    )


# A <-> B, so B's stoichiometry is minus A's and the two are in a
# conservation relation.
CYCLE = dict(
    stoichiometry={"f": {"A": -1.0, "B": 1.0}, "b": {"A": 1.0, "B": -1.0}},
    species=["A", "B"],
    balanced_species=["A", "B"],
)
# A and B are each consumed by their own reaction, so neither determines the
# other.
TWO_DRAINS = dict(
    stoichiometry={"ra": {"A": -1.0}, "rb": {"B": -1.0}},
    species=["A", "B"],
    balanced_species=["A", "B"],
)
# A cofactor X1/X2 is recycled while A is turned into B, so the model has two
# separate conservation relations: A + B and X1 + X2. Exactly one species from
# each relation can be independent.
TWO_MOIETIES = dict(
    stoichiometry={
        "r": {"A": -1.0, "X1": -1.0, "B": 1.0, "X2": 1.0},
        "regen": {"X2": -1.0, "X1": 1.0},
    },
    species=["A", "B", "X1", "X2"],
    balanced_species=["A", "B", "X1", "X2"],
)


@pytest.mark.parametrize(
    ["structure", "dependent_species"],
    [
        (CYCLE, []),
        (CYCLE, ["B"]),
        (CYCLE, ["A"]),
        (TWO_DRAINS, []),
    ],
    ids=[
        "cycle-no-dependent-species",
        "cycle-dependent-b",
        "cycle-dependent-a",
        "unrelated-species-no-dependent-species",
    ],
)
def test_validate_kinetic_model_valid(structure, dependent_species):
    """Test that valid models pass validation."""
    model = get_model(**structure, dependent_species=dependent_species)
    assert validate_kinetic_model(model) is None


@pytest.mark.parametrize(
    ["structure", "dependent_species", "expected_msg"],
    [
        (
            dict(
                stoichiometry=CYCLE["stoichiometry"],
                species=["A", "B", "C"],
                balanced_species=["A", "B"],
            ),
            ["C"],
            "Dependent species must be balanced species",
        ),
        (CYCLE, ["A", "B"], "must have at least one independent species"),
        (
            TWO_MOIETIES,
            ["X2"],
            "stoichiometries must be linearly independent",
        ),
        (
            TWO_DRAINS,
            ["B"],
            "must take part in a conservation relation",
        ),
    ],
    ids=[
        "dependent-species-is-not-balanced",
        "no-independent-species",
        "independent-species-are-not-independent",
        "no-conservation-relation",
    ],
)
def test_validate_kinetic_model_invalid(
    structure, dependent_species, expected_msg
):
    """Test that invalid models are rejected when they are instantiated."""
    with pytest.raises(ValueError, match=expected_msg):
        get_model(**structure, dependent_species=dependent_species)


@pytest.mark.parametrize(
    ["structure", "dependent_species", "expected_L0"],
    [
        (CYCLE, [], np.zeros(shape=(0, 2))),
        (CYCLE, ["B"], np.array([[-1.0]])),
    ],
    ids=[
        "cycle-no-dependent-species",
        "cycle-dependent-b",
    ],
)
def test_link_matrix(structure, dependent_species, expected_L0):
    """Test that valid models get the expected link matrix."""
    model = get_model(**structure, dependent_species=dependent_species)
    assert model.L0.shape == expected_L0.shape
    assert np.allclose(model.L0, expected_L0)
