"""Unit tests for kinetic models."""

import re
import warnings

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import (
    KineticModel,
    UndeclaredMoietyWarning,
    validate_kinetic_model,
)
from enzax.parameters import pack_parameters
from enzax.reactions import MichaelisMenten
from enzax.steady_state import get_steady_state_hybrid


def get_model(stoichiometry, balanced_species, moiety_pivot_species):
    """Make a model with no rate equations, for testing structure only."""
    return KineticModel(
        balanced_species=balanced_species,
        moiety_pivot_species=moiety_pivot_species,
        reactions={
            reaction: MichaelisMenten(stoichiometry=coefficients)
            for reaction, coefficients in stoichiometry.items()
        },
    )


# A <-> B, so B's stoichiometry is minus A's and the two are in a
# conservation relation.
CYCLE = dict(
    stoichiometry={"f": {"A": -1.0, "B": 1.0}, "b": {"A": 1.0, "B": -1.0}},
    balanced_species=["A", "B"],
)
# A and B are each consumed by their own reaction, so neither determines the
# other.
TWO_DRAINS = dict(
    stoichiometry={"ra": {"A": -1.0}, "rb": {"B": -1.0}},
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
    balanced_species=["A", "B", "X1", "X2"],
)


@pytest.mark.parametrize(
    ["structure", "moiety_pivot_species"],
    [
        (CYCLE, []),
        (CYCLE, ["B"]),
        (CYCLE, ["A"]),
        (TWO_DRAINS, []),
    ],
    ids=[
        "cycle-no-moiety-pivot-species",
        "cycle-pivot-b",
        "cycle-pivot-a",
        "unrelated-species-no-moiety-pivot-species",
    ],
)
def test_validate_kinetic_model_valid(structure, moiety_pivot_species):
    """Test that valid models pass validation."""
    model = get_model(**structure, moiety_pivot_species=moiety_pivot_species)
    assert validate_kinetic_model(model) is None


@pytest.mark.parametrize(
    ["structure", "moiety_pivot_species", "expected_msg"],
    [
        (
            dict(
                # C is a species of the model, but not a balanced one.
                stoichiometry=CYCLE["stoichiometry"] | {"make_c": {"C": 1.0}},
                balanced_species=["A", "B"],
            ),
            ["C"],
            "Moiety pivot species must be balanced species",
        ),
        (CYCLE, ["A", "B"], "must have at least one ODE state species"),
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
        "moiety-pivot-species-is-not-balanced",
        "no-ode-state-species",
        "ode-state-species-are-not-independent",
        "no-conservation-relation",
    ],
)
def test_validate_kinetic_model_invalid(
    structure, moiety_pivot_species, expected_msg
):
    """Test that invalid models are rejected when they are instantiated."""
    with pytest.raises(ValueError, match=expected_msg):
        get_model(**structure, moiety_pivot_species=moiety_pivot_species)


@pytest.mark.parametrize(
    ["structure", "moiety_pivot_species", "expected_L0"],
    [
        (CYCLE, [], np.zeros(shape=(0, 2))),
        (CYCLE, ["B"], np.array([[-1.0]])),
    ],
    ids=[
        "cycle-no-moiety-pivot-species",
        "cycle-pivot-b",
    ],
)
def test_link_matrix(structure, moiety_pivot_species, expected_L0):
    """Test that valid models get the expected link matrix."""
    model = get_model(**structure, moiety_pivot_species=moiety_pivot_species)
    assert model.L0.shape == expected_L0.shape
    assert np.allclose(model.L0, expected_L0)


# 2A <-> B, so the moiety counts B twice.
DIMER = dict(
    stoichiometry={"r": {"A": -2.0, "B": 1.0}},
    balanced_species=["A", "B"],
)
# A -> B + C, which conserves A + B and A + C, so B - C is conserved too.
SPLIT = dict(
    stoichiometry={"r": {"A": -1.0, "B": 1.0, "C": 1.0}},
    balanced_species=["A", "B", "C"],
)


@pytest.mark.parametrize(
    ["structure", "expected_msg"],
    [
        (CYCLE, "form 1 conserved moiety, A + B, but"),
        (TWO_MOIETIES, "form 2 conserved moieties, A + B and X1 + X2, but"),
        (DIMER, "form 1 conserved moiety, A + 2 B, but"),
        (SPLIT, "form 2 conserved moieties, A + C and B - C, but"),
    ],
    ids=["cycle", "two-moieties", "dimer", "split"],
)
def test_undeclared_moieties_warn(structure, expected_msg):
    with pytest.warns(UndeclaredMoietyWarning, match=re.escape(expected_msg)):
        get_model(**structure, moiety_pivot_species=[])


@pytest.mark.parametrize(
    ["structure", "moiety_pivot_species"],
    [(CYCLE, ["B"]), (TWO_MOIETIES, ["B", "X2"]), (TWO_DRAINS, [])],
    ids=["cycle-pivot-b", "two-moieties", "no-moieties"],
)
def test_declared_or_absent_moieties_do_not_warn(
    structure, moiety_pivot_species
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndeclaredMoietyWarning)
        get_model(**structure, moiety_pivot_species=moiety_pivot_species)


def test_independently_built_models_have_equal_tree_structures():
    a = get_model(**TWO_MOIETIES, moiety_pivot_species=["B", "X2"])
    b = get_model(**TWO_MOIETIES, moiety_pivot_species=["B", "X2"])
    assert jax.tree.structure(a) == jax.tree.structure(b)


def get_two_reaction_model(kcat_label):
    model = KineticModel(
        balanced_species=["a"],
        reactions={
            "r1": MichaelisMenten(stoichiometry={"x": -1.0, "a": 1.0}),
            "r2": MichaelisMenten(
                stoichiometry={"a": -1.0}, reversible=False, kcat=kcat_label
            ),
        },
    )
    labelling = model.parameter_labelling
    spec = {
        "log_saturation_constant": {
            label: 0.0 for label in labelling["log_saturation_constant"]
        },
        "log_kcat": {label: 0.0 for label in labelling["log_kcat"]},
        "log_enzyme": {label: 0.0 for label in labelling["log_enzyme"]},
        "dgf": {"x": -5.0, "a": -3.0},
        "log_conc_unbalanced": {"x": 0.0},
        "temperature": 310.0,
    }
    return model, pack_parameters(labelling, spec)


def test_differently_structured_models_share_a_jitted_solver():
    steady_states = []
    for kcat_label in ["r2", "k2"]:
        model, parameters = get_two_reaction_model(kcat_label)
        steady_states.append(
            get_steady_state_hybrid(model, jnp.array([0.1]), parameters)
        )
    assert jnp.allclose(steady_states[0], steady_states[1])
