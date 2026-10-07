from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import pytest

from enzax.algebraic_constraint import (
    AlgebraicConstraint,
    ConstraintLabels,
    ConstraintScope,
)
from enzax.kinetic_model import KineticModel
from enzax.parameters import get_parameter_position
from enzax.reactions import Drain


@dataclass(frozen=True)
class CubicLabels(ConstraintLabels):
    k: str

    def by_parameter(self):
        return {"log_custom": (self.k,)}


class CubicIx(eqx.Module):
    ix_species: int
    ix_variable: int
    ix_k: int


class CubicInput(eqx.Module):
    k: jnp.ndarray
    ix_species: int
    ix_variable: int


class Cubic(AlgebraicConstraint):
    species_id: str
    n_residuals: int = 1

    def get_species(self):
        return (self.species_id,)

    def get_labels(self, scope: ConstraintScope) -> CubicLabels:
        return CubicLabels(k=f"cu|{scope.constraint_id}|k")

    def get_input_indexes(self, scope, labelling) -> CubicIx:
        return CubicIx(
            ix_species=scope.species.index(self.species_id),
            ix_variable=scope.algebraic_variables.index(self.variables[0]),
            ix_k=get_parameter_position(
                labelling, "log_custom", self.get_labels(scope).k
            ),
        )

    def get_input(self, parameters, ix: CubicIx) -> CubicInput:
        return CubicInput(
            k=jnp.exp(parameters["log_custom"][ix.ix_k]),
            ix_species=ix.ix_species,
            ix_variable=ix.ix_variable,
        )

    def __call__(self, conc, variables, constraint_input: CubicInput):
        a = conc[constraint_input.ix_species]
        x = variables[constraint_input.ix_variable]
        return jnp.full(self.n_residuals, x**3 + x - constraint_input.k * a)


class ReadsAnother(Cubic):
    other: str = "y"

    def get_variables_read(self):
        return (*self.variables, self.other)


IN_AND_OUT = {
    "r1": Drain(stoichiometry={"a": 1.0}),
    "r2": Drain(stoichiometry={"a": -1.0}),
}


def get_model(**constraints):
    return KineticModel(
        reactions=IN_AND_OUT,
        balanced_species=["a"],
        algebraic_constraints=constraints,
    )


def test_a_model_without_constraints_has_no_algebraic_variables():
    model = get_model()
    assert model.algebraic_variables == []
    assert model.constraint_ix == []


def test_constraints_own_the_model_algebraic_variables():
    model = get_model(
        c1=Cubic(species_id="a", variables=["x"]),
        c2=Cubic(species_id="a", variables=["z"]),
    )
    assert model.algebraic_variables == ["x", "z"]


def test_constraint_labels_join_the_parameter_labelling():
    model = get_model(c1=Cubic(species_id="a", variables=["x"]))
    assert model.parameter_labelling["log_custom"] == ("cu|c1|k",)


def test_a_species_only_a_constraint_names_comes_last():
    model = get_model(c1=Cubic(species_id="b", variables=["x"]))
    assert model.species == ["a", "b"]
    assert model.unbalanced_species == ["b"]


def test_constraint_ids_must_not_clash_with_reactions():
    with pytest.raises(ValueError, match="same ids as reactions"):
        get_model(r1=Cubic(species_id="a", variables=["x"]))


def test_a_variable_belongs_to_one_constraint():
    with pytest.raises(ValueError, match="belong to several"):
        get_model(
            c1=Cubic(species_id="a", variables=["x"]),
            c2=Cubic(species_id="a", variables=["x"]),
        )


def test_a_variable_cannot_be_named_like_a_species():
    with pytest.raises(ValueError, match="same names as species"):
        get_model(c1=Cubic(species_id="a", variables=["a"]))


def test_a_constraint_cannot_read_a_variable_no_constraint_has():
    with pytest.raises(ValueError, match=r"reads algebraic variables \['y'\]"):
        get_model(c1=ReadsAnother(species_id="a", variables=["x"]))


def test_a_constraint_needs_one_residual_per_variable():
    with pytest.raises(ValueError, match=r"returns an array of shape \(2,\)"):
        get_model(c1=Cubic(species_id="a", variables=["x"], n_residuals=2))


def test_variable_names_cannot_contain_the_separator():
    with pytest.raises(ValueError, match="Algebraic variable id"):
        get_model(c1=Cubic(species_id="a", variables=["x|y"]))
