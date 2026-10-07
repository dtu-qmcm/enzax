from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from enzax.algebraic_constraint import (
    AlgebraicConstraint,
    ConstraintLabels,
    ConstraintScope,
)
from enzax.kinetic_model import KineticModel
from enzax.parameters import get_parameter_position, pack_parameters
from enzax.rapid_equilibrium import (
    RapidEquilibriumNetwork,
    RapidEquilibriumReaction,
)
from enzax.reactions import Drain, SymbolicReaction
from enzax.steady_state import get_steady_state_dae, get_steady_state_hybrid


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


# Toy 1: a drain feeds a, which leaves at k2 * x * a, where x is fixed by the
# cubic constraint x**3 + x = kc * a. At steady state,
# v = k2 * (x**4 + x**2) / kc.
OUTFLOW = SymbolicReaction(
    stoichiometry={"a": -1.0},
    expression="k * x * a",
    species=["a"],
    parameters={"k": "log_kcat"},
    algebraic_variables=["x"],
)
TOY = KineticModel(
    reactions={"r1": Drain(stoichiometry={"a": 1.0}), "r2": OUTFLOW},
    balanced_species=["a"],
    algebraic_constraints={"c1": Cubic(species_id="a", variables=["x"])},
)
V, K2, KC = 2.0, 0.5, 3.0


def get_toy_parameters(log_v=np.log(V)):
    return pack_parameters(
        TOY.parameter_labelling,
        {
            "log_drain": {"r1": log_v},
            "log_kcat": {"r2": np.log(K2)},
            "log_custom": {"cu|c1|k": np.log(KC)},
            "dgf": {"a": 0.0},
            "temperature": 298.15,
        },
    )


def get_toy_steady_state():
    x = brentq(lambda x: K2 * (x**4 + x**2) / KC - V, 0.1, 10.0)
    a = (x**3 + x) / KC
    dx_dlog_v = V / (K2 * (4 * x**3 + 2 * x) / KC)
    da_dlog_v = (3 * x**2 + 1) / KC * dx_dlog_v
    return a, x, da_dlog_v


def test_the_nested_solve_satisfies_the_constraint():
    parameters = get_toy_parameters()
    (x,) = TOY.get_algebraic_variables(jnp.array([1.5]), parameters)
    assert np.isclose(x**3 + x, KC * 1.5, rtol=1e-10)


def test_dcdt_reads_the_algebraic_variable():
    parameters = get_toy_parameters()
    (x,) = TOY.get_algebraic_variables(jnp.array([1.5]), parameters)
    assert np.isclose(
        TOY.dcdt(jnp.array([1.5]), parameters)[0], V - K2 * x * 1.5
    )


def test_a_rate_law_cannot_read_a_variable_no_constraint_has():
    with pytest.raises(ValueError, match="reads algebraic variable 'x'"):
        KineticModel(
            reactions={"r1": Drain(stoichiometry={"a": 1.0}), "r2": OUTFLOW},
            balanced_species=["a"],
        )


def test_a_symbol_cannot_be_both_a_species_and_a_variable():
    with pytest.raises(ValueError, match="both species and algebraic"):
        KineticModel(
            reactions={
                "r1": Drain(stoichiometry={"a": 1.0}),
                "r2": SymbolicReaction(
                    stoichiometry={"a": -1.0},
                    expression="k * a",
                    species=["a"],
                    parameters={"k": "log_kcat"},
                    algebraic_variables={"a": "x"},
                ),
            },
            balanced_species=["a"],
            algebraic_constraints={
                "c1": Cubic(species_id="a", variables=["x"])
            },
        )


def test_the_nested_steady_state_matches_the_closed_form():
    a, _, _ = get_toy_steady_state()
    steady = get_steady_state_hybrid(
        TOY, jnp.array([1.0]), get_toy_parameters()
    )
    assert np.isclose(steady[0], a, rtol=1e-8)


@pytest.mark.parametrize("suppress_algebraic_error", [True, False])
def test_the_dae_steady_state_matches_the_closed_form(suppress_algebraic_error):
    pytest.importorskip("diffrax_bdf")
    a, x, _ = get_toy_steady_state()
    parameters = get_toy_parameters()
    steady = get_steady_state_dae(
        TOY,
        jnp.array([1.0]),
        parameters,
        suppress_algebraic_error=suppress_algebraic_error,
    )
    assert np.isclose(steady[0], a, rtol=1e-7)
    assert np.isclose(TOY.get_algebraic_variables(steady, parameters)[0], x)


@pytest.mark.parametrize("solve", ["hybrid", "dae"])
def test_steady_state_gradients_match_the_closed_form(solve):
    if solve == "dae":
        pytest.importorskip("diffrax_bdf")
    steady_state = {
        "hybrid": get_steady_state_hybrid,
        "dae": get_steady_state_dae,
    }[solve]
    _, _, da_dlog_v = get_toy_steady_state()

    def steady_a(log_v):
        return steady_state(TOY, jnp.array([1.0]), get_toy_parameters(log_v))[0]

    assert np.isclose(jax.grad(steady_a)(np.log(V)), da_dlog_v, rtol=1e-6)


# Toy 2: toy 1, but m binds a rapidly, the outflow and the constraint read
# free a, and m is conserved.
BIND_M = RapidEquilibriumNetwork(
    reactions={
        "f1": RapidEquilibriumReaction(
            stoichiometry={"a": -1.0, "m": -1.0, "am": 1.0}
        )
    }
)
TOY_WITH_BINDING = KineticModel(
    reactions={"r1": Drain(stoichiometry={"a": 1.0}), "r2": OUTFLOW},
    rapid_equilibrium_network=BIND_M,
    balanced_species=["a", "m", "am"],
    moiety_label_species=["m"],
    algebraic_constraints={"c1": Cubic(species_id="a", variables=["x"])},
)


def get_binding_parameters(log_v=np.log(V)):
    return pack_parameters(
        TOY_WITH_BINDING.parameter_labelling,
        {
            "log_drain": {"r1": log_v},
            "log_kcat": {"r2": np.log(K2)},
            "log_custom": {"cu|c1|k": np.log(KC)},
            "dgf": {"a": 0.0, "m": 0.0, "am": -5.0},
            "moiety_totals": {"m": 1.0},
            "temperature": 298.15,
        },
    )


def test_the_dae_residuals_vanish_at_a_consistent_state():
    parameters = get_binding_parameters()
    y = TOY_WITH_BINDING.get_dae_state(jnp.array([2.0]), parameters)
    rates, residuals = TOY_WITH_BINDING.dae_vector_field(0.0, y, parameters)
    assert np.allclose(
        rates, TOY_WITH_BINDING.dcdt(jnp.array([2.0]), parameters)
    )
    assert residuals["log_conc"].shape == (3,)
    assert np.allclose(residuals["log_conc"], 0.0, atol=1e-9)
    assert np.allclose(residuals["log_variables"], 0.0, atol=1e-9)


def test_with_rapid_equilibria_the_dae_matches_the_nested_solve():
    pytest.importorskip("diffrax_bdf")
    guess = jnp.array([1.0])

    def free_a(log_v, steady_state):
        parameters = get_binding_parameters(log_v)
        steady = steady_state(TOY_WITH_BINDING, guess, parameters)
        conc = TOY_WITH_BINDING.get_balanced_conc(steady, parameters)
        return conc[0]

    for solve in [get_steady_state_hybrid, get_steady_state_dae]:
        assert np.isclose(
            free_a(np.log(V), solve), get_toy_steady_state()[0], rtol=1e-7
        )
    dae = jax.grad(free_a)(np.log(V), get_steady_state_dae)
    hybrid = jax.grad(free_a)(np.log(V), get_steady_state_hybrid)
    assert np.isclose(dae, hybrid, rtol=1e-6)
