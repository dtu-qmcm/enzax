import jax
import numpy as np
import pytest
import sympy
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters
from enzax.rate_equation import ReactionScope
from enzax.rate_equations import Drain, MichaelisMenten, SymbolicRateEquation
from enzax.rate_equations.symbolic import parse_expression

jax.config.update("jax_enable_x64", True)

SCOPE = ReactionScope(
    reaction_id="r1",
    species=("a", "b", "e"),
    stoichiometry=np.array([-1.0, 1.0, 0.0]),
    species_to_dgf_ix=np.array([0, 1, 2]),
)

MM_SPECIES = {"s": "a"}
MM_PARAMETERS = {
    "kcat": "log_kcat",
    "enzyme": "log_enzyme",
    "km": {"kind": "log_saturation_constant", "label": "km|r1|a"},
}


def get_labels(expression, species=MM_SPECIES, parameters=MM_PARAMETERS):
    rate_equation = SymbolicRateEquation(
        expression=expression, species=species, parameters=parameters
    )
    return rate_equation.get_labels(SCOPE)


def test_names_sympy_treats_specially_are_plain_symbols():
    expression = parse_expression("S * E / (K + S) + exp(I)")
    assert {s.name for s in expression.free_symbols} == {"S", "E", "K", "I"}
    assert expression.has(sympy.exp)


def test_a_sympy_expression_is_used_as_it_is():
    s, k = sympy.symbols("s k")
    assert parse_expression(s / (k + s)) == s / (k + s)


def test_labels_default_by_kind():
    labels = get_labels(
        "kcat * enzyme * s / (km + s) * r * temperature",
        parameters=MM_PARAMETERS
        | {"r": "log_custom", "temperature": "temperature"},
    )
    assert labels.by_parameter() == {
        "log_kcat": ("r1",),
        "log_enzyme": ("r1",),
        "log_saturation_constant": ("km|r1|a",),
        "log_custom": ("cu|r1|r",),
        "temperature": (),
    }


def test_explicit_labels_can_be_shared():
    labels = get_labels(
        "kf * s - kr * s",
        parameters={
            "kf": {"kind": "log_kcat", "label": "k"},
            "kr": {"kind": "log_kcat", "label": "k"},
        },
    )
    assert labels.by_parameter() == {"log_kcat": ("k",)}


def test_get_species_reports_every_declared_species():
    rate_equation = SymbolicRateEquation(
        expression="s * e", species={"s": "a", "e": "e"}
    )
    assert rate_equation.get_species() == ("a", "e")


@pytest.mark.parametrize(
    ["expression", "species", "parameters", "match"],
    [
        ("kcat * s * x", MM_SPECIES, {"kcat": "log_kcat"}, "not declared"),
        ("kcat", MM_SPECIES, {"kcat": "log_kcat"}, "does not use"),
        ("s", {"s": "a"}, {"s": "log_kcat"}, "both species and parameters"),
        (
            "s * reversibility",
            {"s": "a", "reversibility": "b"},
            {},
            "reserves",
        ),
        ("s * k", MM_SPECIES, {"k": "log_bogus"}, "must be one of"),
        ("s * g", MM_SPECIES, {"g": "dgf"}, "must be one of"),
        (
            "s * km",
            MM_SPECIES,
            {"km": "log_saturation_constant"},
            "no default label",
        ),
        (
            "s * k",
            MM_SPECIES,
            {"k": {"kind": "log_kcat", "lable": "x"}},
            "optionally a",
        ),
        (
            "s * t",
            MM_SPECIES,
            {"t": {"kind": "temperature", "label": "x"}},
            "unlabelled",
        ),
        (
            "kf * s - kr * s",
            MM_SPECIES,
            {"kf": "log_kcat", "kr": "log_kcat"},
            "both default",
        ),
    ],
)
def test_bad_declarations_are_rejected(expression, species, parameters, match):
    with pytest.raises(ValueError, match=match):
        get_labels(expression, species, parameters)


CONC = jnp.array([0.5, 0.2, 0.1])
VALUES = {
    "log_saturation_constant": {"km|r1|a": 0.1, "km|r1|b": -0.2},
    "log_kcat": {"r1": -0.1},
    "log_enzyme": {"r1": jnp.log(0.3)},
    "log_drain": {"r1": jnp.log(0.7)},
    "log_custom": {"cu|r1|r": jnp.log(2.0)},
    "custom": {"cu|r1|c": -0.5},
    "dgf": {"a": -3.0, "b": -1.0, "e": 1.0},
    "temperature": 310.0,
}


def get_model_and_parameters(rate_equation):
    model = RateEquationModel(
        stoichiometry={"r1": {"a": -1.0, "b": 1.0}},
        balanced_species=["a", "b", "e"],
        extra_species=["a", "b", "e"],
        rate_equations={"r1": rate_equation},
    )
    labelling = model.parameter_labelling
    spec = {
        parameter: (
            VALUES[parameter]
            if not labels
            else {label: VALUES[parameter][label] for label in labels}
        )
        for parameter, labels in labelling.items()
    }
    return model, pack_parameters(labelling, spec)


def get_flux_and_gradient(rate_equation):
    model, parameters = get_model_and_parameters(rate_equation)

    def flux(parameters):
        return model.flux(CONC, parameters)[0]

    return flux(parameters), jax.jit(jax.grad(flux))(parameters)


def assert_same_flux_and_gradient(symbolic, built_in):
    flux, gradient = get_flux_and_gradient(symbolic)
    expected_flux, expected_gradient = get_flux_and_gradient(built_in)
    assert jnp.isclose(flux, expected_flux, rtol=1e-12)
    for parameter, expected in expected_gradient.items():
        assert jnp.allclose(gradient[parameter], expected, rtol=1e-12)


def test_agrees_with_irreversible_michaelis_menten():
    symbolic = SymbolicRateEquation(
        expression="kcat * enzyme * (s / km) / (1 + s / km)",
        species=MM_SPECIES,
        parameters=MM_PARAMETERS,
    )
    assert_same_flux_and_gradient(symbolic, MichaelisMenten(reversible=False))


def test_agrees_with_drain():
    symbolic = SymbolicRateEquation(
        expression="-v", parameters={"v": "log_drain"}
    )
    assert_same_flux_and_gradient(symbolic, Drain(sign=-1.0))


def test_effectors_custom_parameters_and_temperature():
    symbolic = SymbolicRateEquation(
        expression="kcat * s * (1 + r * e) + c * temperature / 310",
        species={"s": "a", "e": "e"},
        parameters={
            "kcat": "log_kcat",
            "r": "log_custom",
            "c": "custom",
            "temperature": "temperature",
        },
    )
    flux, _ = get_flux_and_gradient(symbolic)
    expected = jnp.exp(-0.1) * 0.5 * (1 + 2.0 * 0.1) - 0.5
    assert jnp.isclose(flux, expected, rtol=1e-12)


def test_reversibility_is_not_implemented_yet():
    symbolic = SymbolicRateEquation(
        expression="kcat * s * reversibility",
        species=MM_SPECIES,
        parameters={"kcat": "log_kcat"},
    )
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        get_model_and_parameters(symbolic)
