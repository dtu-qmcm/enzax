"""Unit tests for the mass action rate law."""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import KineticModel
from enzax.parameters import pack_parameters
from enzax.reactions import MassAction, SymbolicReaction
from enzax.thermodynamics import GAS_CONSTANT

TEMPERATURE = 310.0
RT = TEMPERATURE * GAS_CONSTANT
# 2a + c <-> d
STOICHIOMETRY = {"a": -2.0, "c": -1.0, "d": 1.0}
DGF = {"a": -3.0, "c": 1.0, "d": -4.0}
LOG_KEQ = -(DGF["d"] - 2 * DGF["a"] - DGF["c"]) / RT


def get_model(reactions):
    species = sorted(
        {s for reaction in reactions.values() for s in reaction.stoichiometry},
    )
    return KineticModel(reactions=reactions, balanced_species=species)


def get_parameters(model, log_k_plus=np.log(2.0), dgf=DGF):
    labelling = model.parameter_labelling
    return pack_parameters(
        labelling,
        {
            "log_k_plus": {
                label: log_k_plus for label in labelling["log_k_plus"]
            },
            "dgf": {label: dgf[label] for label in labelling["dgf"]},
            "temperature": TEMPERATURE,
        },
    )


def get_flux(reaction, conc, log_k_plus=np.log(2.0), dgf=DGF):
    model = get_model({"r1": reaction})
    parameters = get_parameters(model, log_k_plus, dgf)
    conc_balanced = jnp.array([conc[s] for s in model.balanced_species])
    return model.flux(conc_balanced, parameters)[0]


def test_the_irreversible_rate_honours_stoichiometric_exponents():
    conc = {"a": 0.3, "c": 0.5, "d": 0.2}
    flux = get_flux(
        MassAction(stoichiometry=STOICHIOMETRY, reversible=False),
        conc,
    )
    assert np.isclose(flux, 2.0 * 0.3**2 * 0.5)


def test_the_reversible_rate_subtracts_the_backward_rate():
    conc = {"a": 0.3, "c": 0.5, "d": 0.2}
    flux = get_flux(MassAction(stoichiometry=STOICHIOMETRY), conc)
    expected = 2.0 * (0.3**2 * 0.5 - 0.2 / np.exp(LOG_KEQ))
    assert np.isclose(flux, expected, rtol=1e-10)


def test_the_flux_is_zero_at_equilibrium():
    a, c = 0.3, 0.5
    conc = {"a": a, "c": c, "d": np.exp(LOG_KEQ) * a**2 * c}
    flux = get_flux(MassAction(stoichiometry=STOICHIOMETRY), conc)
    assert np.isclose(flux, 0.0, atol=1e-12)


@pytest.mark.parametrize("factor, sign", [(0.5, 1.0), (2.0, -1.0)])
def test_the_flux_follows_the_driving_force(factor, sign):
    a, c = 0.3, 0.5
    conc = {"a": a, "c": c, "d": factor * np.exp(LOG_KEQ) * a**2 * c}
    flux = get_flux(MassAction(stoichiometry=STOICHIOMETRY), conc)
    assert np.sign(flux) == sign


def test_water_enters_the_equilibrium_constant():
    reaction = MassAction(
        stoichiometry={"a": -1.0, "d": 1.0},
        water_stoichiometry=-1.0,
    )
    water_dgf = get_model({"r1": reaction}).water_dgf
    # a + water <-> d, with an equilibrium constant of 4
    dgf = {"a": -3.0, "d": -3.0 + water_dgf - RT * np.log(4.0)}
    conc = {"a": 0.3, "d": 0.3 * 4.0}
    assert np.isclose(get_flux(reaction, conc, dgf=dgf), 0.0, atol=1e-12)
    conc_without_water = {
        "a": 0.3,
        "d": 0.3 * np.exp(-(dgf["d"] - dgf["a"]) / RT),
    }
    assert not np.isclose(get_flux(reaction, conc_without_water, dgf=dgf), 0.0)


def test_the_gradient_matches_finite_differences():
    reaction = MassAction(stoichiometry=STOICHIOMETRY)
    model = get_model({"r1": reaction})
    conc = jnp.array([0.3, 0.5, 0.2])

    def flux(conc, log_k_plus):
        return model.flux(conc, get_parameters(model, log_k_plus))[0]

    grad_conc, grad_k = jax.grad(flux, argnums=(0, 1))(conc, np.log(2.0))
    h = 1e-6
    fd_conc = [
        (
            flux(conc.at[i].add(h), np.log(2.0))
            - flux(conc.at[i].add(-h), np.log(2.0))
        )
        / (2 * h)
        for i in range(3)
    ]
    fd_k = (flux(conc, np.log(2.0) + h) - flux(conc, np.log(2.0) - h)) / (2 * h)
    assert np.allclose(grad_conc, fd_conc, rtol=1e-6)
    assert np.isclose(grad_k, fd_k, rtol=1e-6)


def test_k_plus_is_labelled_by_the_reaction_by_default():
    model = get_model({"r1": MassAction(stoichiometry=STOICHIOMETRY)})
    assert model.parameter_labelling["log_k_plus"] == ("r1",)


def test_reactions_can_share_a_k_plus():
    model = get_model(
        {
            "r1": MassAction(
                stoichiometry={"a": -1.0, "c": 1.0},
                k_plus_label="k",
            ),
            "r2": MassAction(
                stoichiometry={"c": -1.0, "d": 1.0},
                k_plus_label="k",
            ),
        },
    )
    assert model.parameter_labelling["log_k_plus"] == ("k",)


def test_a_symbolic_reaction_can_use_k_plus():
    model = get_model(
        {
            "r1": SymbolicReaction(
                stoichiometry={"a": -1.0, "d": 1.0},
                expression="k * s",
                species={"s": "a"},
                parameters={"k": "log_k_plus"},
            ),
        },
    )
    assert model.parameter_labelling["log_k_plus"] == ("r1",)


def test_the_flux_is_zero_at_equilibrium_for_dilute_products():
    reaction = MassAction(stoichiometry={"a": -1.0, "d": 1.0})
    # a <-> d with an equilibrium constant of 1e-10
    dgf = {"a": 0.0, "d": RT * np.log(1e10)}
    conc = {"a": 0.3, "d": 0.3e-10}
    assert np.isclose(get_flux(reaction, conc, dgf=dgf), 0.0, atol=1e-15)
