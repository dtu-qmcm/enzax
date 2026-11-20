from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from enzax.algebraic_constraint import (
    AlgebraicConstraint,
    ConstraintLabels,
    ConstraintScope,
)
from enzax.examples import linear
from enzax.kinetic_model import KineticModel
from enzax.parameters import get_parameter_position, pack_parameters
from enzax.rapid_equilibrium import (
    RapidEquilibriumNetwork,
    RapidEquilibriumReaction,
)
from enzax.reactions import MassAction, MichaelisMenten, SymbolicReaction
from enzax.thermodynamics import FARADAY, GAS_CONSTANT, MEMBRANE_POTENTIAL
from tests.test_batch_rule import (
    assert_close_to_scale,
    get_batch_groups,
    get_batched_fluxes,
    get_per_reaction_fluxes,
)

TEMPERATURE = 310.0
PSI = -0.09
F_OVER_RT = FARADAY / (GAS_CONSTANT * TEMPERATURE)
NA_IMPORT = {"na_e": -1.0, "na": 1.0}
TRANSPORTS = {
    "mass_action": MassAction(stoichiometry=NA_IMPORT, transported_charge=1.0),
    "michaelis_menten": MichaelisMenten(
        stoichiometry=NA_IMPORT,
        transported_charge=1.0,
    ),
    "symbolic": SymbolicReaction(
        stoichiometry=NA_IMPORT,
        expression="k * na_e * reversibility",
        species=["na_e"],
        default_parameter_kind="log_custom",
        transported_charge=1.0,
    ),
}


def get_parameters(model, psi=PSI, **values):
    spec = {
        kind: {label: 0.0 for label in labels}
        for kind, labels in model.parameter_labelling.items()
        if labels
    }
    for kind, kind_values in values.items():
        spec[kind] |= kind_values
    spec["temperature"] = TEMPERATURE
    if MEMBRANE_POTENTIAL in model.parameter_labelling:
        spec[MEMBRANE_POTENTIAL] = psi
    return pack_parameters(model.parameter_labelling, spec)


def get_transport_model(reaction):
    return KineticModel(
        reactions={"t": reaction},
        balanced_species=["na"],
        compound_to_species={"na": ["na", "na_e"]},
    )


def test_models_without_charge_have_no_membrane_potential():
    assert MEMBRANE_POTENTIAL not in linear.model.parameter_labelling


def test_a_charged_reaction_makes_the_membrane_potential_a_parameter():
    model = get_transport_model(TRANSPORTS["mass_action"])
    assert model.parameter_labelling[MEMBRANE_POTENTIAL] == ()


@pytest.mark.parametrize("name", list(TRANSPORTS))
def test_a_charged_transport_settles_at_the_electrochemical_equilibrium(name):
    model = get_transport_model(TRANSPORTS[name])
    parameters = get_parameters(model)
    na = jnp.exp(-F_OVER_RT * PSI)
    at_equilibrium = model.flux(jnp.array([na]), parameters)[0]
    at_equal_conc = model.flux(jnp.array([1.0]), parameters)[0]
    assert jnp.abs(at_equilibrium) < 1e-12 * jnp.abs(at_equal_conc)
    assert at_equal_conc > 0


@pytest.mark.parametrize("name", list(TRANSPORTS))
def test_flux_derivatives_with_respect_to_membrane_potential(name):
    model = get_transport_model(TRANSPORTS[name])

    def flux(psi):
        return model.flux(jnp.array([2.0]), get_parameters(model, psi))[0]

    step = 1e-6
    finite_difference = (flux(PSI + step) - flux(PSI - step)) / (2 * step)
    np.testing.assert_allclose(
        jax.grad(flux)(PSI),
        finite_difference,
        rtol=1e-7,
    )


def test_charged_and_uncharged_reactions_batch_together():
    model = KineticModel(
        reactions={
            "na_in": MassAction(
                stoichiometry=NA_IMPORT,
                transported_charge=1.0,
            ),
            "cl_in": MassAction(
                stoichiometry={"cl_e": -1.0, "cl": 1.0},
                transported_charge=-1.0,
            ),
            "binding": MassAction(
                stoichiometry={"na": -1.0, "cl": -1.0, "nacl": 1.0},
            ),
        },
        balanced_species=["na", "cl", "nacl"],
        compound_to_species={"na": ["na", "na_e"], "cl": ["cl", "cl_e"]},
    )
    parameters = get_parameters(model, dgf={"nacl": -3.0})
    conc = jnp.array([0.7, 1.3, 0.4, 2.0, 0.5])
    [(rule, positions)] = get_batch_groups(model)
    assert_close_to_scale(
        get_batched_fluxes(model, conc, parameters, rule, positions),
        get_per_reaction_fluxes(model, conc, parameters, positions),
    )


def test_a_charged_rapid_equilibrium_follows_the_membrane_potential():
    model = KineticModel(
        reactions={"r": MassAction(stoichiometry={"a": -1.0, "b": 1.0})},
        rapid_equilibrium_network=RapidEquilibriumNetwork(
            reactions={
                "cl_in": RapidEquilibriumReaction(
                    stoichiometry={"cl_e": -1.0, "cl": 1.0},
                    transported_charge=-1.0,
                ),
            },
        ),
        balanced_species=["a", "b", "cl"],
        compound_to_species={"cl": ["cl", "cl_e"]},
    )
    parameters = get_parameters(model)
    state = model.get_ode_state(jnp.array([1.0, 2.0, 1.0]))
    cl = model.get_balanced_conc(state, parameters)[2]
    np.testing.assert_allclose(cl, np.exp(F_OVER_RT * PSI), rtol=1e-10)


@dataclass(frozen=True)
class ChargeBalanceLabels(ConstraintLabels):
    fixed_anion: str

    def by_parameter(self):
        return {"log_custom": (self.fixed_anion,)}


class ChargeBalanceIx(eqx.Module):
    ix_cation: int
    ix_anion: int
    ix_fixed_anion: int


class ChargeBalance(AlgebraicConstraint):
    """A cation balances a permeant anion and a fixed amount of anion."""

    cation: str
    anion: str

    def get_species(self):
        return (self.cation, self.anion)

    def get_labels(self, scope: ConstraintScope) -> ChargeBalanceLabels:
        return ChargeBalanceLabels(fixed_anion=f"cu|{scope.constraint_id}|x")

    def get_input_indexes(self, scope, labelling) -> ChargeBalanceIx:
        return ChargeBalanceIx(
            ix_cation=scope.species.index(self.cation),
            ix_anion=scope.species.index(self.anion),
            ix_fixed_anion=get_parameter_position(
                labelling,
                "log_custom",
                self.get_labels(scope).fixed_anion,
            ),
        )

    def get_input(self, parameters, ix: ChargeBalanceIx):
        return (jnp.exp(parameters["log_custom"][ix.ix_fixed_anion]), ix)

    def __call__(self, conc, variables, constraint_input):
        fixed_anion, ix = constraint_input
        return jnp.array(
            [conc[ix.ix_cation] - conc[ix.ix_anion] - fixed_anion],
        )


ALGEBRAIC_MODEL = KineticModel(
    reactions={
        "na_in": MassAction(stoichiometry=NA_IMPORT, transported_charge=1.0),
        "reads_psi": SymbolicReaction(
            stoichiometry={"na": -1.0},
            expression="psi",
            parameters={"psi": "membrane_potential"},
        ),
    },
    rapid_equilibrium_network=RapidEquilibriumNetwork(
        reactions={
            "cl_in": RapidEquilibriumReaction(
                stoichiometry={"cl_e": -1.0, "cl": 1.0},
                transported_charge=-1.0,
            ),
        },
    ),
    balanced_species=["na", "cl"],
    compound_to_species={"na": ["na", "na_e"], "cl": ["cl", "cl_e"]},
    algebraic_constraints={
        "charge": ChargeBalance(
            cation="k",
            anion="cl",
            variables=[MEMBRANE_POTENTIAL],
        ),
    },
)
K, FIXED_ANION, CL_E = 120.0, 80.0, 100.0


def test_an_algebraic_membrane_potential_is_not_a_parameter():
    assert MEMBRANE_POTENTIAL not in ALGEBRAIC_MODEL.parameter_labelling


def test_an_algebraic_membrane_potential_reaches_reactions_and_equilibria():
    parameters = get_parameters(
        ALGEBRAIC_MODEL,
        log_conc_unbalanced={
            "k": np.log(K),
            "cl_e": np.log(CL_E),
            "na_e": np.log(140.0),
        },
        log_custom={"cu|charge|x": np.log(FIXED_ANION)},
    )
    state = ALGEBRAIC_MODEL.get_ode_state(jnp.array([10.0, 1.0]))
    [psi] = ALGEBRAIC_MODEL.get_algebraic_variables(state, parameters)
    expected_psi = np.log((K - FIXED_ANION) / CL_E) / F_OVER_RT
    np.testing.assert_allclose(psi, expected_psi, rtol=1e-8)
    na = 140.0 * jnp.exp(-F_OVER_RT * psi)
    flux = ALGEBRAIC_MODEL.flux(
        jnp.array([na, K - FIXED_ANION]),
        parameters,
        jnp.array([psi]),
    )
    assert jnp.abs(flux[0]) < 1e-10
    np.testing.assert_allclose(flux[1], psi, rtol=1e-12)
