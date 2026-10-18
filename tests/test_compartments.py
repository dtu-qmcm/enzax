import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.compartment import Compartment
from enzax.kinetic_model import KineticModel, UndeclaredMoietyWarning
from enzax.parameters import pack_parameters
from enzax.rapid_equilibrium import (
    RapidEquilibriumNetwork,
    RapidEquilibriumReaction,
)
from enzax.reactions import Drain, SymbolicReaction
from enzax.steady_state import get_steady_state_dae, get_steady_state_hybrid

jax.config.update("jax_enable_x64", True)


def transport(compartment=None, stoichiometry=None):
    return SymbolicReaction(
        stoichiometry=stoichiometry or {"a_c": -1.0, "a_e": 1.0},
        expression="k * a_c",
        species=["a_c"],
        parameters={"k": "log_kcat"},
        compartment=compartment,
    )


def get_model(compartments, transporter=None, balanced=("a_c", "a_e")):
    return KineticModel(
        reactions={
            "in": Drain(stoichiometry={"a_c": 1.0}),
            "t": transporter or transport(compartment="c"),
            "out": Drain(stoichiometry={"a_e": -1.0}),
        },
        balanced_species=list(balanced),
        compartments=compartments,
    )


C_AND_E = {
    "c": Compartment(volume=1.0, species=["a_c"]),
    "e": Compartment(volume=3.0, species=["a_e"]),
}


def test_a_model_without_compartments_assigns_nothing():
    model = get_model(None, transporter=transport())
    assert model.species_to_compartment == {}
    assert model.reaction_to_compartment == {}


def test_reactions_are_in_their_species_compartment_or_the_stated_one():
    model = get_model(C_AND_E)
    assert model.species_to_compartment == {"a_c": "c", "a_e": "e"}
    assert model.reaction_to_compartment == {"in": "c", "t": "c", "out": "e"}


def test_a_transport_can_be_per_either_compartment():
    model = get_model(C_AND_E, transporter=transport(compartment="e"))
    assert model.reaction_to_compartment["t"] == "e"


def test_an_unbalanced_species_elsewhere_makes_a_reaction_a_transport():
    with pytest.raises(ValueError, match="must say which compartment"):
        get_model(C_AND_E, transporter=transport(), balanced=["a_c"])


def test_a_transport_must_state_its_compartment():
    with pytest.raises(ValueError, match="must say which compartment"):
        get_model(C_AND_E, transporter=transport())


def test_a_stated_compartment_must_hold_some_of_the_reaction_species():
    within_c = transport(
        compartment="e",
        stoichiometry={"a_c": -1.0, "b_c": 1.0},
    )
    compartments = {
        "c": Compartment(volume=1.0, species=["a_c", "b_c"]),
        "e": Compartment(volume=3.0, species=["a_e"]),
    }
    with pytest.raises(ValueError, match=r"species are in \['c'\]"):
        KineticModel(
            reactions={
                "t": within_c,
                "out": Drain(stoichiometry={"a_e": -1.0}),
            },
            balanced_species=["a_c"],
            compartments=compartments,
        )


def test_a_stated_compartment_must_exist():
    with pytest.raises(ValueError, match="compartment 'm'"):
        get_model(C_AND_E, transporter=transport(compartment="m"))


def test_a_stated_compartment_needs_declared_compartments():
    with pytest.raises(ValueError, match="declares no compartments"):
        get_model(None)


def test_every_species_must_be_in_a_compartment():
    with pytest.raises(ValueError, match=r"in none: \['a_e'\]"):
        get_model({"c": Compartment(volume=1.0, species=["a_c"])})


def test_a_species_cannot_be_in_two_compartments():
    with pytest.raises(ValueError, match="in several"):
        get_model(
            {
                "c": Compartment(volume=1.0, species=["a_c"]),
                "e": Compartment(volume=3.0, species=["a_e", "a_c"]),
            },
        )


def test_compartments_cannot_name_unknown_species():
    with pytest.raises(ValueError, match=r"species \['b'\]"):
        get_model(
            {
                "c": Compartment(volume=1.0, species=["a_c", "b"]),
                "e": Compartment(volume=3.0, species=["a_e"]),
            },
        )


def test_volumes_must_be_positive():
    with pytest.raises(ValueError, match="positive volumes"):
        get_model(
            {
                "c": Compartment(volume=1.0, species=["a_c"]),
                "e": Compartment(volume=0.0, species=["a_e"]),
            },
        )


V_C, V_E = 1.0, 3.0
V_IN, K_T, K_BACK, K_OUT = 1.5, 2.0, 0.25, 0.5


def first_order(species, stoichiometry, compartment=None):
    return SymbolicReaction(
        stoichiometry=stoichiometry,
        expression=f"k * {species}",
        species=[species],
        parameters={"k": "log_kcat"},
        compartment=compartment,
    )


INFLOW = Drain(stoichiometry={"a_c": 1.0})
OUTFLOW = first_order("a_e", {"a_e": -1.0})
OPEN = KineticModel(
    reactions={"in": INFLOW, "t": transport(compartment="c"), "out": OUTFLOW},
    balanced_species=["a_c", "a_e"],
    compartments=C_AND_E,
)
BACK = first_order("a_e", {"a_e": -1.0, "a_c": 1.0}, compartment="e")
CLOSED = KineticModel(
    reactions={"t": transport(compartment="c"), "back": BACK},
    balanced_species=["a_c", "a_e"],
    moiety_label_species=["a_e"],
    compartments=C_AND_E,
)


def get_parameters(model, **extra):
    values = {
        "log_drain": {"in": np.log(V_IN)},
        "log_kcat": {
            "t": np.log(K_T),
            "back": np.log(K_BACK),
            "out": np.log(K_OUT),
        },
        "dgf": {"a_c": 0.0, "a_e": 0.0, "a": 0.0},
        "temperature": 298.15,
        **extra,
    }
    labelling = model.parameter_labelling
    return pack_parameters(
        labelling,
        {
            parameter: (
                {k: v for k, v in value.items() if k in labelling[parameter]}
                if isinstance(value, dict)
                else value
            )
            for parameter, value in values.items()
            if parameter in labelling
        },
    )


def test_rates_are_per_their_compartment_volume():
    state = jnp.array([0.7, 0.4])
    dcdt = OPEN.dcdt(state, get_parameters(OPEN))
    assert np.allclose(
        dcdt,
        [V_IN - K_T * 0.7, V_C / V_E * K_T * 0.7 - K_OUT * 0.4],
    )


def test_dndt_is_the_volume_times_dcdt():
    state = jnp.array([0.7, 0.4])
    parameters = get_parameters(OPEN)
    assert np.allclose(
        OPEN.dndt(state, parameters),
        np.array([V_C, V_E]) * OPEN.dcdt(state, parameters),
    )


def test_the_open_model_reaches_its_closed_form_steady_state():
    steady = get_steady_state_hybrid(
        OPEN,
        jnp.array([1.0, 1.0]),
        get_parameters(OPEN),
    )
    assert np.allclose(
        steady,
        [V_IN / K_T, V_C * V_IN / (V_E * K_OUT)],
        rtol=1e-8,
    )


def test_transport_conserves_the_total_amount():
    with pytest.warns(UndeclaredMoietyWarning, match=r"a_c \+ a_e"):
        model = KineticModel(
            reactions={"t": transport(compartment="c"), "back": BACK},
            balanced_species=["a_c", "a_e"],
            compartments=C_AND_E,
        )
    dndt = model.dndt(jnp.array([0.7, 0.4]), get_parameters(model))
    assert not np.isclose(dndt[0], 0.0)
    assert np.isclose(jnp.sum(dndt), 0.0)


def test_a_moiety_total_is_an_amount():
    total = 2.0
    parameters = get_parameters(CLOSED, moiety_totals={"a_e": total})
    conc = CLOSED.get_balanced_conc(jnp.array([0.5]), parameters)
    assert np.allclose(conc, [0.5, (total - V_C * 0.5) / V_E])
    assert np.allclose(CLOSED.get_ode_state(conc), [0.5])


def test_the_closed_model_shares_its_amount_at_equilibrium():
    total = 2.0
    parameters = get_parameters(CLOSED, moiety_totals={"a_e": total})
    steady = get_steady_state_hybrid(CLOSED, jnp.array([1.0]), parameters)
    a_c = total / (V_C + V_E * V_C * K_T / (V_E * K_BACK))
    assert np.allclose(steady, [a_c], rtol=1e-8)


# a_c and a_e are one compound, so the rapid equilibrium makes their
# concentrations equal, and the fast moiety's total is the amount of both.
ACROSS = KineticModel(
    reactions={"in": INFLOW, "out": OUTFLOW},
    rapid_equilibrium_network=RapidEquilibriumNetwork(
        reactions={
            "f": RapidEquilibriumReaction(
                stoichiometry={"a_c": -1.0, "a_e": 1.0},
            ),
        },
    ),
    balanced_species=["a_c", "a_e"],
    compound_to_species={"a": ["a_c", "a_e"]},
    compartments=C_AND_E,
)


def test_a_rapid_equilibrium_across_compartments_shares_the_amount():
    parameters = get_parameters(ACROSS)
    assert ACROSS.ode_state_species == ["a_c"]
    conc = ACROSS.get_balanced_conc(jnp.array([2.0]), parameters)
    shared = 2.0 * V_C / (V_C + V_E)
    assert np.allclose(conc, [shared, shared], rtol=1e-9)
    assert np.allclose(ACROSS.get_ode_state(conc), [2.0], rtol=1e-9)
    assert np.allclose(
        ACROSS.dcdt(jnp.array([2.0]), parameters),
        [(V_C * V_IN - V_E * K_OUT * shared) / V_C],
        rtol=1e-9,
    )


def test_across_compartments_the_dae_matches_the_nested_solve():
    pytest.importorskip("diffrax_bdf")
    parameters = get_parameters(ACROSS)
    a_e = V_C * V_IN / (V_E * K_OUT)
    for solve in [get_steady_state_hybrid, get_steady_state_dae]:
        steady = solve(ACROSS, jnp.array([1.0]), parameters)
        conc = ACROSS.get_balanced_conc(steady, parameters)
        assert np.allclose(conc, [a_e, a_e], rtol=1e-7)
