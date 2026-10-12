import pytest

from enzax.compartment import Compartment
from enzax.kinetic_model import KineticModel
from enzax.reactions import Drain, SymbolicReaction


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
