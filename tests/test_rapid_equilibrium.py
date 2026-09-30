"""Unit tests for rapid equilibrium reactions."""

import warnings

import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import KineticModel, UndeclaredMoietyWarning
from enzax.rapid_equilibrium import (
    RapidEquilibriumReaction,
    UnusedFastMoietyPivotWarning,
)
from enzax.reactions import Drain, MichaelisMenten

# Mg binds ATP quickly, while ATP is made and used slowly.
MAKE_AND_USE_ATP = {
    "make": Drain(stoichiometry={"atp": 1.0}),
    "use": MichaelisMenten(stoichiometry={"atp": -1.0, "adp": 1.0}),
}
BIND_MG = {
    "bind": RapidEquilibriumReaction(
        stoichiometry={"atp": -1.0, "mg": -1.0, "mgatp": 1.0}
    ),
}
MG_ATP = dict(
    reactions=MAKE_AND_USE_ATP,
    rapid_equilibrium_reactions=BIND_MG,
    balanced_species=["atp", "adp", "mg", "mgatp"],
)


def test_a_model_without_rapid_equilibria_has_no_fast_columns():
    model = KineticModel(
        reactions=MAKE_AND_USE_ATP, balanced_species=["atp", "adp"]
    )
    assert model.rapid_equilibria.reaction_ids == ()
    assert model.S_fast.shape == (2, 0)


def test_rapid_equilibria_are_not_reactions_with_fluxes():
    model = KineticModel(**MG_ATP)
    assert model.reaction_ids == ["make", "use"]
    assert model.rapid_equilibria.reaction_ids == ("bind",)
    assert list(model.stoichiometry) == ["make", "use"]
    assert model.S.shape == (4, 2)


def test_s_fast_holds_the_rapid_equilibria_stoichiometries():
    model = KineticModel(**MG_ATP)
    assert np.array_equal(
        model.S_fast, np.array([[-1.0], [0.0], [-1.0], [1.0]])
    )


def test_species_only_in_rapid_equilibria_come_last():
    model = KineticModel(
        reactions={
            "make": Drain(stoichiometry={"atp": 1.0}),
            "use": MichaelisMenten(
                stoichiometry={"atp": -1.0, "adp": 1.0},
                allosteric_activators=["amp"],
            ),
        },
        rapid_equilibrium_reactions=BIND_MG,
        balanced_species=["atp", "adp", "mg", "mgatp"],
    )
    assert model.species == ["atp", "adp", "amp", "mg", "mgatp"]


def test_a_species_only_in_rapid_equilibria_gets_a_formation_energy():
    model = KineticModel(**MG_ATP)
    assert "mgatp" in model.parameter_labelling["dgf"]


def test_reaction_ids_must_not_clash():
    with pytest.raises(ValueError, match="needs an id of its own"):
        KineticModel(
            reactions=MAKE_AND_USE_ATP,
            rapid_equilibrium_reactions={"use": BIND_MG["bind"]},
            balanced_species=["atp", "adp", "mg", "mgatp"],
        )


def test_fast_moiety_pivot_species_must_be_balanced():
    with pytest.raises(ValueError, match="must be balanced species"):
        KineticModel(
            reactions=MAKE_AND_USE_ATP,
            rapid_equilibrium_reactions=BIND_MG,
            balanced_species=["atp", "adp", "mgatp"],
            fast_moiety_pivot_species=["mg"],
        )


def test_a_fast_moiety_pivot_species_in_no_rapid_equilibrium_warns():
    with pytest.warns(UnusedFastMoietyPivotWarning, match=r"\['adp'\]"):
        KineticModel(**MG_ATP, fast_moiety_pivot_species=["adp"])


def test_a_fast_moiety_pivot_species_in_a_rapid_equilibrium_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnusedFastMoietyPivotWarning)
        KineticModel(**MG_ATP, fast_moiety_pivot_species=["mg"])


def test_models_with_rapid_equilibria_cannot_be_simulated_yet():
    model = KineticModel(**MG_ATP)
    with pytest.raises(NotImplementedError):
        model.dcdt(jnp.ones(4), {})


def get_fast_only_model(stoichiometry, balanced_species, **kwargs):
    return KineticModel(
        reactions={},
        rapid_equilibrium_reactions={
            reaction: RapidEquilibriumReaction(stoichiometry=coefficients)
            for reaction, coefficients in stoichiometry.items()
        },
        balanced_species=balanced_species,
        **kwargs,
    )


# a <-> b <-> c, whose one fast moiety a + b + c can take any pivot.
CHAIN = dict(
    stoichiometry={"ab": {"a": -1.0, "b": 1.0}, "bc": {"b": -1.0, "c": 1.0}},
    balanced_species=["a", "b", "c"],
)


def test_without_rapid_equilibria_each_species_is_a_fast_moiety():
    model = KineticModel(
        reactions=MAKE_AND_USE_ATP, balanced_species=["atp", "adp"]
    )
    assert model.rapid_equilibria.fast_moiety_pivots == ("atp", "adp")
    assert np.array_equal(model.rapid_equilibria.fast_moiety_matrix, np.eye(2))
    assert np.array_equal(model.S_reduced, model.S[model.balanced_species_ix])


def test_mg_binding_fast_moieties_are_the_totals_of_atp_and_mg():
    model = KineticModel(**MG_ATP)
    assert model.rapid_equilibria.fast_moiety_coefficients == {
        "atp": {"atp": 1.0, "mgatp": 1.0},
        "adp": {"adp": 1.0},
        "mg": {"mg": 1.0, "mgatp": 1.0},
    }
    assert model.ode_state_species == ["atp", "adp", "mg"]


def test_rapid_equilibria_leave_fast_moieties_unchanged():
    model = KineticModel(**MG_ATP)
    S_fb = model.S_fast[model.balanced_species_ix, :]
    assert np.allclose(model.rapid_equilibria.fast_moiety_matrix @ S_fb, 0.0)


def test_competing_ligands_share_one_mg_moiety():
    model = KineticModel(
        reactions={
            "make": Drain(stoichiometry={"atp": 1.0}),
            "use": MichaelisMenten(stoichiometry={"mgatp": -1.0, "mgadp": 1.0}),
        },
        rapid_equilibrium_reactions={
            "bind_atp": RapidEquilibriumReaction(
                stoichiometry={"atp": -1.0, "mg": -1.0, "mgatp": 1.0}
            ),
            "bind_adp": RapidEquilibriumReaction(
                stoichiometry={"adp": -1.0, "mg": -1.0, "mgadp": 1.0}
            ),
        },
        balanced_species=["atp", "adp", "mg", "mgatp", "mgadp"],
        moiety_pivot_species=["mg"],
    )
    assert model.rapid_equilibria.fast_moiety_coefficients == {
        "atp": {"atp": 1.0, "mgatp": 1.0},
        "adp": {"adp": 1.0, "mgadp": 1.0},
        "mg": {"mg": 1.0, "mgatp": 1.0, "mgadp": 1.0},
    }
    assert len(model.rapid_equilibria.subnetworks) == 1


def test_pivots_are_chosen_to_avoid_negative_coefficients():
    model = get_fast_only_model(
        stoichiometry={"split": {"a": -1.0, "b": 1.0, "c": 1.0}},
        balanced_species=["a", "b", "c"],
    )
    assert model.rapid_equilibria.fast_moiety_coefficients == {
        "b": {"a": 1.0, "b": 1.0},
        "c": {"a": 1.0, "c": 1.0},
    }


def test_fast_moiety_coefficients_can_be_fractional():
    model = get_fast_only_model(
        stoichiometry={"adk": {"amp": -1.0, "atp": -1.0, "adp": 2.0}},
        balanced_species=["amp", "atp", "adp"],
    )
    assert model.rapid_equilibria.fast_moiety_coefficients == {
        "amp": {"amp": 1.0, "adp": 0.5},
        "atp": {"atp": 1.0, "adp": 0.5},
    }


def test_by_default_pivots_follow_species_order():
    model = get_fast_only_model(**CHAIN)
    assert model.rapid_equilibria.fast_moiety_pivots == ("a",)


def test_fast_moiety_pivot_species_choose_the_labels():
    model = get_fast_only_model(**CHAIN, fast_moiety_pivot_species=["c"])
    assert model.rapid_equilibria.fast_moiety_coefficients == {
        "c": {"a": 1.0, "b": 1.0, "c": 1.0}
    }


def test_two_fast_moiety_pivots_in_one_fast_moiety_are_rejected():
    with pytest.raises(ValueError, match=r"in which \['a', 'c'\] are pivots"):
        get_fast_only_model(**CHAIN, fast_moiety_pivot_species=["a", "c"])


def test_a_fast_moiety_pivot_needing_negative_coefficients_is_rejected():
    with pytest.raises(ValueError, match=r"in which \['a'\] are pivots"):
        get_fast_only_model(
            stoichiometry={"split": {"a": -1.0, "b": 1.0, "c": 1.0}},
            balanced_species=["a", "b", "c"],
            fast_moiety_pivot_species=["a"],
        )


def test_a_moiety_pivot_species_is_a_fast_moiety_pivot():
    model = KineticModel(**MG_ATP, moiety_pivot_species=["mg"])
    assert "mg" in model.rapid_equilibria.fast_moiety_pivots
    assert model.ode_state_species == ["atp", "adp"]


def test_the_link_matrix_is_computed_over_fast_moieties():
    model = KineticModel(**MG_ATP, moiety_pivot_species=["mg"])
    assert np.array_equal(model.L0, np.zeros((1, 2)))


def test_a_moiety_pivot_species_that_cannot_be_a_pivot_is_rejected():
    with pytest.raises(ValueError, match=r"in which \['mgatp'\] are pivots"):
        KineticModel(**MG_ATP, moiety_pivot_species=["mgatp"])


def test_an_undeclared_moiety_through_rapid_equilibria_warns():
    with pytest.warns(UndeclaredMoietyWarning, match="mg \\+ mgatp"):
        KineticModel(**MG_ATP)


def test_a_subnetwork_can_have_no_fast_moieties():
    model = KineticModel(
        reactions={"make": Drain(stoichiometry={"a": 1.0})},
        rapid_equilibrium_reactions={
            "eq": RapidEquilibriumReaction(stoichiometry={"a": -1.0, "b": 1.0})
        },
        balanced_species=["a"],
    )
    assert model.rapid_equilibria.fast_moiety_pivots == ()
    assert model.ode_state_species == []


def test_a_rapid_equilibrium_needs_a_balanced_species():
    with pytest.raises(ValueError, match="involve no balanced species"):
        KineticModel(
            reactions=MAKE_AND_USE_ATP,
            rapid_equilibrium_reactions={
                "bind": RapidEquilibriumReaction(
                    stoichiometry={"x": -1.0, "y": 1.0}
                )
            },
            balanced_species=["atp", "adp"],
        )


def test_rapid_equilibria_must_be_independent():
    with pytest.raises(ValueError, match="linearly independent"):
        get_fast_only_model(
            stoichiometry={
                "f1": {"a": -1.0, "b": 1.0},
                "f2": {"a": 1.0, "b": -1.0},
            },
            balanced_species=["a", "b"],
        )


def test_rapid_equilibria_have_no_rate_parameters():
    model = KineticModel(**MG_ATP)
    assert model.parameter_labelling["log_kcat"] == ("use",)
    assert len(model.reaction_ix) == 2


def test_rapid_equilibria_keep_their_water_stoichiometries():
    model = KineticModel(
        reactions=MAKE_AND_USE_ATP,
        rapid_equilibrium_reactions={
            "hydrolyse": RapidEquilibriumReaction(
                stoichiometry={"atp": -1.0, "adp": 1.0},
                water_stoichiometry=-1.0,
            ),
        },
        balanced_species=["atp", "adp"],
    )
    assert model.rapid_equilibria.water_stoichiometry == (-1.0,)
