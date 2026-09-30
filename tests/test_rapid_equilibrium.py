"""Unit tests for rapid equilibrium reactions."""

import warnings

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from enzax.kinetic_model import KineticModel, UndeclaredMoietyWarning
from enzax.rapid_equilibrium import (
    RapidEquilibriumReaction,
    UnusedFastMoietyPivotWarning,
    solve_rapid_equilibria,
)
from enzax.parameters import pack_parameters
from enzax.reactions import Drain, MichaelisMenten
from enzax.steady_state import get_steady_state_hybrid
from enzax.thermodynamics import GAS_CONSTANT

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


RT = 298.15 * GAS_CONSTANT


def get_dgf(model, values):
    """Get formation energies per species from values keyed by compound."""
    compounds = model.parameter_labelling["dgf"]
    by_compound = jnp.array([values.get(c, 0.0) for c in compounds])
    return by_compound[model.species_to_dgf_ix]


def solve(model, totals, dgf_values, log_conc_unbalanced=(), **kwargs):
    return solve_rapid_equilibria(
        model.rapid_equilibria,
        jnp.array(totals),
        jnp.array(log_conc_unbalanced),
        get_dgf(model, dgf_values),
        298.15,
        model.water_dgf,
        **kwargs,
    )


def mg_binding_dgf(log_k):
    return {"atp": -2000.0, "mg": -450.0, "mgatp": -2450.0 - RT * log_k}


@pytest.mark.parametrize("keq", [1e-3, 1.0, 1e4, 1e8])
@pytest.mark.parametrize("atp_total, mg_total", [(1.0, 0.5), (1e-4, 3.0)])
def test_single_binding_matches_the_quadratic(keq, atp_total, mg_total):
    model = KineticModel(**MG_ATP)
    conc = solve(model, [atp_total, 0.2, mg_total], mg_binding_dgf(np.log(keq)))
    b = keq * (atp_total + mg_total) + 1.0
    bound = (b - np.sqrt(b**2 - 4 * keq**2 * atp_total * mg_total)) / (2 * keq)
    expected = [atp_total - bound, 0.2, mg_total - bound, bound]
    assert np.allclose(conc, expected, rtol=1e-8)


COMPETING = dict(
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


def test_competing_ligands_satisfy_totals_and_equilibria_across_scales():
    model = KineticModel(**COMPETING)
    rng = np.random.default_rng(0)
    totals = jnp.array(10 ** rng.uniform(-6, 2, (500, 3)))
    log_k = rng.uniform(np.log(1e-4), np.log(1e9), (500, 2))
    compounds = model.parameter_labelling["dgf"]
    dgf_values = np.zeros((500, len(compounds)))
    dgf_values[:, compounds.index("atp")] = -2000.0
    dgf_values[:, compounds.index("adp")] = -1500.0
    dgf_values[:, compounds.index("mg")] = -450.0
    dgf_values[:, compounds.index("mgatp")] = -2450.0 - RT * log_k[:, 0]
    dgf_values[:, compounds.index("mgadp")] = -1950.0 - RT * log_k[:, 1]
    dgf = jnp.array(dgf_values)[:, model.species_to_dgf_ix]
    conc = jax.vmap(
        lambda y, g: solve_rapid_equilibria(
            model.rapid_equilibria, y, jnp.array([]), g, 298.15, model.water_dgf
        )
    )(totals, dgf)
    P = model.rapid_equilibria.fast_moiety_matrix
    S_fb = model.S_fast[model.balanced_species_ix, :]
    assert np.allclose(conc @ P.T, totals, rtol=1e-8)
    assert np.allclose(jnp.log(conc) @ S_fb, log_k, atol=1e-7)


def test_an_unbalanced_species_shifts_the_equilibrium():
    model = KineticModel(
        reactions=MAKE_AND_USE_ATP,
        rapid_equilibrium_reactions=BIND_MG,
        balanced_species=["atp", "adp", "mgatp"],
    )
    conc = solve(model, [1.0, 0.2], mg_binding_dgf(np.log(10.0)), [np.log(0.3)])
    atp, _, mgatp = conc
    assert np.isclose(mgatp / atp, 10.0 * 0.3, rtol=1e-8)
    assert np.isclose(atp + mgatp, 1.0, rtol=1e-8)


def test_water_enters_the_equilibrium_constant():
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
    dgf_values = {"atp": -2000.0, "adp": -1850.0}
    atp, adp = solve(model, [1.0], dgf_values)
    dgr = -1850.0 + 2000.0 - model.water_dgf
    assert np.isclose(adp / atp, np.exp(-dgr / RT), rtol=1e-8)


def test_a_species_in_a_subnetwork_with_no_fast_moieties_is_fixed():
    model = KineticModel(
        reactions={"make": Drain(stoichiometry={"a": 1.0})},
        rapid_equilibrium_reactions={
            "eq": RapidEquilibriumReaction(stoichiometry={"a": -1.0, "b": 1.0})
        },
        balanced_species=["a"],
    )
    (a,) = solve(model, [], {"a": 0.0, "b": -RT * np.log(4.0)}, [np.log(2.0)])
    assert np.isclose(a, 2.0 / 4.0, rtol=1e-8)


def test_a_non_positive_total_gives_nan_only_for_that_member():
    model = KineticModel(**MG_ATP)
    totals = jnp.array([[1.0, 0.2, 0.5], [1.0, 0.2, 0.0], [2.0, 0.1, 1.0]])
    dgf = get_dgf(model, mg_binding_dgf(np.log(100.0)))
    conc = jax.vmap(
        lambda y: solve_rapid_equilibria(
            model.rapid_equilibria, y, jnp.array([]), dgf, 298.15, -150.9
        )
    )(totals)
    assert np.all(np.isnan(conc[1]))
    assert np.all(np.isfinite(conc[np.array([0, 2])]))


def test_gradients_match_finite_differences():
    model = KineticModel(**COMPETING)
    compounds = model.parameter_labelling["dgf"]
    base = {
        "atp": -2000.0,
        "adp": -1500.0,
        "mg": -450.0,
        "mgatp": -2450.0 - RT * np.log(50.0),
        "mgadp": -1950.0 - RT * np.log(5.0),
    }
    by_compound = jnp.array([base[c] for c in compounds])
    totals = jnp.array([1.0, 0.4, 0.8])

    def free_mg(totals, by_compound):
        conc = solve_rapid_equilibria(
            model.rapid_equilibria,
            totals,
            jnp.array([]),
            by_compound[model.species_to_dgf_ix],
            298.15,
            model.water_dgf,
        )
        return conc[model.balanced_species.index("mg")]

    def central_difference(f, x, h):
        return np.array(
            [
                (f(x.at[i].add(h)) - f(x.at[i].add(-h))) / (2 * h)
                for i in range(len(x))
            ]
        )

    fd_totals = central_difference(
        lambda y: free_mg(y, by_compound), totals, 1e-6
    )
    fd_dgf = central_difference(lambda g: free_mg(totals, g), by_compound, 1e-4)
    grad_totals, grad_dgf = jax.grad(free_mg, argnums=(0, 1))(
        totals, by_compound
    )
    fwd_totals = jax.jacfwd(free_mg)(totals, by_compound)
    assert np.allclose(grad_totals, fd_totals, rtol=1e-5, atol=1e-9)
    assert np.allclose(fwd_totals, fd_totals, rtol=1e-5, atol=1e-9)
    assert np.allclose(grad_dgf, fd_dgf, rtol=1e-5, atol=1e-9)


# Mg binds ATP and ADP quickly, while ADK, an ATPase and an ATP synthase are
# slow. Total Mg and total adenylate are conserved.
ENERGY = KineticModel(
    reactions={
        "adk": MichaelisMenten(
            stoichiometry={"adp": -2.0, "atp": 1.0, "amp": 1.0}
        ),
        "atpase": MichaelisMenten(
            stoichiometry={"mgatp": -1.0, "mgadp": 1.0, "pi": 1.0},
            reversible=False,
        ),
        "synthase": MichaelisMenten(
            stoichiometry={"mgadp": -1.0, "pi": -1.0, "mgatp": 1.0},
            reversible=False,
        ),
    },
    rapid_equilibrium_reactions={
        "bind_atp": RapidEquilibriumReaction(
            stoichiometry={"atp": -1.0, "mg": -1.0, "mgatp": 1.0}
        ),
        "bind_adp": RapidEquilibriumReaction(
            stoichiometry={"adp": -1.0, "mg": -1.0, "mgadp": 1.0}
        ),
    },
    balanced_species=["atp", "adp", "amp", "mg", "mgatp", "mgadp"],
    moiety_pivot_species=["mg", "amp"],
)


def get_energy_parameters(log_kcat_synthase=np.log(2.0)):
    labelling = ENERGY.parameter_labelling
    dgf = {"mgatp": -RT * np.log(10.0), "mgadp": -RT * np.log(2.0)}
    return pack_parameters(
        labelling,
        {
            "log_saturation_constant": {
                label: np.log(0.5)
                for label in labelling["log_saturation_constant"]
            },
            "log_kcat": {
                "adk": np.log(5.0),
                "atpase": np.log(1.0),
                "synthase": log_kcat_synthase,
            },
            "log_enzyme": {
                label: np.log(0.1) for label in labelling["log_enzyme"]
            },
            "dgf": {c: dgf.get(c, 0.0) for c in labelling["dgf"]},
            "log_conc_unbalanced": {"pi": np.log(1.0)},
            "moiety_totals": {"mg": 1.0, "amp": 3.0},
            "temperature": 298.15,
        },
    )


def test_the_ode_state_is_the_non_conserved_fast_moieties():
    assert ENERGY.ode_state_species == ["atp", "adp"]
    assert ENERGY.rapid_equilibria.fast_moiety_coefficients["atp"] == {
        "atp": 1.0,
        "mgatp": 1.0,
    }


def test_balanced_concentrations_honour_totals_and_equilibria():
    parameters = get_energy_parameters()
    conc = ENERGY.get_balanced_conc(jnp.array([1.2, 0.9]), parameters)
    c = dict(zip(ENERGY.balanced_species, conc.tolist()))
    assert np.isclose(c["atp"] + c["mgatp"], 1.2)
    assert np.isclose(c["adp"] + c["mgadp"], 0.9)
    assert np.isclose(c["amp"], 3.0 - 1.2 - 0.9)
    assert np.isclose(c["mg"] + c["mgatp"] + c["mgadp"], 1.0)
    assert np.isclose(c["mgatp"] / (c["atp"] * c["mg"]), 10.0)
    assert np.isclose(c["mgadp"] / (c["adp"] * c["mg"]), 2.0)


def test_get_ode_state_inverts_get_balanced_conc():
    parameters = get_energy_parameters()
    state = jnp.array([1.2, 0.9])
    conc = ENERGY.get_balanced_conc(state, parameters)
    assert np.allclose(ENERGY.get_ode_state(conc), state)


def test_dcdt_is_the_fast_moieties_rate_of_change():
    parameters = get_energy_parameters()
    state = jnp.array([1.2, 0.9])
    conc = ENERGY.get_balanced_conc(state, parameters)
    v = ENERGY.flux(conc, parameters)
    S = dict(zip(ENERGY.species, ENERGY.S.tolist()))
    atp_moiety = (np.array(S["atp"]) + np.array(S["mgatp"])) @ v
    adp_moiety = (np.array(S["adp"]) + np.array(S["mgadp"])) @ v
    assert np.allclose(ENERGY.dcdt(state, parameters), [atp_moiety, adp_moiety])


def test_a_model_with_rapid_equilibria_reaches_a_steady_state():
    parameters = get_energy_parameters()
    steady = get_steady_state_hybrid(ENERGY, jnp.array([1.0, 1.0]), parameters)
    assert np.allclose(ENERGY.dcdt(steady, parameters), 0.0, atol=1e-9)
    conc = ENERGY.get_balanced_conc(steady, parameters)
    assert np.all(conc > 0)
    c = dict(zip(ENERGY.balanced_species, conc.tolist()))
    assert np.isclose(c["mg"] + c["mgatp"] + c["mgadp"], 1.0)
    assert np.isclose(
        c["atp"] + c["adp"] + c["amp"] + c["mgatp"] + c["mgadp"], 3.0
    )


def test_steady_state_gradients_match_finite_differences():
    guess = jnp.array([1.0, 1.0])

    def free_mg_at_steady_state(log_kcat_synthase):
        parameters = get_energy_parameters(log_kcat_synthase)
        steady = get_steady_state_hybrid(ENERGY, guess, parameters)
        conc = ENERGY.get_balanced_conc(steady, parameters)
        return conc[ENERGY.balanced_species.index("mg")]

    x = jnp.log(2.0)
    h = 1e-5
    fd = (free_mg_at_steady_state(x + h) - free_mg_at_steady_state(x - h)) / (
        2 * h
    )
    assert np.isclose(jax.grad(free_mg_at_steady_state)(x), fd, rtol=1e-5)
