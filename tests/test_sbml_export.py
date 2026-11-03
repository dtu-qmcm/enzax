"""Tests for writing enzax models as SBML.

The main check is a round trip: a model written by `enzax_to_sbml` and read
back by `sbml_to_enzax` has the same fluxes as the original.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from enzax import sbml
from enzax.examples import (
    conserved_moiety,
    gdp_fucose,
    glycolysis,
    linear,
    methionine,
    red_blood_cell,
)
from enzax.kinetic_model import KineticModel
from enzax.parameters import pack_parameters
from enzax.reactions import (
    Drain,
    MassAction,
    MichaelisMenten,
    SymbolicReaction,
)


def get_small_model():
    """A model with every kind of reaction the exporter handles."""
    model = KineticModel(
        reactions={
            "source": Drain(stoichiometry={"a": 1.0}),
            "r1": MassAction(stoichiometry={"a": -2.0, "b": 1.0}),
            "r2": SymbolicReaction(
                stoichiometry={"b": -1.0, "c": 1.0},
                expression="k * s * reversibility",
                species={"s": "b"},
                parameters={"k": "log_kcat"},
            ),
            "r3": SymbolicReaction(
                stoichiometry={"c": -1.0, "d": 1.0},
                expression="k * (s - p / keq)",
                species={"s": "c", "p": "d"},
                parameters={"k": "log_kcat"},
            ),
            "r4": MichaelisMenten(
                stoichiometry={"d": -1.0, "e": 1.0},
                allosteric_inhibitors=["a"],
            ),
        },
        balanced_species=["a", "b", "c", "d"],
    )
    parameters = pack_parameters(
        model.parameter_labelling,
        {
            "log_drain": {"source": 0.1},
            "log_k_plus": {"r1": -0.5},
            "log_kcat": {"r2": 0.2, "r3": -0.3, "r4": 0.4},
            "log_enzyme": {"r4": -1.0},
            "log_saturation_constant": dict.fromkeys(
                model.parameter_labelling["log_saturation_constant"],
                -0.7,
            ),
            "log_tc": {"r4": 0.5},
            "dgf": {"a": -3.0, "b": -1.0, "c": 2.0, "d": 0.5, "e": -2.0},
            "log_conc_unbalanced": {"e": -1.0},
            "temperature": 310.0,
        },
    )
    return model, parameters, jnp.array([0.5, 0.3, 0.2, 0.1])


EXAMPLES = {
    name: (
        example.model,
        example.parameters,
        example.model.get_balanced_conc(
            example.steady_state,
            example.parameters,
        ),
    )
    for name, example in [
        ("linear", linear),
        ("methionine", methionine),
        ("conserved_moiety", conserved_moiety),
        ("glycolysis", glycolysis),
    ]
}
EXAMPLES["small"] = get_small_model()


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_exported_model_has_no_sbml_errors(name):
    model, parameters, conc_balanced = EXAMPLES[name]
    doc = sbml.enzax_to_sbml(model, parameters, conc_balanced, name)
    doc.checkConsistency()
    errors = [
        doc.getError(i).getMessage()
        for i in range(doc.getNumErrors())
        if doc.getError(i).isError()
    ]
    assert not errors


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_round_trip_keeps_the_fluxes(name):
    model, parameters, conc_balanced = EXAMPLES[name]
    doc = sbml.enzax_to_sbml(model, parameters, conc_balanced, name)
    imported, imported_parameters = sbml.sbml_to_enzax(doc.getModel())
    rng = np.random.default_rng(0)
    conc = rng.lognormal(sigma=1.0, size=len(model.species))
    by_id = dict(zip(map(sbml.get_sbml_id, model.species), conc))
    imported_conc = jnp.array([by_id[s] for s in imported.species])
    expected = dict(
        zip(
            map(sbml.get_sbml_id, model.reaction_ids),
            np.asarray(
                model.flux(
                    jnp.asarray(conc)[model.balanced_species_ix],
                    parameters,
                ),
            ),
        ),
    )
    actual = dict(
        zip(
            imported.reaction_ids,
            np.asarray(
                imported.flux(
                    imported_conc[imported.balanced_species_ix],
                    imported_parameters,
                ),
            ),
        ),
    )
    assert actual.keys() == expected.keys()
    scale = max(abs(v) for v in expected.values())
    for reaction_id, value in expected.items():
        assert abs(actual[reaction_id] - value) <= 1e-12 * scale


@pytest.mark.parametrize(
    "example",
    [red_blood_cell, gdp_fucose],
    ids=["red_blood_cell", "gdp_fucose"],
)
def test_rapid_equilibria_are_not_supported(example):
    model = example.model
    with pytest.raises(NotImplementedError, match="rapid equilibria"):
        sbml.enzax_to_sbml(
            model,
            example.parameters,
            jnp.ones(len(model.balanced_species)),
        )
