"""Guardrail for changes that are meant to leave every number alone.

Assertions here are exact equality rather than `isclose`: a refactor that
reorders a sum or a product changes the last bits, and this is the test that
says so.

The expected values in `data/expected_fluxes.json` were captured before the
binding polynomial work started, and regenerated once, when the Monod Wyman
Changeux factor became a ratio of two binding polynomials. That step replaced
`free_enzyme_ratio * qnum / qdenom` with `Z_T / Z_R`, which is the same
quantity associated differently, so every allosteric reaction moved by a
couple of units in the last place -- at most 2 ulps, or 3.6e-16 relative, over
the three examples. They were regenerated for jax 0.12.0, which moved the
linear example's fluxes by 1 ulp. Nothing else has changed them.
"""

import json
from pathlib import Path

import jax
import pytest
from jax import numpy as jnp

from enzax.examples import conserved_moiety, linear, methionine

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
expected_flux_file = HERE / "data" / "expected_fluxes.json"


def get_expected(name: str) -> dict[str, list[float]]:
    with open(expected_flux_file, "r") as f:
        return json.load(f)[name]


@pytest.mark.parametrize(
    ["name", "example"],
    [
        ("linear", linear),
        ("methionine", methionine),
        ("conserved_moiety", conserved_moiety),
    ],
)
def test_flux_and_dndt_are_unchanged(name, example):
    model = example.model
    parameters = example.parameters
    expected = get_expected(name)
    conc_balanced = model.get_balanced_conc(example.steady_state, parameters)
    flux = model.flux(conc_balanced, parameters)
    dndt = model.dndt(example.steady_state, parameters)
    assert jnp.array_equal(flux, jnp.array(expected["flux"]))
    assert jnp.array_equal(dndt, jnp.array(expected["dndt"]))
