import functools
import json
from pathlib import Path

import jax
from jax import numpy as jnp

from enzax.examples import methionine
from enzax.statistical_modelling import (
    enzax_log_density,
    enzax_log_density_grapevine,
    prior_from_truth,
)
from enzax.steady_state import get_steady_state

jax.config.update("jax_enable_x64", True)
SEED = 1234

HERE = Path(__file__).parent
methionine_pldf_grad_file = HERE / "data" / "expected_methionine_gradient.json"

obs_conc = jnp.array(
    [
        3.99618131e-05,  # met-L
        1.24186458e-03,  # atp
        9.44053469e-04,  # pi
        4.72041839e-04,  # ppi
        2.92625684e-05,  # amet
        2.04876101e-07,  # ahcys
        1.37054850e-03,  # gly
        9.44053469e-08,  # sarcs
        3.32476221e-06,  # hcys-L
        9.53494003e-07,  # adn
        2.11467977e-05,  # thf
        6.16881926e-06,  # 5mthf
        1.00785260e-03,  # glyb
        4.72026734e-05,  # dmgly
        1.49849607e-03,  # ser-L
        2.11467977e-06,  # cyst-L
        2.97376843e-06,  # mlthf
        1.15174523e-06,  # nadp
        2.31424323e-04,  # nadph
    ],
    dtype=jnp.float64,
)
obs_flux = jnp.array(
    [
        -0.00425181,
        0.03739644,
        0.01397071,
        -0.04154405,
        -0.05396867,
        0.01236334,
        -0.07089178,
        -0.02136595,
        0.00152784,
        -0.02482788,
        -0.01588131,
    ],
    dtype=jnp.float64,
)
# In `model.parameter_labelling["log_enzyme"]` order, i.e. the order the model's
# rate equations first label their enzymes in. Note that
# this is not the same as the order of `obs_flux`, which includes the drain
# reaction.
obs_enzyme = jnp.array(
    [
        0.00097884,  # MAT1
        0.00100336,  # MAT3
        0.00105027,  # METH-Gen
        0.00099059,  # GNMT1
        0.00096148,  # AHC1
        0.00107917,  # MS1
        0.00104588,  # BHMT1
        0.00138744,  # CBS1
        0.00107483,  # MTHFR1
        0.0009662,  # PROT1
    ],
    dtype=jnp.float64,
)


class JAXEncoder(json.JSONEncoder):
    def default(self, obj):  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(obj, jnp.ndarray):
            return {
                "_type": "jax_array",
                "data": obj.tolist(),
                "shape": obj.shape,
                "dtype": str(obj.dtype),
            }
        return super().default(obj)


def serialize_jax_dict(jax_dict):
    return json.dumps(jax_dict, cls=JAXEncoder)


def deserialize_jax_dict(file_path):
    def object_hook(dct):
        if "_type" in dct and dct["_type"] == "jax_array":
            return jnp.array(dct["data"], dtype=dct["dtype"])
        return dct

    with open(file_path, "r") as f:
        return json.load(f, object_hook=object_hook)


DEFAULT_STATE_GUESS = jnp.full((5,), 0.01)


def get_methionine_measurements_and_prior():
    """Get the methionine model's measurements and prior."""
    error_conc = jnp.full_like(obs_conc, 0.03)
    error_flux = jnp.full_like(obs_flux, 0.05)
    error_enzyme = jnp.full_like(obs_enzyme, 0.03)
    measurement_values = obs_conc, obs_enzyme, obs_flux
    measurement_errors = error_conc, error_enzyme, error_flux
    measurements = tuple(zip(measurement_values, measurement_errors))
    prior = prior_from_truth(methionine.parameters, sd=0.1)  # pyright: ignore[reportArgumentType]
    return measurements, prior


def get_methionine_gradient():
    """Get the gradient of the methionine model's log posterior density."""
    true_parameters = methionine.parameters
    true_model = methionine.model
    measurements, prior = get_methionine_measurements_and_prior()
    posterior_log_density = jax.jit(
        functools.partial(
            enzax_log_density,
            model=true_model,
            split=None,
            measurements=measurements,
            prior=prior,
            guess=DEFAULT_STATE_GUESS,
            # Tighter than `get_steady_state`'s defaults, which are set for the
            # speed of a sampling run. At those defaults this gradient varies
            # between platforms by far more than the tolerance asserted below.
            ivp_rtol=1e-11,
            ivp_atol=1e-11,
            steady_state_rtol=1e-11,
            steady_state_atol=1e-11,
        )
    )
    return jax.jacrev(posterior_log_density)(true_parameters)


def test_lp_grad():
    gradient = get_methionine_gradient()
    expected_gradient = deserialize_jax_dict(methionine_pldf_grad_file)
    assert set(gradient.keys()) == set(expected_gradient.keys())
    for key, actual in gradient.items():
        assert jnp.isclose(actual, expected_gradient[key]).all(), key


def get_methionine_log_density_and_grad(guess):
    """Get the methionine log posterior density and its gradient."""
    measurements, prior = get_methionine_measurements_and_prior()
    posterior_log_density = functools.partial(
        enzax_log_density_grapevine,
        model=methionine.model,
        split=None,
        measurements=measurements,
        prior=prior,
        guess=guess,
    )
    gradient, _ = jax.jacrev(posterior_log_density, has_aux=True)(
        methionine.parameters
    )
    log_density, _ = posterior_log_density(methionine.parameters)
    return log_density, gradient


def test_log_density_is_guess_invariant():
    """Check that the guess does not change the target grapevine samples.

    The grapevine method is only valid if the solver reaches the same answer
    whatever guess it starts from. `get_steady_state` stops once
    `norm(dcdt) < atol + rtol * norm(conc)`, so its terminal state does depend
    on where it started, and this test bounds by how much. Methionine's
    concentrations are of order 1e-5, so the event tolerances have to be
    tight relative to that: at 1e-9 the log density moves by ~1e-3 between
    guesses, which is why `get_steady_state` defaults to 1e-12.
    """
    steady = get_steady_state(
        methionine.model,
        DEFAULT_STATE_GUESS,
        methionine.parameters,
    )
    expected_lp, expected_grad = get_methionine_log_density_and_grad(
        DEFAULT_STATE_GUESS
    )
    guesses = {
        "steady state": steady,
        "perturbed up": steady * 1.5,
        "perturbed down": steady * 0.5,
        "distant": jnp.full((5,), 0.001),
    }
    for name, guess in guesses.items():
        log_density, gradient = get_methionine_log_density_and_grad(guess)
        assert jnp.isclose(log_density, expected_lp, rtol=1e-9, atol=1e-6), name
        for key, actual in gradient.items():
            assert jnp.isclose(
                actual, expected_grad[key], rtol=1e-5, atol=1e-6
            ).all(), f"{name}, {key}"


if __name__ == "__main__":
    # Regenerate the expected gradient, e.g. after changing the model or the
    # parameter labels. Inspect the diff before committing it. Do not
    # regenerate it at looser tolerances than the ones set above: the values
    # would then only reproduce on the machine that wrote them.
    with open(methionine_pldf_grad_file, "w") as f:
        f.write(serialize_jax_dict(get_methionine_gradient()))
    print(f"wrote {methionine_pldf_grad_file}")
