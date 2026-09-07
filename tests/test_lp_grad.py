import functools
import json
from pathlib import Path

import jax
from jax import numpy as jnp

from enzax.examples import methionine
from enzax.statistical_modelling import enzax_log_density, prior_from_truth

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


def get_methionine_gradient():
    """Get the gradient of the methionine model's log posterior density."""
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_state_guess = jnp.full((5,), 0.01)
    error_conc = jnp.full_like(obs_conc, 0.03)
    error_flux = jnp.full_like(obs_flux, 0.05)
    error_enzyme = jnp.full_like(obs_enzyme, 0.03)
    measurement_values = obs_conc, obs_enzyme, obs_flux
    measurement_errors = error_conc, error_enzyme, error_flux
    measurements = tuple(zip(measurement_values, measurement_errors))
    prior = prior_from_truth(true_parameters, sd=0.1)  # pyright: ignore[reportArgumentType]
    posterior_log_density = jax.jit(
        functools.partial(
            enzax_log_density,
            model=true_model,
            split=None,
            measurements=measurements,
            prior=prior,
            guess=default_state_guess,
        )
    )
    return jax.jacrev(posterior_log_density)(true_parameters)


def test_lp_grad():
    gradient = get_methionine_gradient()
    expected_gradient = deserialize_jax_dict(methionine_pldf_grad_file)
    assert set(gradient.keys()) == set(expected_gradient.keys())
    for key, actual in gradient.items():
        assert jnp.isclose(actual, expected_gradient[key]).all(), key


if __name__ == "__main__":
    # Regenerate the expected gradient, e.g. after changing the model or the
    # parameter labels. Inspect the diff before committing it.
    with open(methionine_pldf_grad_file, "w") as f:
        f.write(serialize_jax_dict(get_methionine_gradient()))
    print(f"wrote {methionine_pldf_grad_file}")
