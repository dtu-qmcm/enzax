from pathlib import Path
import json
import jax
from jax import numpy as jnp
from jaxtyping import Array, Scalar

from enzax.examples import methionine
from enzax.steady_state import get_steady_state
from enzax.statistical_modelling import enzax_log_density, prior_from_truth

import functools

jax.config.update("jax_enable_x64", True)
SEED = 1234

HERE = Path(__file__).parent
methionine_pldf_grad_file = HERE / "data" / "expected_methionine_gradient.json"

obs_conc = jnp.array(
    [
        3.99618131e-05,
        1.24186458e-03,
        9.44053469e-04,
        4.72041839e-04,
        2.92625684e-05,
        2.04876101e-07,
        1.37054850e-03,
        9.44053469e-08,
        3.32476221e-06,
        9.53494003e-07,
        2.11467977e-05,
        6.16881926e-06,
        2.97376843e-06,
        1.00785260e-03,
        4.72026734e-05,
        1.49849607e-03,
        1.15174523e-06,
        2.31424323e-04,
        2.11467977e-06,
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
obs_enzyme = jnp.array(
    [
        0.00097884,
        0.00100336,
        0.00105027,
        0.00099059,
        0.00096148,
        0.00107917,
        0.00104588,
        0.00138744,
        0.00107483,
        0.0009662,
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


def test_lp_grad():
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_state_guess = jnp.full((5,), 0.01)
    true_states = get_steady_state(
        true_model,
        default_state_guess,
        true_parameters,
    )
    # get true concentration
    true_conc = jnp.zeros(true_model.S.shape[0])
    true_conc = true_conc.at[true_model.balanced_species_ix].set(true_states)
    true_conc = true_conc.at[true_model.unbalanced_species_ix].set(
        jnp.exp(true_parameters["log_conc_unbalanced"])  # pyright: ignore[reportArgumentType]
    )
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
            fixed_parameters=None,
            measurements=measurements,
            prior=prior,
            guess=default_state_guess,
        )
    )
    gradient = jax.jacrev(posterior_log_density)(methionine.parameters)
    expected_gradient = deserialize_jax_dict(methionine_pldf_grad_file)
    for k, obs in gradient.items():
        exp = expected_gradient[k]
        if isinstance(obs, Scalar):
            assert jnp.isclose(obs, exp)
        elif isinstance(obs, Array):
            assert jnp.isclose(obs, exp).all()
        elif isinstance(obs, dict):
            for kk in obs.keys():
                if isinstance(obs[kk], list):
                    for o, e in zip(obs[kk], exp[kk]):
                        assert jnp.isclose(o, e).all()
                elif isinstance(obs[kk], Scalar):
                    assert jnp.isclose(obs[kk], exp[kk])
                elif len(obs[kk]) > 0:
                    assert jnp.isclose(obs[kk], exp[kk]).all()


if __name__ == "__main__":
    test_lp_grad()
