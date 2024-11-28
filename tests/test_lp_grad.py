import json
import jax
from jax import numpy as jnp
from jax.scipy.stats import norm
from jax.flatten_util import ravel_pytree

from enzax.examples import methionine
from enzax.kinetic_model import RateEquationModel
from enzax.mcmc import ObservationSet
from enzax.steady_state import get_kinetic_model_steady_state

import importlib.resources
from tests import data

import functools

jax.config.update("jax_enable_x64", True)
SEED = 1234

methionine_pldf_grad_file = (
    importlib.resources.files(data) / "expected_methionine_gradient.json"
)

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


def test_lp_grad():
    structure = methionine.structure
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_state_guess = jnp.full((5,), 0.01)
    true_states = get_kinetic_model_steady_state(
        true_model, default_state_guess
    )
    flat_true_params, _ = ravel_pytree(methionine.parameters)
    # get true concentration
    true_conc = jnp.zeros(methionine.structure.S.shape[0])
    true_conc = true_conc.at[methionine.structure.balanced_species_ix].set(
        true_states
    )
    true_conc = true_conc.at[methionine.structure.unbalanced_species_ix].set(
        jnp.exp(true_parameters.log_conc_unbalanced)
    )
    error_conc = 0.03
    error_flux = 0.05
    error_enzyme = 0.03
    obs = ObservationSet(
        conc=obs_conc,
        flux=obs_flux,
        enzyme=obs_enzyme,
        conc_scale=error_conc,
        flux_scale=error_flux,
        enzyme_scale=error_enzyme,
    )

    def joint_log_density(params, prior_mean, prior_sd, obs):
        flat_params, _ = ravel_pytree(params)
        model = RateEquationModel(params, methionine.structure)
        steady = get_kinetic_model_steady_state(model, default_state_guess)
        unbalanced = jnp.exp(params.log_conc_unbalanced)
        conc = jnp.zeros(structure.S.shape[0])
        conc = conc.at[structure.balanced_species_ix].set(steady)
        conc = conc.at[structure.unbalanced_species_ix].set(unbalanced)
        flux = model.flux(steady)
        log_prior = norm.pdf(flat_params, prior_mean, prior_sd).sum()
        flat_log_enzyme, _ = ravel_pytree(params.log_enzyme)
        log_liklihood = jnp.sum(
            jnp.array(
                [
                    norm.logpdf(
                        jnp.log(obs.conc), jnp.log(conc), obs.conc_scale
                    ).sum(),
                    norm.logpdf(
                        jnp.log(obs.enzyme), flat_log_enzyme, obs.enzyme_scale
                    ).sum(),
                    norm.logpdf(obs.flux, flux, obs.flux_scale).sum(),
                ]
            )
        )
        return log_prior + log_liklihood

    posterior_log_density = functools.partial(
        joint_log_density,
        prior_mean=flat_true_params,
        prior_sd=0.1,
        obs=obs,
    )
    gradient = jax.jacrev(posterior_log_density)(methionine.parameters)
    _, grad_pytree_def = ravel_pytree(gradient)
    with open(methionine_pldf_grad_file, "r") as file:
        expected_gradient = grad_pytree_def(jnp.array(json.load(file)))
    assert gradient == expected_gradient
