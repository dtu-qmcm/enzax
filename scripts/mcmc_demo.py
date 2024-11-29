"""Demonstration of how to make a Bayesian kinetic model with enzax."""

import functools
import logging
import warnings

import arviz as az
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import norm
from jaxtyping import Array

from enzax.examples import methionine
from enzax.kinetic_model import RateEquationModel, get_conc
from enzax.mcmc import (
    ObservationSet,
    get_idata,
    run_nuts,
)
from enzax.steady_state import get_kinetic_model_steady_state

SEED = 1234

jax.config.update("jax_enable_x64", True)


def joint_log_density(params, prior_mean, prior_sd, obs, guess):
    # find the steady state concentration and flux
    model = RateEquationModel(params, methionine.structure)
    steady = get_kinetic_model_steady_state(model, guess)
    conc = get_conc(steady, params.log_conc_unbalanced, methionine.structure)
    flux = model.flux(steady)
    # prior
    flat_params, _ = ravel_pytree(params)
    log_prior = norm.logpdf(flat_params, loc=prior_mean, scale=prior_sd).sum()
    # likelihood
    flat_log_enzyme, _ = ravel_pytree(params.log_enzyme)
    log_likelihood = (
        norm.logpdf(jnp.log(obs.conc), jnp.log(conc), obs.conc_scale).sum()
        + norm.logpdf(
            jnp.log(obs.enzyme), flat_log_enzyme, obs.enzyme_scale
        ).sum()
        + norm.logpdf(obs.flux, flux, obs.flux_scale).sum()
    )
    return log_prior + log_likelihood


def main():
    """Demonstrate How to make a Bayesian kinetic model with enzax."""
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_guess = jnp.full((5,), 0.01)
    true_steady = get_kinetic_model_steady_state(true_model, default_guess)
    # get true concentration
    true_conc = get_conc(
        true_steady,
        true_parameters.log_conc_unbalanced,
        methionine.structure,
    )
    # get true flux
    true_flux = true_model.flux(true_steady)
    # simulate observations
    error_conc = 0.03
    error_flux = 0.05
    error_enzyme = 0.03
    key = jax.random.key(SEED)
    true_log_enz_flat, _ = ravel_pytree(true_parameters.log_enzyme)
    key_conc, key_enz, key_flux, key_nuts = jax.random.split(key, num=4)
    obs_conc = jnp.exp(
        jnp.log(true_conc) + jax.random.normal(key_conc) * error_conc
    )
    obs_enzyme = jnp.exp(
        true_log_enz_flat + jax.random.normal(key_enz) * error_enzyme
    )
    obs_flux = true_flux + jax.random.normal(key_flux) * error_conc
    obs = ObservationSet(
        conc=obs_conc,
        flux=obs_flux,
        enzyme=obs_enzyme,
        conc_scale=error_conc,
        flux_scale=error_flux,
        enzyme_scale=error_enzyme,
    )
    flat_true_params, _ = ravel_pytree(true_parameters)
    posterior_log_density = jax.jit(
        functools.partial(
            joint_log_density,
            obs=obs,
            prior_mean=flat_true_params,
            prior_sd=0.1,
            guess=default_guess,
        )
    )
    samples, info = run_nuts(
        posterior_log_density,
        key_nuts,
        true_parameters,
        num_warmup=200,
        num_samples=200,
        initial_step_size=0.0001,
        max_num_doublings=10,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=0.95,
    )
    idata = get_idata(samples, info)
    print(az.summary(idata))
    if jnp.any(info.is_divergent):
        n_divergent = info.is_divergent.sum()
        msg = f"There were {n_divergent} post-warmup divergent transitions."
        warnings.warn(msg)
    else:
        logging.info("No post-warmup divergent transitions!")
    print("True parameter values vs posterior:")
    for param in true_parameters.__dataclass_fields__.keys():
        true_val = getattr(true_parameters, param)
        model_p = getattr(samples.position, param)
        if isinstance(true_val, Array):
            model_low = jnp.quantile(model_p, 0.01, axis=0)
            model_high = jnp.quantile(model_p, 0.99, axis=0)
        elif isinstance(true_val, dict):
            model_low, model_high = (
                {k: jnp.quantile(v, q, axis=0) for k, v in model_p.items()}
                for q in (0.01, 0.99)
            )
        print(f" {param}:")
        print(f"  true value: {true_val}")
        print(f"  posterior 1%: {model_low}")
        print(f"  posterior 99%: {model_high}")


if __name__ == "__main__":
    main()
