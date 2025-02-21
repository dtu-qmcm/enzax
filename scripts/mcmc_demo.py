"""Demonstration of how to make a Bayesian kinetic model with enzax."""

import functools
import logging
import warnings


import arviz as az
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from enzax.examples import methionine
from enzax.kinetic_model import get_conc
from enzax.mcmc import get_idata, run_nuts
from enzax.steady_state import get_kinetic_model_steady_state
from enzax.statistical_modelling import enzax_log_density, prior_from_truth

import equinox as eqx

SEED = 1234

jax.config.update("jax_enable_x64", True)


def simulate(key, truth, error):
    key_conc, key_enz, key_flux = jax.random.split(key, num=3)
    true_conc, true_log_enz, true_flux = truth
    conc_err, enz_err, flux_err = error
    return (
        jnp.exp(jnp.log(true_conc) + jax.random.normal(key_conc) * conc_err),
        jnp.exp(true_log_enz + jax.random.normal(key_enz) * enz_err),
        true_flux + jax.random.normal(key_flux) * flux_err,
    )


def main():
    """Demonstrate How to make a Bayesian kinetic model with enzax."""
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_guess = jnp.full((5,), 0.01)
    true_steady = get_kinetic_model_steady_state(true_model, default_guess)

    def get_free_params(params):
        return (
            params["log_kcat"]["MAT1"],
            params["temperature"],
            params["dgf"],
        )

    is_free = eqx.tree_at(
        get_free_params,
        jax.tree.map(lambda _: False, true_parameters),
        replace_fn=lambda _: True,
    )
    free_params, fixed_params = eqx.partition(true_parameters, is_free)
    is_mv = eqx.tree_at(
        lambda params: params["dgf"],
        jax.tree.map(lambda _: False, free_params),
        replace=True,
    )
    prior = prior_from_truth(free_params, sd=0.1, is_multivariate=is_mv)
    # get true concentration, flux and log enzyme
    true_conc = get_conc(
        true_steady,
        true_parameters["log_conc_unbalanced"],
        methionine.structure,
    )
    true_flux = true_model.flux(true_steady)
    true_log_enz_flat, _ = ravel_pytree(true_parameters["log_enzyme"])
    # simulate observations
    conc_err = jnp.full_like(true_conc, 0.03)
    flux_err = jnp.full_like(true_flux, 0.05)
    enz_err = jnp.full_like(true_log_enz_flat, 0.03)
    key = jax.random.key(SEED)
    key_sim, key_nuts = jax.random.split(key, num=2)
    measurement_errors = (conc_err, enz_err, flux_err)
    measurement_values = simulate(
        key=key_sim,
        truth=(true_conc, true_log_enz_flat, true_flux),
        error=measurement_errors,
    )
    measurements = tuple(zip(measurement_values, measurement_errors))
    posterior_log_density = jax.jit(
        functools.partial(
            enzax_log_density,
            structure=true_model.structure,
            fixed_parameters=fixed_params,
            measurements=measurements,
            prior=prior,
            guess=default_guess,
        )
    )
    samples, info = run_nuts(
        posterior_log_density,
        key_nuts,
        free_params,
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
    for param in free_params.keys():
        true_val = true_parameters[param]
        model_p = samples.position[param]
        if isinstance(true_val, Array):
            model_low = jnp.quantile(model_p, 0.01, axis=0)
            model_high = jnp.quantile(model_p, 0.99, axis=0)
        elif isinstance(true_val, dict):
            model_low, model_high = (
                {k: jnp.quantile(v, q, axis=0) for k, v in model_p.items()}
                for q in (0.01, 0.99)
            )
        else:
            raise ValueError("Unexpectd parameter type")
        print(f" {param}:")
        print(f"  true value: {true_val}")
        print(f"  posterior 1%: {model_low}")
        print(f"  posterior 99%: {model_high}")


if __name__ == "__main__":
    main()
