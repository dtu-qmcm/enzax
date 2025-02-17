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
from enzax.statistical_modelling import (
    enzax_log_density,
    FreeParamSpec,
    split_given_free,
)

SEED = 1234

jax.config.update("jax_enable_x64", True)


def main():
    """Demonstrate How to make a Bayesian kinetic model with enzax."""
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_guess = jnp.full((5,), 0.01)
    true_steady = get_kinetic_model_steady_state(true_model, default_guess)

    # get free and fixed parameter pytrees in the right format
    free_spec = [
        FreeParamSpec(path=("log_kcat", "r1"), ix=(), inits=jnp.array([0.3])),
        FreeParamSpec(path=("temperature",), ix=(), inits=jnp.array([3.3])),
        FreeParamSpec(path=("dgf",), ix=((0, 2),), inits=jnp.array([33.3])),
    ]
    free_params, fixed_params = split_given_free(true_parameters, free_spec)
    prior_mean = free_params
    prior_sd = jax.tree.map(lambda arr: jnp.full_like(arr, 0.1), prior_mean)
    prior = jax.tree.transpose(
        outer_treedef=jax.tree.structure(prior_mean),
        inner_treedef=None,
        pytree_to_transpose=[prior_mean, prior_sd],
    )
    # get true concentration
    true_conc = get_conc(
        true_steady,
        true_parameters["log_conc_unbalanced"],
        methionine.structure,
    )
    # get true flux
    true_flux = true_model.flux(true_steady)
    # simulate observations
    error_conc = 0.03
    error_flux = 0.05
    error_enzyme = 0.03
    key = jax.random.key(SEED)
    true_log_enz_flat, _ = ravel_pytree(true_parameters["log_enzyme"])
    key_conc, key_enz, key_flux, key_nuts = jax.random.split(key, num=4)
    obs_conc = jnp.exp(
        jnp.log(true_conc) + jax.random.normal(key_conc) * error_conc
    )
    obs_enzyme = jnp.exp(
        true_log_enz_flat + jax.random.normal(key_enz) * error_enzyme
    )
    obs_flux = true_flux + jax.random.normal(key_flux) * error_flux
    print(obs_conc)
    print(obs_enzyme)
    print(obs_flux)
    posterior_log_density = jax.jit(
        functools.partial(
            enzax_log_density,
            fixed_parameters=fixed_params,
            observations=[],
            prior=prior,
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
