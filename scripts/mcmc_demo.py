"""Demonstration of how to make a Bayesian kinetic model with enzax."""

import functools
import logging
import warnings

import equinox as eqx
import jax
from jax import numpy as jnp

from enzax.examples import methionine
from enzax.mcmc import run_nuts
from enzax.parameter_split import (
    count_free_parameters,
    get_free_labels,
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.statistical_modelling import enzax_log_density, prior_from_truth
from enzax.steady_state import get_steady_state

SEED = 1234

jax.config.update("jax_enable_x64", True)


# The parameters to infer: everything else is held at its true value.
#
# A parameter mapped to a list of labels frees just those values; a parameter
# mapped to None frees the whole thing. So this infers MAT1's turnover number,
# but not any other enzyme's.
FREE_PARAMETERS = {
    "log_kcat": ["MAT1"],
    "temperature": None,
    "dgf": None,
}


def simulate(key, truth, error):
    """Simulate observations from the true model.

    Args:
        key: jax.random key
        truth: tuple of true concentration, log enzyme and flux
        error: tuple of concentration, enzyme and flux error

    """
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
    model = methionine.model
    default_guess = jnp.full((5,), 0.01)
    true_steady = get_steady_state(model, default_guess, true_parameters)
    split = split_parameters_by_freeing(
        model.parameter_labels,
        true_parameters,
        FREE_PARAMETERS,
    )
    free_params = get_free_parameters(split, true_parameters)
    is_mv = eqx.tree_at(
        lambda params: params["dgf"],
        jax.tree.map(lambda _: False, free_params),
        replace=True,
    )
    prior = prior_from_truth(free_params, sd=0.1, is_multivariate=is_mv)
    # get true concentration, flux and log enzyme
    true_conc = methionine.model.get_conc(
        true_steady,
        model.get_log_conc_unbalanced(true_parameters),
    )
    true_flux = model.flux(true_steady, methionine.parameters)
    # Already flat, and in `model.parameter_labels["log_enzyme"]` order,
    # which is the order enzyme measurements have to be given in.
    true_log_enz = true_parameters["log_enzyme"]
    # simulate observations
    conc_err = jnp.full_like(true_conc, 0.03)
    flux_err = jnp.full_like(true_flux, 0.05)
    enz_err = jnp.full_like(true_log_enz, 0.03)
    key = jax.random.key(SEED)
    key_sim, key_nuts = jax.random.split(key, num=2)
    measurement_errors = (conc_err, enz_err, flux_err)
    measurement_values = simulate(
        key=key_sim,
        truth=(true_conc, true_log_enz, true_flux),
        error=measurement_errors,
    )
    measurements = tuple(zip(measurement_values, measurement_errors))
    posterior_log_density = functools.partial(
        enzax_log_density,
        model=model,
        split=split,
        measurements=measurements,
        prior=prior,
        guess=default_guess,
    )
    states, info = run_nuts(
        posterior_log_density,
        key_nuts,
        free_params,
        num_warmup=2,
        num_samples=2,
        initial_step_size=0.0001,
        max_num_doublings=10,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=0.95,
    )
    if jnp.any(info.is_divergent):
        n_divergent = info.is_divergent.sum()
        msg = f"There were {n_divergent} post-warmup divergent transitions."
        warnings.warn(msg)
    else:
        logging.info("No post-warmup divergent transitions!")
    n_free = count_free_parameters(split)
    print(f"True parameter values vs posterior ({n_free} free):")
    for (path, leaf_true), leaf_model in zip(
        jax.tree.leaves_with_path(free_params), jax.tree.leaves(states.position)
    ):
        parameter = path[0].key
        model_low = jnp.quantile(leaf_model, 0.01, axis=0)
        model_high = jnp.quantile(leaf_model, 0.99, axis=0)
        labels = get_free_labels(split, parameter)
        print(f" {parameter}:")
        if jnp.ndim(leaf_true) == 0:
            print(f"  true value: {leaf_true}")
            print(f"  posterior 1%: {model_low}")
            print(f"  posterior 99%: {model_high}")
        else:
            for label, true, low, high in zip(
                labels, leaf_true, model_low, model_high
            ):
                print(
                    f"  {label}: true {true:.4g}, "
                    f"posterior 1% {low:.4g}, 99% {high:.4g}"
                )


if __name__ == "__main__":
    main()
