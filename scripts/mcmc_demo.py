"""Fit a kinetic model to simulated data with enzax and grapevine.

Change the `enzax.examples` import below to fit a different model.
"""

# ruff: noqa: E402

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import functools

import blackjax
import jax
from blackjax_utils import run_sampler
from grapevine import grapenuts, guess_implicit
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from enzax.examples import methionine as example
from enzax.parameter_split import (
    combine_parameters,
    count_free_parameters,
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.statistical_modelling import (
    enzax_log_density_grapevine,
    prior_from_truth,
)
from enzax.steady_state import get_steady_state_hybrid

jax.config.update("jax_enable_x64", True)

SEED = 1234
N_CHAIN = 4
N_WARMUP = 2
N_SAMPLE = 2
MAX_TREEDEPTH = 10
INIT_SD = 0.01
INITIAL_STEP_SIZE = 0.001
TARGET_ACCEPTANCE = 0.95
FREE_PARAMETERS = {"log_kcat": None}
PRIOR_SD = 0.1
CONC_ERROR = 0.03
ENZYME_ERROR = 0.03
FLUX_ERROR = 0.05


def simulate(key, truth, error):
    key_conc, key_enzyme, key_flux = jax.random.split(key, 3)
    true_conc, true_log_enzyme, true_flux = truth
    conc_error, enzyme_error, flux_error = error
    return (
        jnp.exp(jnp.log(true_conc) + jax.random.normal(key_conc) * conc_error),
        jnp.exp(true_log_enzyme + jax.random.normal(key_enzyme) * enzyme_error),
        true_flux + jax.random.normal(key_flux) * flux_error,
    )


def get_guess_fn(model, split, free_parameters):
    _, unflatten = ravel_pytree(free_parameters)

    def target_function(conc_ind, position):
        parameters = combine_parameters(split, unflatten(position))
        return model.dcdt(conc_ind, parameters)

    return functools.partial(guess_implicit, target_function=target_function)


def report(split, free_true, states):
    n_free = count_free_parameters(split)
    print(f"True values against the posterior ({n_free} free):")
    for (path, true), draws in zip(
        jax.tree.leaves_with_path(free_true),
        jax.tree.leaves(states.position),
    ):
        pooled = draws.reshape(-1, *draws.shape[2:])
        low = jnp.quantile(pooled, 0.01, axis=0)
        high = jnp.quantile(pooled, 0.99, axis=0)
        covered = int(jnp.sum((true >= low) & (true <= high)))
        print(f"  {path[0].key}: {covered}/{jnp.size(true)} covered")


def main():
    model = example.model
    true_parameters = example.parameters
    default_guess = example.steady_state
    split = split_parameters_by_freeing(
        model.parameter_labelling, true_parameters, FREE_PARAMETERS
    )
    free_true = get_free_parameters(split, true_parameters)
    prior = prior_from_truth(free_true, sd=PRIOR_SD)
    steady = get_steady_state_hybrid(model, default_guess, true_parameters)
    balanced = model.get_balanced_conc(
        steady, model.get_moiety_totals(true_parameters)
    )
    true_conc = model.get_conc(
        balanced, model.get_log_conc_unbalanced(true_parameters)
    )
    true_flux = model.flux(balanced, true_parameters)
    true_log_enzyme = true_parameters["log_enzyme"]
    errors = (
        jnp.full_like(true_conc, CONC_ERROR),
        jnp.full_like(true_log_enzyme, ENZYME_ERROR),
        jnp.full_like(true_flux, FLUX_ERROR),
    )
    key_sim, key_mcmc = jax.random.split(jax.random.key(SEED), 2)
    values = simulate(key_sim, (true_conc, true_log_enzyme, true_flux), errors)
    measurements = tuple(zip(values, errors))
    log_density = functools.partial(
        enzax_log_density_grapevine,
        model=model,
        split=split,
        measurements=measurements,
        prior=prior,
    )
    sampler = grapenuts(
        default_guess, guess_fn=get_guess_fn(model, split, free_true)
    )
    with blackjax.progress_bar("enzax"):
        states, info = run_sampler(
            key=key_mcmc,
            log_posterior=log_density,
            init_params=free_true,
            init_sd=INIT_SD,
            n_chain=N_CHAIN,
            n_warmup=N_WARMUP,
            n_sample=N_SAMPLE,
            max_num_doublings=MAX_TREEDEPTH,
            sampler=sampler,
            warmup_options=dict(
                initial_step_size=INITIAL_STEP_SIZE,
                is_mass_matrix_diagonal=True,
                target_acceptance_rate=TARGET_ACCEPTANCE,
            ),
        )
    print(f"Divergent transitions: {int(info.is_divergent.sum())}")
    report(split, free_true, states)


if __name__ == "__main__":
    main()
