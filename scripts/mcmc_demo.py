"""Demonstration of how to make a Bayesian kinetic model with enzax."""

import functools
import logging
import warnings

import arviz as az
import jax
from jax import numpy as jnp

from enzax.examples import methionine
from enzax.mcmc import (
    ObservationSet,
    AllostericMichaelisMentenPriorSet,
    get_idata,
    ind_prior_from_truth,
    posterior_logdensity_amm,
    run_nuts,
)
from enzax.steady_state import get_kinetic_model_steady_state

SEED = 1234

jax.config.update("jax_enable_x64", True)


def main():
    """Demonstrate How to make a Bayesian kinetic model with enzax."""
    structure = methionine.structure
    rate_equations = methionine.rate_equations
    true_parameters = methionine.parameters
    true_model = methionine.model
    default_state_guess = jnp.full((5,), 0.01)
    true_states = get_kinetic_model_steady_state(
        true_model, default_state_guess
    )
    prior = AllostericMichaelisMentenPriorSet(
        log_kcat=ind_prior_from_truth(true_parameters.log_kcat, 0.1),
        log_enzyme=ind_prior_from_truth(true_parameters.log_enzyme, 0.1),
        log_drain=ind_prior_from_truth(true_parameters.log_drain, 0.1),
        dgf=ind_prior_from_truth(true_parameters.dgf, 0.1),
        log_km=ind_prior_from_truth(true_parameters.log_km, 0.1),
        log_conc_unbalanced=ind_prior_from_truth(
            true_parameters.log_conc_unbalanced, 0.1
        ),
        temperature=ind_prior_from_truth(true_parameters.temperature, 0.1),
        log_ki=ind_prior_from_truth(true_parameters.log_ki, 0.1),
        log_transfer_constant=ind_prior_from_truth(
            true_parameters.log_transfer_constant, 0.1
        ),
        log_dissociation_constant=ind_prior_from_truth(
            true_parameters.log_dissociation_constant, 0.1
        ),
    )
    # get true concentration
    true_conc = jnp.zeros(methionine.structure.S.shape[0])
    true_conc = true_conc.at[methionine.structure.balanced_species].set(
        true_states
    )
    true_conc = true_conc.at[methionine.structure.unbalanced_species].set(
        jnp.exp(true_parameters.log_conc_unbalanced)
    )
    # get true flux
    true_flux = true_model.flux(true_states)
    # simulate observations
    error_conc = 0.03
    error_flux = 0.05
    error_enzyme = 0.03
    key = jax.random.key(SEED)
    obs_conc = jnp.exp(jnp.log(true_conc) + jax.random.normal(key) * error_conc)
    obs_enzyme = jnp.exp(
        true_parameters.log_enzyme + jax.random.normal(key) * error_enzyme
    )
    obs_flux = true_flux + jax.random.normal(key) * error_conc
    obs = ObservationSet(
        conc=obs_conc,
        flux=obs_flux,
        enzyme=obs_enzyme,
        conc_scale=error_conc,
        flux_scale=error_flux,
        enzyme_scale=error_enzyme,
    )
    pldf = functools.partial(
        posterior_logdensity_amm,
        obs=obs,
        prior=prior,
        structure=structure,
        rate_equations=rate_equations,
        guess=default_state_guess,
    )
    samples, info = run_nuts(
        pldf,
        key,
        true_parameters,
        num_warmup=200,
        num_samples=200,
        initial_step_size=0.0001,
        max_num_doublings=10,
        is_mass_matrix_diagonal=False,
        target_acceptance_rate=0.95,
    )
    idata = get_idata(
        samples, info, coords=methionine.coords, dims=methionine.dims
    )
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
        model_low = jnp.quantile(getattr(samples.position, param), 0.01, axis=0)
        model_high = jnp.quantile(
            getattr(samples.position, param), 0.99, axis=0
        )
        print(f" {param}:")
        print(f"  true value: {true_val}")
        print(f"  posterior 1%: {model_low}")
        print(f"  posterior 99%: {model_high}")


if __name__ == "__main__":
    main()
