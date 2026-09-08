import functools
import logging
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

import blackjax
from blackjax_utils import run_nuts

from enzax.examples.smallbone import (
    enzax_log_density_sbml,
    get_conc_assingment_species,
    load_smallbone,
)
from enzax.statistical_modelling import prior_from_truth
from enzax.steady_state import get_steady_state

SEED = 1234
N_CHAIN = 4
N_WARMUP = 10
N_SAMPLE = 10

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_free_params(params: PyTree) -> PyTree:
    return (
        params["kcat_ADH_ADH1"],
        params["Vmax_ATPase"],
        params["kcat_ENO_ENO1"],
        params["kcat_ENO_ENO2"],
        params["kcat_FBA"],
        params["Vmax_GPD"],
        params["kcat_GPM"],
        params["Vmax_GPP"],
        params["kcat_HXK_HXK1"],
        params["kcat_HXK_HXK2"],
        params["kcat_PDC_PDC1"],
        params["kcat_PDC_PDC5"],
        params["kcat_PDC_PDC6"],
        params["kcat_PFK"],
        params["kcat_PGI"],
        params["kcat_PGK"],
        params["Vmax_PGM"],
        params["kcat_PYK_CDC19"],
        params["kcat_TDH_TDH1"],
        params["kcat_TDH_TDH2"],
        params["kcat_TDH_TDH3"],
        params["kcat_TPI"],
        params["Vmax_TPP"],
        params["Vmax_TPS"],
        params["Vmax_UGP"],
        params["Vmax_HXT"],
    )


def simulate(key, truth, error):
    key_conc, key_flux = jax.random.split(key, num=2)
    true_conc, true_flux = truth
    conc_err, flux_err = error
    return (
        jnp.exp(jnp.log(true_conc) + jax.random.normal(key_conc) * conc_err),
        true_flux + jax.random.normal(key_flux) * flux_err,
    )


def main():
    model, parameters, initial_conc = load_smallbone()

    y0 = initial_conc
    print(f"Flux at {str(y0)}: " + str(model.flux(y0, parameters)))

    print(f"dcdt at {str(y0)}, t=1: " + str(model.dcdt(y0, parameters)))

    guess = initial_conc
    steady_state = get_steady_state(model, guess, parameters)
    print("Steady state: " + str(steady_state))

    parameters_log = jax.tree.map(lambda x: jnp.log(x), parameters)

    is_free = eqx.tree_at(
        get_free_params,
        jax.tree.map(lambda _: False, parameters),
        replace_fn=lambda _: True,
    )

    free_params, fixed_params = eqx.partition(parameters, is_free)
    free_params_log, _ = eqx.partition(parameters_log, is_free)
    prior_log = prior_from_truth(free_params_log, sd=0.1)

    true_conc = get_conc_assingment_species(steady_state, parameters, model)

    true_flux = model.flux(steady_state, parameters)

    # simulate observations
    conc_err = jnp.full_like(true_conc, 0.03)
    flux_err = jnp.full_like(true_flux, 0.05)
    key = jax.random.key(SEED)
    key_sim, key_nuts = jax.random.split(key, num=2)
    measurement_errors = (conc_err, flux_err)
    measurement_values = simulate(
        key=key_sim,
        truth=(true_conc, true_flux),
        error=measurement_errors,
    )
    measurements = tuple(zip(measurement_values, measurement_errors))

    posterior_log_density = functools.partial(
        enzax_log_density_sbml,
        model=model,
        fixed_parameters=fixed_params,
        measurements=measurements,
        prior_log=prior_log,
        guess=guess,
    )

    # blackjax-utils runs the sampler; enzax's job ends at the log density.
    # `max_num_doublings` goes to both warmup and sampling; the contents of
    # `warmup_options` reach `window_adaptation` alone. See mcmc_demo.py for
    # why the initial step size is set where it is, and for what the progress
    # bar needs in order to see the sampler's scan.
    with blackjax.progress_bar("enzax NUTS"):
        states, info = run_nuts(
            key=key_nuts,
            log_posterior=posterior_log_density,
            init_params=free_params_log,
            n_chain=N_CHAIN,
            n_warmup=N_WARMUP,
            n_sample=N_SAMPLE,
            max_num_doublings=10,
            warmup_options=dict(
                initial_step_size=0.01,
                is_mass_matrix_diagonal=False,
                target_acceptance_rate=0.95,
            ),
        )

    if jnp.any(info.is_divergent):
        n_divergent = info.is_divergent.sum()
        msg = f"There were {n_divergent} post-warmup divergent transitions."
        warnings.warn(msg)
    else:
        logging.info("No post-warmup divergent transitions!")
    print("True parameter values vs posterior:")
    for (path, leaf_true), leaf_model in zip(
        jax.tree.leaves_with_path(free_params), jax.tree.leaves(states.position)
    ):
        # Leaves arrive with shape (n_chain, n_sample, ...): pool the draws
        # from every chain before summarising them.
        draws = leaf_model.reshape(-1, *leaf_model.shape[2:])
        model_low = jnp.quantile(draws, 0.01, axis=0)
        model_high = jnp.quantile(draws, 0.99, axis=0)
        print(f" {'|'.join(k.key for k in path)}:")
        print(f"  true value: {leaf_true}")
        print(f"  posterior 1%: {model_low}")
        print(f"  posterior 99%: {model_high}")


if __name__ == "__main__":
    main()
