"""Compute derived quantities from the posterior draws.

For every posterior draw, finds each fitted experiment's steady state. Records
its concentrations, fluxes, pool MIDs, the largest real part of its Jacobian's
eigenvalues and each measurement's log likelihood. For fits from the first two
rounds, also predicts the next round's experiments and records each of their
measurements' log likelihoods. Then prints the log predictive density of those
experiments, which sums, over measurements, the log of the likelihood averaged
over draws. Runs once for each fit in common.FITS, saving to
results/<fit>/derived.npz.
"""

import sys
import time

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp

from common import (
    FITS,
    HELD_OUT,
    MEASURED_POOLS,
    MODELS,
    POOL_SEPARATOR,
    REFERENCE_EXPERIMENT,
    get_experiment_parameters,
    get_mid_simulator,
    get_results_dir,
    get_z_posterior,
    load_measurements,
    solve,
)
from enzax.experiments import gather_parameters_by_experiment
from enzax.measurements import (
    get_pool_mid,
    pick_concs,
    pick_enzymes,
    pick_fluxes,
)
from enzax.parameter_split import combine_parameters
from enzax.thermodynamics import get_one_way_fluxes
from fit import N_CHAIN, get_chain_file


def get_derived_file(fit):
    return get_results_dir(fit) / "derived.npz"


def load_draws(fit):
    return np.concatenate(
        [
            np.load(get_chain_file(fit, chain))["draws"]
            for chain in range(N_CHAIN)
        ],
    )


def get_growth_rate(model, parameters, steady):
    jacobian = jax.jacfwd(model.dndt)(steady, parameters)
    return jnp.linalg.eigvals(jacobian).real.max()


def score_experiment(model, measurements, experiment, parameters, solved):
    found, conc, flux, mids = solved
    picks = {
        "conc_measurements": (pick_concs, conc),
        "enzyme_measurements": (pick_enzymes, parameters),
        "flux_measurements": (pick_fluxes, flux),
    }
    terms = []
    for argument, group in measurements.items():
        for m in group:
            if m.experiment != experiment:
                continue
            if argument == "mid_measurements":
                predicted = get_pool_mid(model, conc, mids, m.ids)
            else:
                pick, values = picks[argument]
                predicted = pick(model, values, m.ids)
            terms.append(
                jnp.where(found, m.log_likelihood(predicted), jnp.nan),
            )
    return terms


def solve_with_mids(model, mid_simulator, experiment, parameters, guess):
    steady, found, conc, flux = solve(model, parameters, guess)
    forward, reverse = get_one_way_fluxes(
        model,
        conc[model.balanced_species_ix],
        parameters,
    )
    return steady, (
        found,
        conc,
        flux,
        mid_simulator(experiment, forward, reverse),
    )


def get_held_out_likelihood(hypothesis, mid_simulator, experiments):
    model = MODELS[hypothesis]
    measurements = load_measurements(experiments)

    def likelihood(cells, guess):
        terms = []
        for experiment in experiments:
            parameters = get_experiment_parameters(cells, experiment)
            _, solved = solve_with_mids(
                model,
                mid_simulator,
                experiment,
                parameters,
                guess,
            )
            terms.extend(
                score_experiment(
                    model,
                    measurements,
                    experiment,
                    parameters,
                    solved,
                ),
            )
        return jnp.concatenate(terms)

    return likelihood


def derive(
    model,
    layout,
    measurements,
    mid_simulator,
    parameters,
    guess,
    held_out,
):
    by_experiment = gather_parameters_by_experiment(layout, parameters)
    derived = {}
    terms = []
    for i, experiment in enumerate(layout.experiments):
        experiment_parameters = jax.tree.map(
            lambda leaf, i=i: leaf[i],
            by_experiment,
        )
        steady, solved = solve_with_mids(
            model,
            mid_simulator,
            experiment,
            experiment_parameters,
            guess[i],
        )
        found, conc, flux, mids = solved
        derived[f"steady|{experiment}"] = steady
        derived[f"found|{experiment}"] = found
        derived[f"growth_rate|{experiment}"] = get_growth_rate(
            model,
            experiment_parameters,
            steady,
        )
        derived[f"conc|{experiment}"] = conc
        derived[f"flux|{experiment}"] = flux
        for pool in MEASURED_POOLS:
            pool_id = POOL_SEPARATOR.join(pool)
            derived[f"mid|{pool_id}|{experiment}"] = get_pool_mid(
                model,
                conc,
                mids,
                pool,
            )
        terms.extend(
            score_experiment(
                model,
                measurements,
                experiment,
                experiment_parameters,
                solved,
            ),
        )
    derived["log_likelihood"] = jnp.concatenate(terms)
    if held_out is not None:
        reference = layout.experiments.index(REFERENCE_EXPERIMENT)
        derived["held_out_log_likelihood"] = held_out(
            jax.tree.map(lambda leaf: leaf[reference], by_experiment),
            derived[f"steady|{REFERENCE_EXPERIMENT}"],
        )
    return derived


def derive_all(derive_one, positions):
    rows = [
        {key: np.asarray(value) for key, value in derive_one(x).items()}
        for x in positions
    ]
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def get_pointwise_predictive_density(log_likelihoods):
    finite = log_likelihoods[np.isfinite(log_likelihoods).all(axis=1)]
    pointwise = np.asarray(logsumexp(finite, axis=0)) - np.log(len(finite))
    return pointwise, len(finite)


def main(fit):
    hypothesis, experiments = FITS[fit]
    split, layout, _, _, z_zero = get_z_posterior(hypothesis, experiments)
    _, unflatten = ravel_pytree(z_zero)
    guess = jnp.asarray(np.load(get_results_dir(fit) / "mode.npz")["guess"])
    mid_simulator = get_mid_simulator()
    measurements = load_measurements(experiments)
    held_out_experiments = HELD_OUT.get(fit.removeprefix(f"{hypothesis}_"))
    held_out = (
        None
        if held_out_experiments is None
        else get_held_out_likelihood(
            hypothesis,
            mid_simulator,
            held_out_experiments,
        )
    )

    @jax.jit
    def derive_parameters(parameters):
        return derive(
            MODELS[hypothesis],
            layout,
            measurements,
            mid_simulator,
            combine_parameters(split, parameters),
            guess,
            held_out,
        )

    start = time.perf_counter()
    posterior = derive_all(
        lambda x: derive_parameters(unflatten(x)),
        load_draws(fit),
    )
    derived_file = get_derived_file(fit)
    np.savez(derived_file, **posterior)
    minutes = (time.perf_counter() - start) / 60
    print(f"Saved derived quantities to {derived_file} in {minutes:.1f} min")
    if held_out is not None:
        pointwise, n_finite = get_pointwise_predictive_density(
            posterior["held_out_log_likelihood"],
        )
        print(
            f"Log predictive density of {', '.join(held_out_experiments)}: "
            f"{pointwise.sum():.2f} "
            f"({n_finite} of {len(posterior['held_out_log_likelihood'])} "
            "draws finite)",
        )


if __name__ == "__main__":
    for fit in sys.argv[1:] or FITS:
        main(fit)
