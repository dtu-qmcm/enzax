"""Compute derived quantities from the prior and posterior draws.

For every posterior draw, 1000 prior draws and the true parameters, finds
each experiment's steady state and records its concentrations, fluxes, pool
MIDs, PFKM's tense-state fraction and the largest real part of its
Jacobian's eigenvalues. The tense-state fraction is defined as one minus the
ratio of PFKM's flux to its flux with the tense state removed.

Also solves two counterfactual ZeLa states, one for each of PFKM's
tense-state inhibitors, PEP and lactate. Each removes the inhibitor's site
from the tense state by giving it a very large dissociation constant, and
multiplies PFKM's transfer constant by the fourth power of the factor
1 + c/K that the site contributed in the wild type, where c is the
inhibitor's wild-type concentration and K its dissociation constant. The
inhibitor then acts on PFKM in ZeLa exactly as it did in the wild type,
while every other effect of the knockdown remains. Records each
counterfactual's glucose uptake relative to the wild type. Writes
everything to results/derived.npz.
"""

import time

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from common import (
    EXPERIMENTS,
    MEASURED_POOLS,
    MODEL,
    POOL_SEPARATOR,
    RESULTS_DIR,
    get_mid_simulator,
    get_z_posterior,
    load_true_parameters,
)
from enzax.experiments import gather_parameters_by_experiment
from enzax.measurements import get_pool_mid
from enzax.parameter_split import combine_parameters
from enzax.statistical_modelling import get_measurables_single_experiment
from enzax.thermodynamics import get_one_way_fluxes
from fit import N_CHAIN, get_chain_file

N_PRIOR_DRAW = 1000
PRIOR_SEED = 4321
TENSE_REACTION = "PFKM"
INHIBITORS = {"pep": "pep_c", "lac": "lac_c"}
FROZEN_LOG_DC = 50.0
DERIVED_FILE = RESULTS_DIR / "derived.npz"


def get_tense_fraction(parameters, steady):
    position = MODEL.reaction_ids.index(TENSE_REACTION)
    tc = MODEL.parameter_labelling["log_tc"].index(TENSE_REACTION)
    relaxed = parameters | {
        "log_tc": parameters["log_tc"].at[tc].set(-jnp.inf),
    }
    flux = MODEL.flux(steady, parameters)[position]
    relaxed_flux = MODEL.flux(steady, relaxed)[position]
    return 1.0 - flux / relaxed_flux


def freeze_site(parameters, species_id, conc):
    tc = MODEL.parameter_labelling["log_tc"].index(TENSE_REACTION)
    dc = MODEL.parameter_labelling["log_saturation_constant"].index(
        f"dc|{TENSE_REACTION}|{species_id}",
    )
    log_dc = parameters["log_saturation_constant"][dc]
    occupancy = conc[MODEL.species.index(species_id)] / jnp.exp(log_dc)
    return parameters | {
        "log_saturation_constant": parameters["log_saturation_constant"]
        .at[dc]
        .set(FROZEN_LOG_DC),
        "log_tc": parameters["log_tc"].at[tc].add(4 * jnp.log1p(occupancy)),
    }


def get_growth_rate(parameters, steady):
    jacobian = jax.jacfwd(MODEL.dndt)(steady, parameters)
    return jnp.linalg.eigvals(jacobian).real.max()


def derive(layout, mid_simulator, parameters, guess):
    by_experiment = gather_parameters_by_experiment(layout, parameters)
    uptake = MODEL.reaction_ids.index("GLUT4")
    derived = {}
    solutions = {}
    for i, experiment in enumerate(EXPERIMENTS):
        experiment_parameters = jax.tree.map(
            lambda leaf, i=i: leaf[i],
            by_experiment,
        )
        steady, found, conc, flux = get_measurables_single_experiment(
            MODEL,
            experiment_parameters,
            guess[i],
        )
        solutions[experiment] = experiment_parameters, steady, conc, flux
        derived[f"found|{experiment}"] = found
        derived[f"growth_rate|{experiment}"] = get_growth_rate(
            experiment_parameters,
            steady,
        )
        derived[f"tense_fraction|{experiment}"] = get_tense_fraction(
            experiment_parameters,
            steady,
        )
        derived[f"conc|{experiment}"] = conc
        derived[f"flux|{experiment}"] = flux
        forward, reverse = get_one_way_fluxes(
            MODEL,
            conc[MODEL.balanced_species_ix],
            experiment_parameters,
        )
        mids = mid_simulator(experiment, forward, reverse)
        for pool in MEASURED_POOLS:
            pool_id = POOL_SEPARATOR.join(pool)
            derived[f"mid|{pool_id}|{experiment}"] = get_pool_mid(
                MODEL,
                conc,
                mids,
                pool,
            )
    _, _, wild_type_conc, wild_type_flux = solutions["wild_type"]
    zela_parameters, zela_steady, _, zela_flux = solutions["zela"]
    derived["uptake_ratio"] = zela_flux[uptake] / wild_type_flux[uptake]
    for name, species_id in INHIBITORS.items():
        _, found, _, flux = get_measurables_single_experiment(
            MODEL,
            freeze_site(zela_parameters, species_id, wild_type_conc),
            zela_steady,
        )
        derived[f"found|{name}_frozen"] = found
        derived[f"uptake_ratio|{name}_frozen"] = (
            flux[uptake] / wild_type_flux[uptake]
        )
    return derived


def derive_all(derive_one, positions):
    rows = [
        {key: np.asarray(value) for key, value in derive_one(x).items()}
        for x in positions
    ]
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def main():
    split, layout, from_z_scores, _, z_zero = get_z_posterior()
    _, unflatten = ravel_pytree(z_zero)
    guess = jnp.asarray(np.load(RESULTS_DIR / "mode.npz")["guess"])
    mid_simulator = get_mid_simulator()

    @jax.jit
    def derive_parameters(parameters):
        return derive(
            layout,
            mid_simulator,
            combine_parameters(split, parameters),
            guess,
        )

    start = time.perf_counter()
    truth = derive(layout, mid_simulator, load_true_parameters(), guess)
    posterior = derive_all(
        lambda x: derive_parameters(unflatten(x)),
        np.concatenate(
            [
                np.load(get_chain_file(chain))["draws"]
                for chain in range(N_CHAIN)
            ],
        ),
    )
    z_prior = jax.random.normal(
        jax.random.key(PRIOR_SEED),
        (N_PRIOR_DRAW, *ravel_pytree(z_zero)[0].shape),
    )
    prior = derive_all(
        lambda z: derive_parameters(from_z_scores(unflatten(z))),
        z_prior,
    )
    np.savez(
        DERIVED_FILE,
        **{f"truth|{k}": np.asarray(v) for k, v in truth.items()},
        **{f"posterior|{k}": v for k, v in posterior.items()},
        **{f"prior|{k}": v for k, v in prior.items()},
    )
    minutes = (time.perf_counter() - start) / 60
    print(f"Saved derived quantities to {DERIVED_FILE} in {minutes:.1f} min")


if __name__ == "__main__":
    main()
