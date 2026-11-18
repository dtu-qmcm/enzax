"""Simulate measurements for the glycolysis labelling case study.

Draws true parameters from the prior for the wild type and ZeLa, then gives
every experiment the wild type's offsets in clamped concentrations, so that
the experiments differ only by design. The high-lactate experiment copies the
wild type's parameters, so adding it leaves the other experiments' truths
unchanged. Finds each experiment's steady state and adds noise to its
concentrations, enzyme levels, boundary fluxes and MIDs. Writes the
measurements to data/measurements.csv and the true values to data/truth.json,
then checks the log density at the truth.
"""

import csv
import json

import jax
from jax import numpy as jnp

from common import (
    CONC_ERROR,
    CSV_COLUMNS,
    DESIGN,
    ENZYME_ERROR,
    EXPERIMENT_SPECIFIC,
    EXPERIMENTS,
    FLUX_ERROR,
    KNOCKDOWNS,
    MEASURED_ENZYMES,
    MEASURED_FLUXES,
    MEASURED_POOLS,
    MEASUREMENTS_FILE,
    MID_ERROR,
    MODEL,
    N_REPLICATE,
    POOL_SEPARATOR,
    TRUTH_FILE,
    get_condition,
    get_expanded_reference,
    get_initial_guess,
    get_labelled_values,
    get_layout,
    get_mid_simulator,
    get_prior,
    get_prior_scale,
    get_split,
    load_measurements,
    load_true_parameters,
)
from enzax.experiments import (
    gather_parameters_by_experiment,
    get_experiment_layout,
    merge_experiment_parameters,
)
from enzax.measurements import get_pool_mid, pick_enzymes
from enzax.parameter_split import combine_parameters, get_free_parameters
from enzax.statistical_modelling import (
    get_measurables_single_experiment,
    log_density,
)
from enzax.thermodynamics import get_one_way_fluxes

TRUTH_SEED = 1234
DATA_SEED = 1235
DRAWN_EXPERIMENTS = ("wild_type", "zela")
SAME_CELLS = {"high_lactate": "wild_type"}


def draw_true_parameters(key, layout, split):
    free_reference = get_free_parameters(split, get_expanded_reference(layout))
    scale = get_prior_scale(free_reference)
    keys = dict(
        zip(free_reference, jax.random.split(key, len(free_reference))),
    )
    free_true = {
        kind: leaf
        + scale[kind] * jax.random.normal(keys[kind], jnp.shape(leaf))
        for kind, leaf in free_reference.items()
    }
    return share_clamp_offsets(layout, combine_parameters(split, free_true))


def share_clamp_offsets(layout, parameters):
    kind = "log_conc_unbalanced"
    reference, drawn = (
        gather_parameters_by_experiment(layout, p)
        for p in (get_expanded_reference(layout), parameters)
    )
    shared = drawn | {
        kind: reference[kind] + drawn[kind][0] - reference[kind][0],
    }
    return merge_experiment_parameters(
        layout,
        [
            jax.tree.map(lambda leaf, i=i: leaf[i], shared)
            for i in range(len(layout.experiments))
        ],
    )


def extend_to_experiments(drawn_layout, drawn, layout):
    by_experiment = gather_parameters_by_experiment(drawn_layout, drawn)
    parameters = {
        experiment: jax.tree.map(lambda leaf, i=i: leaf[i], by_experiment)
        for i, experiment in enumerate(drawn_layout.experiments)
    }
    for experiment in layout.experiments:
        if experiment not in parameters:
            parameters[experiment] = get_condition(
                parameters[SAME_CELLS[experiment]],
                DESIGN[experiment],
            )
    return merge_experiment_parameters(
        layout,
        [parameters[experiment] for experiment in layout.experiments],
    )


def get_rows(
    experiment,
    kind,
    ids,
    values,
    errors,
    replicate=0,
    isotopologues=None,
):
    if isotopologues is None:
        isotopologues = [""] * len(ids)
    return [
        {
            "experiment": experiment,
            "kind": kind,
            "id": i,
            "replicate": replicate,
            "isotopologue": isotopologue,
            "value": float(value),
            "error": float(error),
        }
        for i, value, error, isotopologue in zip(
            ids,
            values,
            errors,
            isotopologues,
            strict=True,
        )
    ]


def simulate_experiment(key, experiment, parameters, guess, mid_simulator):
    key_conc, key_enzyme, key_flux, key_mid = jax.random.split(key, 4)
    _, found, true_conc, true_flux = get_measurables_single_experiment(
        MODEL,
        parameters,
        guess,
    )
    if not found:
        msg = f"No steady state at the true parameters in {experiment}."
        raise RuntimeError(msg)
    forward, reverse = get_one_way_fluxes(
        MODEL,
        true_conc[MODEL.balanced_species_ix],
        parameters,
    )
    true_mids = mid_simulator(experiment, forward, reverse)
    pool_mids = {
        POOL_SEPARATOR.join(pool): get_pool_mid(
            MODEL,
            true_conc,
            true_mids,
            pool,
        )
        for pool in MEASURED_POOLS
    }

    rows = get_rows(
        experiment,
        "conc",
        MODEL.species,
        true_conc
        * jnp.exp(CONC_ERROR * jax.random.normal(key_conc, true_conc.shape)),
        jnp.full_like(true_conc, CONC_ERROR),
    )
    enzymes = [
        e for e in MEASURED_ENZYMES if e not in KNOCKDOWNS.get(experiment, {})
    ]
    true_enzyme = pick_enzymes(MODEL, parameters, enzymes)
    rows += get_rows(
        experiment,
        "enzyme",
        enzymes,
        true_enzyme
        * jnp.exp(
            ENZYME_ERROR * jax.random.normal(key_enzyme, true_enzyme.shape),
        ),
        jnp.full_like(true_enzyme, ENZYME_ERROR),
    )
    measured_flux = jnp.array(
        [true_flux[MODEL.reaction_ids.index(r)] for r in MEASURED_FLUXES],
    )
    flux_error = FLUX_ERROR * jnp.abs(measured_flux)
    rows += get_rows(
        experiment,
        "flux",
        MEASURED_FLUXES,
        measured_flux
        + flux_error * jax.random.normal(key_flux, measured_flux.shape),
        flux_error,
    )
    for i, (pool, mid) in enumerate(pool_mids.items()):
        noise = jax.random.normal(
            jax.random.fold_in(key_mid, i),
            (N_REPLICATE, *mid.shape),
        )
        intensity = mid * jnp.exp(MID_ERROR * noise)
        measured = intensity / intensity.sum(axis=-1, keepdims=True)
        for replicate, values in enumerate(measured):
            rows += get_rows(
                experiment,
                "mid",
                [pool] * len(values),
                values,
                [MID_ERROR] * len(values),
                replicate=replicate,
                isotopologues=range(len(values)),
            )
    truth = {
        "conc": dict(zip(MODEL.species, true_conc.tolist(), strict=True)),
        "flux": dict(zip(MODEL.reaction_ids, true_flux.tolist(), strict=True)),
        "mids": {pool: mid.tolist() for pool, mid in pool_mids.items()},
    }
    return rows, truth


def simulate(truth_seed, data_seed):
    layout = get_layout(EXPERIMENTS)
    key_data = jax.random.key(data_seed)
    drawn_layout = get_experiment_layout(
        MODEL.parameter_labelling,
        DRAWN_EXPERIMENTS,
        EXPERIMENT_SPECIFIC,
    )
    true_parameters = extend_to_experiments(
        drawn_layout,
        draw_true_parameters(
            jax.random.key(truth_seed),
            drawn_layout,
            get_split(drawn_layout),
        ),
        layout,
    )
    by_experiment = gather_parameters_by_experiment(layout, true_parameters)
    guess = get_initial_guess(EXPERIMENTS)
    mid_simulator = get_mid_simulator()
    rows = []
    truth = {
        "truth_seed": truth_seed,
        "data_seed": data_seed,
        "parameters": get_labelled_values(layout.labelling, true_parameters),
    }
    for i, experiment in enumerate(layout.experiments):
        experiment_rows, truth[experiment] = simulate_experiment(
            jax.random.fold_in(key_data, i),
            experiment,
            jax.tree.map(lambda leaf, i=i: leaf[i], by_experiment),
            guess[i],
            mid_simulator,
        )
        rows += experiment_rows
    return rows, truth


def write(rows, truth):
    MEASUREMENTS_FILE.parent.mkdir(exist_ok=True)
    with MEASUREMENTS_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    TRUTH_FILE.write_text(json.dumps(truth, indent=2) + "\n")


def check():
    layout = get_layout(EXPERIMENTS)
    split = get_split(layout)
    prior = get_prior(layout, split)
    measurements = load_measurements(EXPERIMENTS)
    free_true = get_free_parameters(split, load_true_parameters(layout))
    mid_simulator = get_mid_simulator()
    guess = get_initial_guess(EXPERIMENTS)

    def density(free_parameters, with_mids=True):
        return log_density(
            free_parameters,
            MODEL,
            prior,
            layout=layout,
            split=split,
            guess=guess,
            conc_measurements=measurements["conc_measurements"],
            enzyme_measurements=measurements["enzyme_measurements"],
            flux_measurements=measurements["flux_measurements"],
            mid_measurements=(
                measurements["mid_measurements"] if with_mids else []
            ),
            mid_simulator=mid_simulator,
        )

    with_mids = float(density(free_true))
    without_mids = float(density(free_true, with_mids=False))
    print(f"Log density at the truth: {with_mids:.2f}")
    print(f"  of which MIDs: {with_mids - without_mids:.2f}")
    gradient = jax.grad(density)(free_true)
    finite = all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(gradient))
    print(f"Gradient finite: {finite}")


def main():
    rows, truth = simulate(TRUTH_SEED, DATA_SEED)
    write(rows, truth)
    print(f"Wrote {len(rows)} measurements to {MEASUREMENTS_FILE}")
    check()


if __name__ == "__main__":
    main()
