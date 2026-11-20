"""Screen candidate next experiments for the glycolysis case study.

For each hypothesis, predicts the steady state at every candidate condition
from 200 posterior draws of one round's fit. The candidates are eight external
glucose levels and six enzyme interventions: PFKM knockdown, DCA (a faster
pyruvate drain) and PKM2 overexpression. For each candidate and each pair of
hypotheses, scores how far apart the pair's predictions are. The score sums,
over balanced log concentrations and measured fluxes, the squared difference of
the two posterior means divided by twice the measurement variance plus both
posterior variances. Runs after rounds one and two, saving to
results/design_<phase>.npz.
"""

import itertools
import sys

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from analyse import get_derived_file, load_draws
from common import (
    CONC_ERROR,
    FLUX_ERROR,
    HYPOTHESES,
    MEASURED_FLUXES,
    MODEL,
    MODELS,
    PHASES,
    REFERENCE_EXPERIMENT,
    RESULTS_DIR,
    get_z_posterior,
    scale_values,
    solve,
)
from enzax.experiments import gather_parameters_by_experiment
from enzax.parameter_split import combine_parameters

SCREENED_PHASES = ("high_glucose", "series")
N_DRAW = 200
N_SUBSTEP = 9
GLUCOSE_MM = (30, 10, 4.5, 1.5, 0.45, 0.15, 0.045, 0.015)
REFERENCE_GLUCOSE_MM = 30
INTERVENTIONS = {
    "PFKM x0.3": {"PFKM": 0.3},
    "PFKM x0.1": {"PFKM": 0.1},
    "DCA x3": {"pyr_drain": 3.0},
    "DCA x10": {"pyr_drain": 10.0},
    "PKM2 x3": {"PKM2": 3.0},
    "PKM2 x10": {"PKM2": 10.0},
}
CANDIDATES = {
    **{
        f"Glucose {mm:g} mM": ({"glc_e": mm / REFERENCE_GLUCOSE_MM}, {})
        for mm in GLUCOSE_MM
    },
    **{name: ({}, factors) for name, factors in INTERVENTIONS.items()},
}
OBSERVABLES = (
    *(MODEL.species[i] for i in MODEL.balanced_species_ix),
    *MEASURED_FLUXES,
)
N_CONC = len(MODEL.balanced_species_ix)
FLUX_POSITIONS = np.array(
    [MODEL.reaction_ids.index(r) for r in MEASURED_FLUXES],
)


def get_design_file(phase):
    return RESULTS_DIR / f"design_{phase}.npz"


def get_target(cells, conc_factors, enzyme_factors):
    return scale_values(
        scale_values(cells, "log_conc_unbalanced", conc_factors),
        "log_enzyme",
        enzyme_factors,
    )


def to_observables(conc, flux):
    return np.concatenate(
        [
            np.log(np.asarray(conc)[np.asarray(MODEL.balanced_species_ix)]),
            np.asarray(flux)[FLUX_POSITIONS],
        ],
    )


def predict(solve_jit, cells, guess):
    predictions = np.full((len(CANDIDATES), len(OBSERVABLES)), np.nan)
    origins = {}
    for position, (conc_factors, enzyme_factors) in enumerate(
        CANDIDATES.values(),
    ):
        target = get_target(cells, conc_factors, enzyme_factors)
        origin, step_guess = origins.get(bool(conc_factors), (cells, guess))
        for step in np.linspace(0, 1, N_SUBSTEP)[1:]:
            step_guess, found, conc, flux = solve_jit(
                jax.tree.map(lambda a, b: a + step * (b - a), origin, target),
                step_guess,
            )
            if not found:
                break
        else:
            predictions[position] = to_observables(conc, flux)
            if conc_factors:
                origins[True] = (target, step_guess)
    return predictions


def predict_hypothesis(hypothesis, phase):
    fit = f"{hypothesis}_{phase}"
    split, layout, _, _, z_zero = get_z_posterior(hypothesis, PHASES[phase])
    _, unflatten = ravel_pytree(z_zero)
    draws = load_draws(fit)
    keep = np.linspace(0, len(draws) - 1, N_DRAW).astype(int)
    steady = np.load(get_derived_file(fit))[f"steady|{REFERENCE_EXPERIMENT}"]
    reference = layout.experiments.index(REFERENCE_EXPERIMENT)
    model = MODELS[hypothesis]
    solve_jit = jax.jit(
        lambda parameters, guess: solve(model, parameters, guess),
    )
    predictions = []
    for position in keep:
        parameters = combine_parameters(
            split,
            unflatten(jnp.asarray(draws[position])),
        )
        cells = jax.tree.map(
            lambda leaf: leaf[reference],
            gather_parameters_by_experiment(layout, parameters),
        )
        predictions.append(
            predict(solve_jit, cells, jnp.asarray(steady[position])),
        )
    return np.array(predictions)


def get_discrimination(predictions):
    pairs = list(itertools.combinations(HYPOTHESES, 2))
    discrimination = np.full((len(CANDIDATES), len(pairs)), np.nan)
    for (i, _), (j, (a, b)) in itertools.product(
        enumerate(CANDIDATES),
        enumerate(pairs),
    ):
        draws_a, draws_b = (
            predictions[h][:, i][np.isfinite(predictions[h][:, i]).all(axis=1)]
            for h in (a, b)
        )
        mean_a, mean_b = draws_a.mean(0), draws_b.mean(0)
        noise = (
            np.concatenate(
                [
                    np.full(N_CONC, CONC_ERROR),
                    FLUX_ERROR * np.abs(mean_a[N_CONC:] + mean_b[N_CONC:]) / 2,
                ],
            )
            ** 2
        )
        discrimination[i, j] = np.sum(
            (mean_a - mean_b) ** 2
            / (2 * noise + draws_a.var(0) + draws_b.var(0)),
        )
    return pairs, discrimination


def main(phase):
    predictions = {h: predict_hypothesis(h, phase) for h in HYPOTHESES}
    pairs, discrimination = get_discrimination(predictions)
    pair_names = [f"{a}-{b}" for a, b in pairs]
    print(f"{'condition':>18s} " + " ".join(f"{p:>9s}" for p in pair_names))
    for name, row in zip(CANDIDATES, discrimination, strict=True):
        print(f"{name:>18s} " + " ".join(f"{v:9.1f}" for v in row))
    for hypothesis in HYPOTHESES:
        failed = (~np.isfinite(predictions[hypothesis]).all(axis=2)).sum(0)
        print(f"{hypothesis}: failed draws per condition {failed.tolist()}")
    design_file = get_design_file(phase)
    np.savez(
        design_file,
        candidates=np.array(list(CANDIDATES)),
        observables=np.array(OBSERVABLES),
        pairs=np.array(pair_names),
        discrimination=discrimination,
        **{f"predictions|{h}": p for h, p in predictions.items()},
    )
    print(f"Saved the screen to {design_file}")


if __name__ == "__main__":
    for phase in sys.argv[1:] or SCREENED_PHASES:
        main(phase)
