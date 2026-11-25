"""Find the least enzyme that reliably meets Huber et al.'s O3 target.

Huber et al.'s O3 problem asks for the least enzyme mass that gives at least
15 mM GDP-fucose at 24 h, starting from FE18_07FKP's substrate amounts. Here
the target has to be met with 90% posterior probability, so a design's
guaranteed titre is the 10% quantile of its titre over 100 posterior draws.
For a given enzyme mass, projected gradient ascent maximises the guaranteed
titre over FKP, PPA and PPK3 within Huber et al.'s bounds, and bisection on
the mass finds the smallest one whose best design meets the target. The
search starts from Huber et al.'s O1 design, which meets the target. Checks
the baseline design, Huber et al.'s O3 design and the optimum on 100 other
draws, and saves them to results/designs.npz.
"""

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import sys  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from common import RESULTS_DIR, SOLVE, get_posterior, get_problem  # noqa: E402
from enzax.examples import gdp_fucose  # noqa: E402
from enzax.experiments import (  # noqa: E402
    gather_parameters_by_experiment,
    pick_experiment,
)
from enzax.parameter_split import (  # noqa: E402
    combine_parameters,
    get_free_labels,
)
from predict import get_draws  # noqa: E402

DESIGN_EXPERIMENT = "FE18_07FKP"
FEASIBLE_EXPERIMENT = "FE20_Validation1"
REFERENCE_EXPERIMENT = "FE20_Validation2"
END_TIME = 24.0
LOWER = np.array([0.00021, 0.00319, 0.00018])
UPPER = np.array([0.0149, 0.224, 0.0129])
MOLECULAR_WEIGHT = np.array([105.660, 19.313, 34.740])
TARGET = 15.0
QUANTILE = 10
N_OPTIMISE = 100
N_STEP = 60
INITIAL_STEP = 0.002
MIN_STEP = 1e-6
REPORT_EVERY = 5
N_BISECTION = 60
N_BUDGET_BISECTION = 8
DESIGNS_FILE = RESULTS_DIR / "designs.npz"


def get_mass(x):
    return x @ MOLECULAR_WEIGHT


def project(x, budget):
    clipped = jnp.clip(x, LOWER, UPPER)
    low, high = 0.0, float(jnp.max((x - LOWER) / MOLECULAR_WEIGHT))
    for _ in range(N_BISECTION):
        middle = (low + high) / 2
        shifted = jnp.clip(x - middle * MOLECULAR_WEIGHT, LOWER, UPPER)
        if get_mass(shifted) > budget:
            low = middle
        else:
            high = middle
    shifted = jnp.clip(x - high * MOLECULAR_WEIGHT, LOWER, UPPER)
    return jnp.where(get_mass(clipped) <= budget, clipped, shifted)


def get_design_parameters(draws, fitted_split):
    layout, split, _, _, nominal = get_problem([DESIGN_EXPERIMENT])
    shared = [
        parameter
        for parameter in nominal
        if get_free_labels(split, parameter)
        == get_free_labels(fitted_split, parameter)
    ]
    stacked = []
    for values in draws:
        free = nominal | {p: values["model"][p] for p in shared}
        by_experiment = gather_parameters_by_experiment(
            layout,
            combine_parameters(split, free),
        )
        stacked.append(
            pick_experiment(layout, by_experiment, DESIGN_EXPERIMENT),
        )
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *stacked)


def get_titre_fn(parameters):
    product = gdp_fucose.model.species.index("gdp_fucose")
    ts = jnp.array([0.0, END_TIME])

    def titre(enzymes, one):
        one = one | {"log_enzyme": jnp.log(enzymes)}
        _, conc, _, _ = SOLVE(
            gdp_fucose.model,
            one,
            jnp.exp(one["log_amount_initial"]),
            ts,
        )
        return conc[-1, product]

    return jax.jit(
        lambda enzymes: jax.vmap(titre, in_axes=(None, 0))(enzymes, parameters),
    )


def get_guaranteed_titre(titres):
    return jnp.nanpercentile(titres, QUANTILE)


def optimise(objective, start, budget):
    x = project(jnp.asarray(start), budget)
    value, gradient = objective(x)
    step = INITIAL_STEP
    for iteration in range(N_STEP):
        direction = gradient / jnp.linalg.norm(gradient)
        while step >= MIN_STEP:
            candidate = project(x + step * direction, budget)
            new_value, new_gradient = objective(candidate)
            if new_value > value:
                break
            step /= 2
        else:
            break
        moved = float(jnp.linalg.norm(candidate - x))
        x, value, gradient = candidate, new_value, new_gradient
        if iteration % REPORT_EVERY == 0:
            print(
                f"Step {iteration}: guaranteed titre {float(value):.3f} mM at "
                f"{np.round(np.asarray(x), 5).tolist()}",
            )
        if moved < MIN_STEP:
            break
        step *= 2
    return np.asarray(x), float(value)


def find_least_enzyme(objective, target, start):
    low, high = float(get_mass(LOWER)), float(get_mass(start))
    best = start
    for _ in range(N_BUDGET_BISECTION):
        budget = (low + high) / 2
        design, value = optimise(objective, best, budget)
        print(
            f"Budget {budget:.3f} g/l: guaranteed titre {value:.3f} mM, "
            f"target {target:.3f} mM",
        )
        if value >= target:
            high, best = budget, design
        else:
            low = budget
    return best


def main(run):
    _, split, _, _, _, start = get_posterior()
    _, unflatten = ravel_pytree(start)
    draws = get_draws(run, unflatten)
    optimise_draws, check_draws = draws[:N_OPTIMISE], draws[N_OPTIMISE:]
    titres = get_titre_fn(get_design_parameters(optimise_draws, split))
    objective = jax.jit(
        jax.value_and_grad(lambda x: get_guaranteed_titre(titres(x))),
    )
    baseline = np.array(gdp_fucose.ENZYME_LEVELS[DESIGN_EXPERIMENT][:3])
    huber = np.array(gdp_fucose.ENZYME_LEVELS[REFERENCE_EXPERIMENT][:3])
    feasible = np.array(gdp_fucose.ENZYME_LEVELS[FEASIBLE_EXPERIMENT][:3])
    best = find_least_enzyme(objective, TARGET, feasible)
    designs = {"baseline": baseline, "huber": huber, "optimum": best}
    check = get_titre_fn(get_design_parameters(check_draws, split))
    results = {"target": TARGET}
    for name, design in designs.items():
        results[f"{name}_design"] = design
        results[f"{name}_titre"] = np.asarray(check(jnp.asarray(design)))
        print(
            f"{name}: {np.round(design, 5).tolist()}, enzyme mass "
            f"{get_mass(design):.3f} g/l, guaranteed titre on held-out "
            f"draws {float(get_guaranteed_titre(results[f'{name}_titre'])):.2f}"
            f" mM, probability of reaching the target "
            f"{np.mean(results[f'{name}_titre'] >= TARGET):.2f}",
        )
    np.savez(DESIGNS_FILE, **results)
    print(f"Saved the designs to {DESIGNS_FILE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
