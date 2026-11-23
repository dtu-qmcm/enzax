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
from enzax.parameter_split import combine_parameters  # noqa: E402
from predict import get_draws  # noqa: E402

DESIGN_EXPERIMENT = "FE18_07FKP"
VALIDATION_EXPERIMENT = "FE20_Validation1"
END_TIME = 24.0
LOWER = np.array([0.00021, 0.00319, 0.00018])
UPPER = np.array([0.0149, 0.224, 0.0129])
GMPK_LOWER = 0.00241
TOTAL_ENZYME = 0.0469
N_OPTIMISE = 100
N_STEP = 60
INITIAL_STEP = 0.002
MIN_STEP = 1e-6
REPORT_EVERY = 5
N_BISECTION = 60
DESIGNS_FILE = RESULTS_DIR / "designs.npz"


def project(x):
    budget = TOTAL_ENZYME - GMPK_LOWER
    clipped = jnp.clip(x, LOWER, UPPER)
    low, high = 0.0, float(jnp.max(x - LOWER))
    for _ in range(N_BISECTION):
        middle = (low + high) / 2
        if jnp.clip(x - middle, LOWER, UPPER).sum() > budget:
            low = middle
        else:
            high = middle
    shifted = jnp.clip(x - high, LOWER, UPPER)
    return jnp.where(clipped.sum() <= budget, clipped, shifted)


def get_design_parameters(draws):
    layout, split, _, _, _ = get_problem(gdp_fucose.FITTING_EXPERIMENTS)
    stacked = []
    for values in draws:
        by_experiment = gather_parameters_by_experiment(
            layout,
            combine_parameters(split, values["model"]),
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


def optimise(titres, start):
    objective = jax.jit(
        jax.value_and_grad(lambda x: jnp.nanmean(titres(x))),
    )
    x = jnp.asarray(start)
    value, gradient = objective(x)
    step = INITIAL_STEP
    for iteration in range(N_STEP):
        direction = gradient / jnp.linalg.norm(gradient)
        while step >= MIN_STEP:
            candidate = project(x + step * direction)
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
                f"Step {iteration}: expected titre {float(value):.3f} mM at "
                f"{np.round(np.asarray(x), 5).tolist()}",
            )
        if moved < MIN_STEP:
            break
        step *= 2
    print(
        f"Stopped after {iteration + 1} steps: expected titre "
        f"{float(value):.3f} mM at {np.round(np.asarray(x), 5).tolist()}",
    )
    return np.asarray(x)


def main(run):
    _, _, _, _, _, start = get_posterior()
    _, unflatten = ravel_pytree(start)
    draws = get_draws(run, unflatten)
    optimise_draws, check_draws = draws[:N_OPTIMISE], draws[N_OPTIMISE:]
    titres = get_titre_fn(get_design_parameters(optimise_draws))
    baseline = np.array(gdp_fucose.ENZYME_LEVELS[DESIGN_EXPERIMENT][:3])
    huber = np.array(gdp_fucose.ENZYME_LEVELS[VALIDATION_EXPERIMENT][:3])
    best = max(
        (optimise(titres, project(jnp.asarray(s))) for s in (baseline, huber)),
        key=lambda design: float(jnp.nanmean(titres(jnp.asarray(design)))),
    )
    designs = {"baseline": baseline, "huber": huber, "optimum": best}
    check = get_titre_fn(get_design_parameters(check_draws))
    results = {}
    for name, design in designs.items():
        results[f"{name}_design"] = design
        results[f"{name}_titre"] = np.asarray(check(jnp.asarray(design)))
        print(
            f"{name}: {np.round(design, 5).tolist()}, expected titre on "
            f"held-out draws {np.nanmean(results[f'{name}_titre']):.2f} mM",
        )
    np.savez(DESIGNS_FILE, **results)
    print(f"Saved the designs to {DESIGNS_FILE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
