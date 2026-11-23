import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import time  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
import optimistix as optx  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from common import (  # noqa: E402
    DOCS_IMG_DIR,
    RESULTS_DIR,
    get_days,
    get_labels,
    get_posterior,
    get_problem,
    get_time_course,
)
from docs_figures import PLOT_EXPERIMENT, plot_fit  # noqa: E402
from enzax.examples import gdp_fucose  # noqa: E402
from enzax.parameter_split import get_free_labels  # noqa: E402

MODE_FILE = RESULTS_DIR / "mode.npz"
HESSIAN_STEP = 1e-4


def find_mode(log_posterior, start):
    solution = optx.minimise(
        lambda values, args: -log_posterior(values),
        optx.BFGS(rtol=1e-6, atol=1e-6),
        start,
        max_steps=1000,
        throw=False,
    )
    print(
        f"BFGS finished with result {solution.result} after "
        f"{int(solution.stats['num_steps'])} steps.",
    )
    return solution.value


def get_hessian(log_posterior, mode):
    flat, unflatten = ravel_pytree(mode)
    gradient = jax.jit(jax.grad(lambda x: log_posterior(unflatten(x))))
    columns = []
    for i in range(len(flat)):
        step = jnp.zeros_like(flat).at[i].set(HESSIAN_STEP)
        columns.append(
            (gradient(flat + step) - gradient(flat - step))
            / (2 * HESSIAN_STEP),
        )
    hessian = np.asarray(jnp.stack(columns, axis=1))
    return (hessian + hessian.T) / 2


def report(title, layout, split, times, measurements, free, bias=None):
    """Print the predicted and measured GDP-fucose at each experiment's end.

    bias maps experiments to the bias of their measurements, which multiplies
    the prediction by e^bias; experiments it leaves out get none.
    """
    bias = bias or {}
    print(title)
    for experiment, measurement in zip(layout.experiments, measurements):
        grid = times[experiment]
        last = list(grid)[-1]
        rows = dict(zip(measurement.ids, measurement.values.tolist()))
        observed = rows.get(f"gdp_fucose|{last}", float("nan"))
        conc = get_time_course(
            layout,
            split,
            free,
            experiment,
            jnp.array(list(grid.values())),
        )
        predicted = conc[-1, gdp_fucose.model.species.index("gdp_fucose")]
        predicted = predicted * np.exp(bias.get(experiment, 0.0))
        print(
            f"  {experiment:17} GDP-fucose at {last:>7}: predicted "
            f"{float(predicted):6.2f} mM, measured {observed:6.2f} mM",
        )


def main():
    layout, split, times, measurements, log_posterior, start = get_posterior()
    begin = time.perf_counter()
    mode = find_mode(log_posterior, start)
    print(f"Mode found in {time.perf_counter() - begin:.0f} s.")
    print(
        f"Log density: {float(log_posterior(start)):.1f} at the prior mean, "
        f"{float(log_posterior(mode)):.1f} at the mode.",
    )
    begin = time.perf_counter()
    hessian = get_hessian(log_posterior, mode)
    eigenvalues = np.linalg.eigvalsh(-hessian)
    print(
        f"Hessian in {time.perf_counter() - begin:.0f} s; eigenvalues of the "
        f"negative Hessian from {eigenvalues[0]:.3g} to {eigenvalues[-1]:.3g}.",
    )
    days = get_days(layout.experiments)
    print("Measurement bias by day, exponentiated:")
    for day, value in zip(days, np.exp(mode["bias"])):
        print(f"  {day} {value:.3f}")
    print("Measurement noise by species (standard deviation, log scale):")
    for species, value in zip(
        gdp_fucose.MEASURED_SPECIES,
        np.exp(mode["log_noise"]),
    ):
        print(f"  {species:10} {value:.3f}")
    print("Free parameters, exponentiated if on the log scale:")
    for parameter in start["model"]:
        transform = (lambda x: x) if parameter == "dgf" else np.exp
        for label, before, after in zip(
            get_free_labels(split, parameter),
            transform(start["model"][parameter]),
            transform(mode["model"][parameter]),
        ):
            print(
                f"  {parameter} {label:18} prior mean {before:10.4g}, "
                f"mode {after:10.4g}",
            )
    bias = {
        e: float(mode["bias"][days.index(gdp_fucose.MEASUREMENT_DAY[e])])
        for e in layout.experiments
    }
    report(
        "Fitting experiments:",
        layout,
        split,
        times,
        measurements,
        mode["model"],
        bias,
    )
    (
        validation_layout,
        validation_split,
        validation_times,
        validation_measurements,
        validation_nominal,
    ) = get_problem(gdp_fucose.VALIDATION_EXPERIMENTS)
    shared = {
        parameter: values
        for parameter, values in mode["model"].items()
        if get_free_labels(split, parameter)
        == get_free_labels(validation_split, parameter)
    }
    report(
        "Validation experiments:",
        validation_layout,
        validation_split,
        validation_times,
        validation_measurements,
        validation_nominal | shared,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODE_FILE,
        mode=np.asarray(ravel_pytree(mode)[0]),
        hessian=hessian,
        labels=np.array(get_labels(split, mode)),
    )
    print(f"Saved the mode to {MODE_FILE}")
    figures = plot_fit(
        layout,
        split,
        times,
        measurements,
        {"prior mean": start["model"], "MAP": mode["model"]},
        {"prior mean": 0.0, "MAP": bias[PLOT_EXPERIMENT]},
    )
    figures["prior mean"].savefig(
        DOCS_IMG_DIR / "time_course_prior_mean.png",
        dpi=150,
    )
    figures["MAP"].savefig(DOCS_IMG_DIR / "time_course_map.png", dpi=150)


if __name__ == "__main__":
    main()
