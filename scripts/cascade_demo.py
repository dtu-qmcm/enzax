"""Fit the GDP-fucose cascade's kinetic parameters to time-course data.

The five fitting experiments share their kinetic parameters and differ in
their enzyme loadings and initial amounts. Inferred by maximum a posteriori
optimisation are the turnover numbers except pyrophosphatase's, the
inhibition constant of FKP_fk by GDP-fucose, the Michaelis constants of the
nuisance reaction UNKNOWN_HYDROLASE, the formation energies, with
eQuilibrator's covariance as their prior, and the amount of polyphosphate,
which is the same in every experiment. Everything else stays at the values in
`enzax.examples.gdp_fucose`. The fitted model then
predicts two validation experiments that it was not fitted to.

Plots of one experiment's time courses at the prior mean and at the MAP are
saved to docs/img, for the time courses docs page. Making them needs
matplotlib, from the mcmc dependency group:

    uv run --group mcmc python scripts/cascade_demo.py
"""

# ruff: noqa: E402

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import time
from pathlib import Path

import jax
import numpy as np
import optimistix as optx
from jax import numpy as jnp

from enzax.examples import gdp_fucose
from enzax.experiments import (
    gather_parameters_by_experiment,
    get_experiment_layout,
    merge_experiment_parameters,
    pick_experiment,
)
from enzax.parameter_split import (
    combine_parameters,
    get_free_labels,
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.statistical_modelling import (
    log_density_time_course,
    prior_from_truth,
)
from enzax.time_course import get_measurables_time_course_single_experiment

jax.config.update("jax_enable_x64", True)

IMG_DIR = Path(__file__).parent.parent / "docs" / "img"
PRIOR_SD = 1.0
FREE_PARAMETERS = {
    "log_kcat": [reaction for reaction in gdp_fucose.KCAT if reaction != "PPA"],
    "log_amount_initial": ["polyp"],
    "log_saturation_constant": [
        "ki|FKP_fk|gdp_fucose",
        "km|UNKNOWN_HYDROLASE|atp",
        "km|UNKNOWN_HYDROLASE|amp",
        "km|UNKNOWN_HYDROLASE|adp",
    ],
    "dgf": None,
}


def get_problem(experiments):
    by_experiment = gather_parameters_by_experiment(
        gdp_fucose.layout,
        gdp_fucose.parameters,
    )
    layout = get_experiment_layout(
        gdp_fucose.labelling,
        experiments,
        gdp_fucose.EXPERIMENT_SPECIFIC,
    )
    expanded = merge_experiment_parameters(
        layout,
        [
            pick_experiment(gdp_fucose.layout, by_experiment, experiment)
            for experiment in experiments
        ],
    )
    split = split_parameters_by_freeing(
        layout.labelling,
        expanded,
        FREE_PARAMETERS,
    )
    times, measurements = gdp_fucose.load_measurements(experiments)
    nominal = get_free_parameters(split, expanded)
    return layout, split, times, measurements, nominal


def report(title, layout, split, times, measurements, free):
    """Print the predicted and measured GDP-fucose at each experiment's end."""
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
        print(
            f"  {experiment:17} GDP-fucose at {last:>7}: predicted "
            f"{float(predicted):6.2f} mM, measured {observed:6.2f} mM",
        )


# --8<-- [start:plots]
import matplotlib.pyplot as plt

PLOT_EXPERIMENT = "FE13"
PLOT_SPECIES = ["gmp", "atp", "gtp", "gdp_fucose"]
COLOUR = "#2a78d6"
GRID = "#e4e3df"
INK = "#52514e"


def get_time_course(layout, split, free, experiment, ts):
    """Get an experiment's concentrations at ts."""
    by_experiment = gather_parameters_by_experiment(
        layout,
        combine_parameters(split, free),
    )
    parameters = pick_experiment(layout, by_experiment, experiment)
    _, conc, _, _ = get_measurables_time_course_single_experiment(
        gdp_fucose.model,
        parameters,
        jnp.exp(parameters["log_amount_initial"]),
        ts,
    )
    return conc


def get_measured(times, measurements, experiment, species):
    """Get an experiment's measured times and values for one species."""
    measurement = next(m for m in measurements if m.experiment == experiment)
    rows = [
        (times[experiment][id_.split("|")[1]], float(value))
        for id_, value in zip(measurement.ids, measurement.values)
        if id_.split("|")[0] == species
    ]
    return zip(*rows)


def plot_time_course(conc, ts, times, measurements, title, y_max):
    """Plot PLOT_EXPERIMENT's time course against its measurements."""
    fig, axes = plt.subplots(
        1,
        len(PLOT_SPECIES),
        figsize=(11, 3),
        sharex=True,
    )
    for ax, species in zip(axes, PLOT_SPECIES):
        ax.plot(
            ts,
            conc[:, gdp_fucose.model.species.index(species)],
            color=COLOUR,
            linewidth=2,
            solid_capstyle="round",
            label="model",
        )
        measured_ts, measured = get_measured(
            times,
            measurements,
            PLOT_EXPERIMENT,
            species,
        )
        ax.plot(
            measured_ts,
            measured,
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=COLOUR,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="measured",
        )
        ax.set_title(species, color=INK, fontsize=11)
        ax.set_ylim(0, y_max[species])
        ax.set_xlabel("time (h)", color=INK)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("concentration (mM)", color=INK)
    axes[0].legend(frameon=False, labelcolor=INK)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    return fig


def plot_fit(layout, split, times, measurements, free_by_label):
    """Plot PLOT_EXPERIMENT at each set of free parameters, on shared axes.

    free_by_label maps a title, such as "MAP", to a set of free parameters.
    """
    ts = jnp.linspace(0.0, max(times[PLOT_EXPERIMENT].values()), 200)
    conc_by_label = {
        label: get_time_course(layout, split, free, PLOT_EXPERIMENT, ts)
        for label, free in free_by_label.items()
    }
    y_max = {}
    for species in PLOT_SPECIES:
        i = gdp_fucose.model.species.index(species)
        _, measured = get_measured(
            times,
            measurements,
            PLOT_EXPERIMENT,
            species,
        )
        y_max[species] = 1.1 * max(
            *(float(conc[:, i].max()) for conc in conc_by_label.values()),
            *measured,
        )
    return {
        label: plot_time_course(
            conc,
            ts,
            times,
            measurements,
            f"{PLOT_EXPERIMENT} at the {label}",
            y_max,
        )
        for label, conc in conc_by_label.items()
    }


# --8<-- [end:plots]


def main():
    layout, split, times, measurements, initial = get_problem(
        gdp_fucose.FITTING_EXPERIMENTS,
    )
    prior = prior_from_truth(initial, sd=PRIOR_SD) | {
        "dgf": (initial["dgf"], jnp.asarray(gdp_fucose.get_dgf_covariance())),
    }

    def negative_log_density(free, args):
        return -log_density_time_course(
            free,
            gdp_fucose.model,
            prior,
            times,
            layout=layout,
            split=split,
            conc_measurements=measurements,
        )

    start = time.perf_counter()
    solution = optx.minimise(
        negative_log_density,
        optx.BFGS(rtol=1e-6, atol=1e-6),
        initial,
        max_steps=1000,
        throw=False,
    )
    fitted = solution.value
    print(
        f"BFGS finished with result {solution.result} after "
        f"{int(solution.stats['num_steps'])} steps and "
        f"{time.perf_counter() - start:.0f} s.",
    )
    print(
        "Log density: "
        f"{-float(negative_log_density(initial, None)):.1f} at the prior "
        f"mean, {-float(negative_log_density(fitted, None)):.1f} at the MAP.",
    )
    print("Free parameters, exponentiated if on the log scale:")
    for parameter in initial:
        transform = (lambda x: x) if parameter == "dgf" else np.exp
        for label, before, after in zip(
            get_free_labels(split, parameter),
            transform(initial[parameter]),
            transform(fitted[parameter]),
        ):
            print(
                f"  {parameter} {label:18} prior mean {before:10.4g}, "
                f"MAP {after:10.4g}",
            )
    report("Fitting experiments:", layout, split, times, measurements, fitted)
    figures = plot_fit(
        layout,
        split,
        times,
        measurements,
        {"prior mean": initial, "MAP": fitted},
    )
    figures["prior mean"].savefig(
        IMG_DIR / "time_course_prior_mean.png",
        dpi=150,
    )
    figures["MAP"].savefig(IMG_DIR / "time_course_map.png", dpi=150)
    (
        validation_layout,
        validation_split,
        validation_times,
        validation_measurements,
        validation_nominal,
    ) = get_problem(gdp_fucose.VALIDATION_EXPERIMENTS)
    shared = {
        parameter: values
        for parameter, values in fitted.items()
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


if __name__ == "__main__":
    main()
