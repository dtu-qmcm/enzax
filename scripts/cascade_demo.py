"""Fit the GDP-fucose cascade's kinetic parameters to time-course data.

The five fitting experiments share their kinetic parameters and differ in
their enzyme loadings and initial amounts. Inferred by maximum a posteriori
optimisation are the turnover numbers except pyrophosphatase's, the inhibition
constant of FKP_fk by GDP-fucose, the rate constant and Michaelis constant for
AMP of the nuisance reaction UNKNOWN_HYDROLASE, PPK3's five Michaelis
constants, and four formation energies: ATP's, polyphosphate's, GDP's and
GMP's. Everything else stays at the values in `enzax.examples.gdp_fucose`.

The other formation energies are fixed at their prior means, since only
reactions' standard Gibbs energy changes matter. The free four have
eQuilibrator's covariance conditioned on the fixed ones, plus DGF_CONDITIONS_SD
on each: eQuilibrator's estimates are not for this pot's temperature, pH and
Mg2+, and the extra uncertainty stands in for that until priors can be fetched
for the actual conditions (enzax issue #76).

Two things belong to the measurements rather than the model. Each measurement
day has its own bias, which multiplies every concentration measured that day,
with prior standard deviation BIAS_SD on the log scale: the measured nucleotides
add up to more than the initial amounts, by a similar factor for every species
in a sample. And each measured species has its own noise, inferred with prior
LOG_NOISE_PRIOR, rather than a fixed 0.1.

The fitted model then predicts two validation experiments that it was not
fitted to; their day has no fitted bias, so their predictions have none. The
solves use Kvaerno4.

Plots of one experiment's time courses at the prior mean and at the MAP are
saved to docs/img, for the time courses docs page. Making them needs
matplotlib, from the mcmc dependency group:

    uv run --group plot python scripts/cascade_demo.py
"""

# ruff: noqa: E402

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import functools
import time
from pathlib import Path

import diffrax
import jax
import numpy as np
import optimistix as optx
from jax import numpy as jnp
from jax.scipy.stats import norm

from enzax.examples import gdp_fucose
from enzax.experiments import (
    gather_parameters_by_experiment,
    get_experiment_layout,
    merge_experiment_parameters,
    pick_experiment,
)
from enzax.measurements import LognormalMeasurement
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
WIDE_PRIOR_SD = {
    "log_saturation_constant": {"ki|FKP_fk|gdp_fucose": 3.0},
}
DGF_CONDITIONS_SD = 2.0
BIAS_SD = 0.2
LOG_NOISE_PRIOR = (np.log(0.1), 0.5)
SOLVE = functools.partial(
    get_measurables_time_course_single_experiment,
    solver=diffrax.Kvaerno4(),
)
FREE_PARAMETERS = {
    "log_kcat": [reaction for reaction in gdp_fucose.KCAT if reaction != "PPA"],
    "log_saturation_constant": [
        "ki|FKP_fk|gdp_fucose",
        "km|UNKNOWN_HYDROLASE|amp",
        "km|PPK3_A|adp",
        "km|PPK3_A|atp",
        "km|PPK3_G|gdp",
        "km|PPK3_G|gtp",
        "km|PPK3|polyp",
    ],
    "dgf": ["atp", "polyp", "gdp", "gmp"],
}


def get_problem(experiments):
    """Get what a fit to some experiments needs: their experiment layout, the
    split into free and fixed parameters, their time grids and measurements,
    and the free parameters' values in gdp_fucose, which serve as prior means.
    """
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


def get_prior(split, initial):
    """Get the prior: normal around initial, with standard deviation PRIOR_SD
    on the log scale except where WIDE_PRIOR_SD says otherwise, and
    multivariate normal for the free formation energies, given the fixed ones,
    with DGF_CONDITIONS_SD added independently to each."""
    free_dgf = list(get_free_labels(split, "dgf"))
    dgf_covariance = gdp_fucose.get_conditional_dgf_covariance(
        free_dgf,
    ) + DGF_CONDITIONS_SD**2 * np.eye(len(free_dgf))
    prior = prior_from_truth(initial, sd=PRIOR_SD)
    for parameter, sd_by_label in WIDE_PRIOR_SD.items():
        loc, scale = prior[parameter]
        labels = list(get_free_labels(split, parameter))
        for label, sd in sd_by_label.items():
            scale = scale.at[labels.index(label)].set(sd)
        prior[parameter] = (loc, scale)
    return prior | {"dgf": (initial["dgf"], jnp.asarray(dgf_covariance))}


def with_day_bias(measurements, bias, days):
    """Give each measurement the bias of the day it was measured on, from bias
    in the order of days."""
    return [
        LognormalMeasurement(
            experiment=m.experiment,
            ids=m.ids,
            values=m.values,
            errors=m.errors,
            bias=bias[days.index(gdp_fucose.MEASUREMENT_DAY[m.experiment])],
        )
        for m in measurements
    ]


def with_species_noise(measurements, log_noise):
    """Give each measured value the noise of its species, from log_noise, the
    log standard deviations in the order of gdp_fucose.MEASURED_SPECIES."""
    noise = jnp.exp(log_noise)
    return [
        LognormalMeasurement(
            experiment=m.experiment,
            ids=m.ids,
            values=m.values,
            errors=noise[
                np.array(
                    [
                        gdp_fucose.MEASURED_SPECIES.index(i.split("|")[0])
                        for i in m.ids
                    ],
                )
            ],
            bias=m.bias,
        )
        for m in measurements
    ]


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
    _, conc, _, _ = SOLVE(
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


def plot_fit(layout, split, times, measurements, free_by_label, bias_by_label):
    """Plot PLOT_EXPERIMENT at each set of free parameters, on shared axes.

    free_by_label maps a title, such as "MAP", to a set of free parameters,
    and bias_by_label maps the same titles to the bias of PLOT_EXPERIMENT's
    measurements, so that the curves show what the measurements should read.
    """
    ts = jnp.linspace(0.0, max(times[PLOT_EXPERIMENT].values()), 200)
    conc_by_label = {
        label: get_time_course(layout, split, free, PLOT_EXPERIMENT, ts)
        * np.exp(bias_by_label[label])
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
    prior = get_prior(split, initial)
    days = sorted({gdp_fucose.MEASUREMENT_DAY[e] for e in layout.experiments})
    start_values = {
        "model": initial,
        "bias": jnp.zeros(len(days)),
        "log_noise": jnp.full(
            len(gdp_fucose.MEASURED_SPECIES),
            LOG_NOISE_PRIOR[0],
        ),
    }

    def negative_log_density(values, args):
        return -(
            log_density_time_course(
                values["model"],
                gdp_fucose.model,
                prior,
                times,
                layout=layout,
                split=split,
                conc_measurements=with_species_noise(
                    with_day_bias(measurements, values["bias"], days),
                    values["log_noise"],
                ),
                get_measurables=SOLVE,
            )
            + norm.logpdf(values["bias"], scale=BIAS_SD).sum()
            + norm.logpdf(values["log_noise"], *LOG_NOISE_PRIOR).sum()
        )

    start = time.perf_counter()
    solution = optx.minimise(
        negative_log_density,
        optx.BFGS(rtol=1e-6, atol=1e-6),
        start_values,
        max_steps=1000,
        throw=False,
    )
    fitted = solution.value["model"]
    bias = {
        e: float(
            solution.value["bias"][days.index(gdp_fucose.MEASUREMENT_DAY[e])],
        )
        for e in layout.experiments
    }
    print(
        f"BFGS finished with result {solution.result} after "
        f"{int(solution.stats['num_steps'])} steps and "
        f"{time.perf_counter() - start:.0f} s.",
    )
    print(
        "Log density: "
        f"{-float(negative_log_density(start_values, None)):.1f} at the "
        f"prior mean, {-float(negative_log_density(solution.value, None)):.1f}"
        " at the MAP.",
    )
    print("Measurement bias by day, exponentiated:")
    for day, value in zip(days, np.exp(solution.value["bias"])):
        print(f"  {day} {value:.3f}")
    print("Measurement noise by species (standard deviation, log scale):")
    for species, value in zip(
        gdp_fucose.MEASURED_SPECIES,
        np.exp(solution.value["log_noise"]),
    ):
        print(f"  {species:10} {value:.3f}")
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
    report(
        "Fitting experiments:",
        layout,
        split,
        times,
        measurements,
        fitted,
        bias,
    )
    figures = plot_fit(
        layout,
        split,
        times,
        measurements,
        {"prior mean": initial, "MAP": fitted},
        {"prior mean": 0.0, "MAP": bias[PLOT_EXPERIMENT]},
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
