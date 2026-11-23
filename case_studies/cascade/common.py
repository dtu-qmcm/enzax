import functools
from pathlib import Path

import diffrax
import jax
import numpy as np
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

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
DOCS_IMG_DIR = HERE.parents[1] / "docs" / "img"
PRIOR_SD = 1.0
WIDE_PRIOR_SD = {
    "log_saturation_constant": {"ki|FKP_fk|gdp_fucose": 3.0},
}
DGF_CONDITIONS_SD = 2.0
BIAS_SD = 0.2
LOG_NOISE_PRIOR = (np.log(0.1), 0.5)
TOLERANCE = 1e-5
SOLVE = functools.partial(
    get_measurables_time_course_single_experiment,
    solver=diffrax.Kvaerno4(),
    rtol=TOLERANCE,
    atol=TOLERANCE,
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


def get_days(experiments):
    return sorted({gdp_fucose.MEASUREMENT_DAY[e] for e in experiments})


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


def get_posterior():
    layout, split, times, measurements, nominal = get_problem(
        gdp_fucose.FITTING_EXPERIMENTS,
    )
    prior = get_prior(split, nominal)
    days = get_days(layout.experiments)

    def log_posterior(values):
        return (
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

    start = {
        "model": nominal,
        "bias": jnp.zeros(len(days)),
        "log_noise": jnp.full(
            len(gdp_fucose.MEASURED_SPECIES),
            LOG_NOISE_PRIOR[0],
        ),
    }
    return layout, split, times, measurements, log_posterior, start


def get_labels(split, values):
    labels = {
        "model": {
            parameter: [
                f"{parameter}[{label}]"
                for label in get_free_labels(split, parameter)
            ]
            for parameter in values["model"]
        },
        "bias": [
            f"bias[{day}]" for day in get_days(gdp_fucose.FITTING_EXPERIMENTS)
        ],
        "log_noise": [
            f"log_noise[{species}]" for species in gdp_fucose.MEASURED_SPECIES
        ],
    }
    leaves = jax.tree.leaves(labels, is_leaf=lambda x: isinstance(x, list))
    return [label for leaf in leaves for label in leaf]


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
