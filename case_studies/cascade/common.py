"""Shared setup for the GDP-fucose cascade case study.

Fits Huber et al.'s fitting runs except FE13, whose GDP-fucose curve alone
supports a second posterior mode that predicts the validation runs badly.
Defines the free parameters and their prior: normal on the log scale, with
each run's FKP and PPK3 levels correlated within a measurement day, and the
free formation energies conditional on the fixed ones. Also defines the
measurement model: fixed noise from the triplicates, fixed GDP and GTP
calibration, a bias for each measurement day, and a fraction of GDP-fucose
read as GMP that varies by day. Provides the log posterior and the solver
settings that the other scripts use.
"""

import dataclasses
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
from enzax.measurements import Measurement
from enzax.parameter_split import (
    combine_parameters,
    get_free_labels,
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.parameters import SEP
from enzax.statistical_modelling import log_density_time_course
from enzax.time_course import get_measurables_time_course_single_experiment

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
DOCS_IMG_DIR = HERE.parents[1] / "docs" / "img"
HUBER_FILE = RESULTS_DIR / "huber.npz"
FITTING_EXPERIMENTS = [
    experiment
    for experiment in gdp_fucose.FITTING_EXPERIMENTS
    if experiment != "FE13"
]
PRIOR_SD = 1.0
WIDE_PRIOR_SD = {
    "log_kcat": {"PPK3_M": 3.0, "ATP_HYDROLYSIS": 3.0},
    "log_saturation_constant": {"ki|FKP_fk|gdp_fucose": 3.0},
}
INITIAL_AMOUNT_SD = {
    "gmp": 0.1,
    "fucose": 0.5,
    "atp": 0.1,
    "amp": 1.0,
    "adp": 0.5,
}
RUN_ENZYMES = ["fkp", "ppk3"]
ENZYME_DAY_SD = 0.2
ENZYME_RUN_SD = 0.03
DGF_CONDITIONS_SD = 2.0
BIAS_SD = 0.2
MEASUREMENT_ERRORS = {
    "amp": 0.1,
    "adp": 0.052,
    "atp": 0.054,
    "gmp": 0.091,
    "gdp": 0.05,
    "gtp": 0.055,
    "gdp_fucose": 0.05,
}
RESPONSE_FACTORS = {"gdp": 1.13, "gtp": 1.33}
ARTEFACT_PRIOR = (np.log(0.05), 1.5)
ARTEFACT_DAY_SD = 0.5
TOLERANCE = 1e-4
SOLVE = functools.partial(
    get_measurables_time_course_single_experiment,
    solver=diffrax.Kvaerno3(),
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
        "km|PPK3_M|amp",
        "km|PPK3_M|adp",
    ],
    "dgf": ["atp", "gdp", "gmp"],
}


def get_freed_labels(experiments):
    runs = [e for e in experiments if e in FITTING_EXPERIMENTS]
    if not runs:
        return FREE_PARAMETERS
    run_specific = {
        "log_enzyme": RUN_ENZYMES,
        "log_amount_initial": list(INITIAL_AMOUNT_SD),
    }
    return FREE_PARAMETERS | {
        parameter: [f"{name}{SEP}{run}" for name in names for run in runs]
        for parameter, names in run_specific.items()
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
        get_freed_labels(experiments),
    )
    times, measurements = gdp_fucose.load_measurements(experiments)
    measurements = [
        dataclasses.replace(
            m,
            errors=np.array(
                [MEASUREMENT_ERRORS[i.split("|")[0]] for i in m.ids],
            ),
        )
        for m in measurements
    ]
    nominal = get_free_parameters(split, expanded)
    return layout, split, times, measurements, nominal


def get_prior_sd(parameter, label):
    if parameter == "log_amount_initial":
        return INITIAL_AMOUNT_SD[label.split(SEP)[0]]
    return WIDE_PRIOR_SD.get(parameter, {}).get(label, PRIOR_SD)


def get_enzyme_covariance(split):
    enzyme_days = [
        (enzyme, gdp_fucose.MEASUREMENT_DAY[experiment])
        for enzyme, experiment in (
            label.split(SEP) for label in get_free_labels(split, "log_enzyme")
        )
    ]
    same_day = np.array([[a == b for b in enzyme_days] for a in enzyme_days])
    return jnp.asarray(
        ENZYME_DAY_SD**2 * same_day
        + ENZYME_RUN_SD**2 * np.eye(len(enzyme_days)),
    )


def get_prior(split, initial):
    """Get the prior: normal around initial, with standard deviation PRIOR_SD
    on the log scale except where WIDE_PRIOR_SD or INITIAL_AMOUNT_SD say
    otherwise, multivariate normal for the free enzyme levels, and
    multivariate normal for the free formation energies, given the fixed ones,
    with DGF_CONDITIONS_SD added independently to each."""
    prior = {
        parameter: (
            loc,
            jnp.array(
                [
                    get_prior_sd(parameter, label)
                    for label in get_free_labels(split, parameter)
                ],
            ),
        )
        for parameter, loc in initial.items()
    }
    if "log_enzyme" in initial:
        prior["log_enzyme"] = (
            initial["log_enzyme"],
            get_enzyme_covariance(split),
        )
    free_dgf = list(get_free_labels(split, "dgf"))
    dgf_covariance = gdp_fucose.get_conditional_dgf_covariance(
        free_dgf,
    ) + DGF_CONDITIONS_SD**2 * np.eye(len(free_dgf))
    return prior | {"dgf": (initial["dgf"], jnp.asarray(dgf_covariance))}


def get_days(experiments):
    return sorted({gdp_fucose.MEASUREMENT_DAY[e] for e in experiments})


class ArtefactMeasurement(Measurement):
    values: jax.Array
    errors: jax.Array
    bias: jax.Array
    own_weights: jax.Array
    source_weights: jax.Array

    def log_likelihood(self, predicted):
        n = len(self.values)
        read = (
            self.own_weights * predicted[:n]
            + self.source_weights * predicted[n:]
        )
        return norm.logpdf(
            jnp.log(self.values),
            jnp.log(read) + self.bias,
            self.errors,
        )


def with_artefact(measurement, fraction, bias):
    species, labels = zip(*(i.split("|") for i in measurement.ids))
    response = np.array([RESPONSE_FACTORS.get(s, 1.0) for s in species])
    is_source = np.array([s == "gdp_fucose" for s in species])
    is_target = np.array([s == "gmp" for s in species])
    return ArtefactMeasurement(
        experiment=measurement.experiment,
        ids=(*measurement.ids, *(f"gdp_fucose|{label}" for label in labels)),
        values=measurement.values,
        errors=measurement.errors,
        bias=bias,
        own_weights=response * jnp.where(is_source, 1 - fraction, 1.0),
        source_weights=jnp.where(is_target, fraction, 0.0),
    )


def get_reading(conc, fraction, bias):
    species = gdp_fucose.MEASURED_SPECIES
    source = conc[..., species.index("gdp_fucose")]
    read = conc * np.array([RESPONSE_FACTORS.get(s, 1.0) for s in species])
    read[..., species.index("gmp")] += fraction * source
    read[..., species.index("gdp_fucose")] -= fraction * source
    return read * np.exp(bias)


def get_fractions(values, days):
    fitted = get_days(FITTING_EXPERIMENTS)
    return {
        day: float(
            np.exp(
                values["log_artefact"]
                + (
                    values["log_artefact_day"][fitted.index(day)]
                    if day in fitted
                    else 0.0
                ),
            ),
        )
        for day in days
    }


def get_posterior():
    layout, split, times, measurements, nominal = get_problem(
        FITTING_EXPERIMENTS,
    )
    prior = get_prior(split, nominal)
    days = get_days(layout.experiments)
    positions = [
        days.index(gdp_fucose.MEASUREMENT_DAY[m.experiment])
        for m in measurements
    ]

    def log_posterior(values):
        fractions = jnp.exp(values["log_artefact"] + values["log_artefact_day"])
        return (
            log_density_time_course(
                values["model"],
                gdp_fucose.model,
                prior,
                times,
                layout=layout,
                split=split,
                conc_measurements=[
                    with_artefact(m, fractions[day], values["bias"][day])
                    for m, day in zip(measurements, positions)
                ],
                get_measurables=SOLVE,
            )
            + norm.logpdf(values["bias"], scale=BIAS_SD).sum()
            + norm.logpdf(values["log_artefact"], *ARTEFACT_PRIOR)
            + norm.logpdf(
                values["log_artefact_day"],
                scale=ARTEFACT_DAY_SD,
            ).sum()
        )

    start = {
        "model": nominal,
        "bias": jnp.zeros(len(days)),
        "log_artefact": jnp.asarray(ARTEFACT_PRIOR[0]),
        "log_artefact_day": jnp.zeros(len(days)),
    }
    return layout, split, times, measurements, log_posterior, start


def get_labels(split, values):
    days = get_days(FITTING_EXPERIMENTS)
    labels = {
        "model": {
            parameter: [
                f"{parameter}[{label}]"
                for label in get_free_labels(split, parameter)
            ]
            for parameter in values["model"]
        },
        "bias": [f"bias[{day}]" for day in days],
        "log_artefact": ["log_artefact"],
        "log_artefact_day": [f"log_artefact_day[{day}]" for day in days],
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
