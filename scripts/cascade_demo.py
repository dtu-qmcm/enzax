"""Fit the GDP-fucose cascade's kinetic parameters to time-course data.

The five fitting experiments share their kinetic parameters and differ in
their enzyme loadings and initial amounts. The turnover numbers, except
pyrophosphatase's, the Michaelis constants of the nuisance reaction
UNKNOWN_HYDROLASE, and the amount of polyphosphate, which is the same in every
experiment, are inferred by maximum a posteriori optimisation; everything else
stays at the values in `enzax.examples.gdp_fucose`. The fitted model then
predicts two validation experiments that it was not fitted to.
"""

# ruff: noqa: E402

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import time

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

jax.config.update("jax_enable_x64", True)

PRIOR_SD = 1.0
FREE_PARAMETERS = {
    "log_kcat": [reaction for reaction in gdp_fucose.KCAT if reaction != "PPA"],
    "log_amount_initial": ["polyp"],
    "log_saturation_constant": [
        "km|UNKNOWN_HYDROLASE|atp",
        "km|UNKNOWN_HYDROLASE|amp",
        "km|UNKNOWN_HYDROLASE|adp",
    ],
}
EXPERIMENT_SPECIFIC = {
    "log_enzyme": None,
    "log_amount_initial": [
        species for species in gdp_fucose.BALANCED_SPECIES if species != "polyp"
    ],
}


def get_problem(experiments):
    layout = get_experiment_layout(
        gdp_fucose.labelling,
        experiments,
        EXPERIMENT_SPECIFIC,
    )
    expanded = merge_experiment_parameters(
        layout,
        [gdp_fucose.get_parameters(e) for e in experiments],
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
    parameters = gather_parameters_by_experiment(
        layout,
        combine_parameters(split, free),
    )
    print(title)
    for experiment, measurement in zip(layout.experiments, measurements):
        grid = times[experiment]
        last = list(grid)[-1]
        rows = dict(zip(measurement.ids, measurement.values.tolist()))
        observed = rows.get(f"gdp_fucose|{last}", float("nan"))
        single = pick_experiment(layout, parameters, experiment)
        _, conc, _, _ = gdp_fucose.get_measurables(
            gdp_fucose.model,
            single,
            jnp.exp(single["log_amount_initial"]),
            jnp.array(list(grid.values())),
        )
        predicted = conc[-1, gdp_fucose.model.species.index("gdp_fucose")]
        print(
            f"  {experiment:17} GDP-fucose at {last:>7}: predicted "
            f"{float(predicted):6.2f} mM, measured {observed:6.2f} mM",
        )


def main():
    layout, split, times, measurements, initial = get_problem(
        gdp_fucose.FITTING_EXPERIMENTS,
    )
    prior = prior_from_truth(initial, sd=PRIOR_SD)

    def negative_log_density(free, args):
        return -log_density_time_course(
            free,
            gdp_fucose.model,
            prior,
            times,
            layout=layout,
            split=split,
            conc_measurements=measurements,
            get_measurables=gdp_fucose.get_measurables,
        )

    start = time.perf_counter()
    solution = optx.minimise(
        negative_log_density,
        optx.BFGS(rtol=1e-6, atol=1e-6),
        initial,
        max_steps=500,
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
    print("Free parameters, exponentiated:")
    for parameter in initial:
        for label, before, after in zip(
            get_free_labels(split, parameter),
            np.exp(initial[parameter]),
            np.exp(fitted[parameter]),
        ):
            print(
                f"  {parameter} {label:18} prior mean {before:10.4g}, "
                f"MAP {after:10.4g}",
            )
    report("Fitting experiments:", layout, split, times, measurements, fitted)
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
