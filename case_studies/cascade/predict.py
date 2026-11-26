"""Draw every run's readings from the posterior predictive distribution.

Takes 200 draws from fit.py and solves each run on a grid of 120 times,
converting the concentrations into what the instrument would read: GDP and
GTP calibrated, and a fraction of GDP-fucose read as GMP. Fitting runs use
their fitted enzyme levels, initial amounts, bias and fraction. Validation
runs use their recorded enzyme levels and initial amounts, a new day's bias
drawn from its prior, and the shared fraction times a new day's factor drawn
from its prior. Each draw's readings are then multiplied by 25 draws of
lognormal measurement noise, with a new day's bias and factor for each in
validation runs. Saves results/predictions.npz.
"""

import os

os.environ.setdefault("EQX_ON_ERROR", "nan")

import sys  # noqa: E402

import arviz as az  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

from common import (  # noqa: E402
    ARTEFACT_DAY_SD,
    BIAS_SD,
    FITTING_EXPERIMENTS,
    MEASUREMENT_ERRORS,
    RESULTS_DIR,
    get_days,
    get_fractions,
    get_posterior,
    get_problem,
    get_reading,
    get_time_course,
)
from enzax.examples import gdp_fucose  # noqa: E402
from enzax.parameter_split import get_free_labels  # noqa: E402
from fit import get_results_dir  # noqa: E402

SEED = 1234
N_DRAW = 200
N_TIME = 120
N_NOISE = 25
NOISE = np.array(
    [MEASUREMENT_ERRORS[species] for species in gdp_fucose.MEASURED_SPECIES],
)
PREDICTIONS_FILE = RESULTS_DIR / "predictions.npz"


def get_draws(run, unflatten):
    posterior = az.from_netcdf(get_results_dir(run) / "draws.nc").posterior
    flat = posterior["parameters"].values
    flat = flat.reshape(-1, flat.shape[-1])
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(flat), size=min(N_DRAW, len(flat)), replace=False)
    return [unflatten(jnp.asarray(x)) for x in flat[chosen]]


def get_day_effects(values, day, days, fitted, rng):
    fraction = get_fractions(values, [day])[day]
    if fitted:
        return float(values["bias"][days.index(day)]), fraction
    return (
        rng.normal(0.0, BIAS_SD),
        fraction * np.exp(rng.normal(0.0, ARTEFACT_DAY_SD)),
    )


def predict(experiments, fitted_split, draws, fitted, rng):
    layout, split, times, _, nominal = get_problem(experiments)
    shared = [
        parameter
        for parameter in nominal
        if get_free_labels(split, parameter)
        == get_free_labels(fitted_split, parameter)
    ]
    days = get_days(FITTING_EXPERIMENTS)
    measured = np.array(
        [
            gdp_fucose.model.species.index(species)
            for species in gdp_fucose.MEASURED_SPECIES
        ],
    )
    predictions = {}
    for experiment in experiments:
        ts = jnp.linspace(0.0, max(times[experiment].values()), N_TIME)
        solve = jax.jit(
            lambda free: get_time_course(layout, split, free, experiment, ts),
        )
        courses = []
        day = gdp_fucose.MEASUREMENT_DAY[experiment]
        for values in draws:
            free = nominal | {p: values["model"][p] for p in shared}
            conc = np.asarray(solve(free))[:, measured]
            for _ in range(N_NOISE):
                bias, fraction = get_day_effects(values, day, days, fitted, rng)
                reading = get_reading(conc, fraction, bias)
                noise = rng.normal(0.0, NOISE, size=reading.shape)
                courses.append(reading * np.exp(noise))
        predictions[experiment] = np.stack(courses)
        predictions[f"{experiment}_time"] = np.asarray(ts)
        failed = np.isnan(predictions[experiment]).any(axis=(1, 2)).sum()
        failed //= N_NOISE
        print(f"{experiment}: {failed} of {len(draws)} solves failed")
    return predictions


def main(run):
    _, split, _, _, _, start = get_posterior()
    _, unflatten = ravel_pytree(start)
    draws = get_draws(run, unflatten)
    rng = np.random.default_rng(SEED)
    predictions = predict(
        FITTING_EXPERIMENTS,
        split,
        draws,
        fitted=True,
        rng=rng,
    ) | predict(
        gdp_fucose.VALIDATION_EXPERIMENTS,
        split,
        draws,
        fitted=False,
        rng=rng,
    )
    np.savez(
        PREDICTIONS_FILE,
        species=np.array(gdp_fucose.MEASURED_SPECIES),
        **predictions,
    )
    print(f"Saved the predictions to {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
