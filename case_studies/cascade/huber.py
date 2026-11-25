"""Simulate Huber et al.'s model of every run, for comparison.

Downloads their COPASI model and their 100-member parameter ensemble, fitted
by evolutionary strategy, from their repository at a fixed commit. Sets each
run's recorded initial amounts and enzyme levels, and simulates each
ensemble member with basico, COPASI's Python interface. Their model has no
AMP, and its ATP and ADP jump sharply in the first moments, as in their own
figures. Needs basico, which is not a dependency of enzax: run it with
uv run --with copasi-basico python huber.py. Saves results/huber.npz.
"""

import urllib.request

import numpy as np
import pandas as pd
from basico import (
    load_model,
    run_time_course,
    set_reaction_parameters,
    set_species,
)

from common import FITTING_EXPERIMENTS, HUBER_FILE, RESULTS_DIR
from enzax.examples import gdp_fucose
from predict import N_TIME

REPOSITORY = (
    "https://raw.githubusercontent.com/klamt-lab/GDP-Fucose_OptMet2023_Paper/"
    "2979fc9abd04f134ea399f6ddccc36804cc28fec/parameter_ensemble/"
)
MODEL_FILE = "GDP-Fucose_v7XGSK_with_PE9XGSK_setup.cps"
ENSEMBLE_FILE = "sampling_output_EvoStrat_100runs.csv"
SPECIES = {
    "atp": "ATP",
    "adp": "ADP",
    "gmp": "GMP",
    "gdp": "GDP",
    "gtp": "GTP",
    "gdp_fucose": "GDP_Fucose",
}
INITIAL_SPECIES = {"fucose": "Fuc", "atp": "ATP", "gmp": "GMP"}
ENZYMES = ["E_FKP", "E_PPA", "E_PPK3", "E_GMPK"]


def download(name):
    path = RESULTS_DIR / name
    if not path.exists():
        urllib.request.urlretrieve(REPOSITORY + name, path)
    return path


def simulate(ensemble, experiment, end):
    amounts = gdp_fucose.INITIAL_AMOUNTS[experiment]
    for species, name in INITIAL_SPECIES.items():
        set_species(name, initial_concentration=amounts[species])
    for name, level in zip(ENZYMES, gdp_fucose.ENZYME_LEVELS[experiment]):
        set_species(name, initial_concentration=level)
    courses = []
    for _, parameters in ensemble.iterrows():
        for name, value in parameters.items():
            set_reaction_parameters(name=name, value=value)
        result = run_time_course(duration=end, intervals=N_TIME - 1)
        courses.append(result[list(SPECIES.values())].to_numpy())
    return result.index.to_numpy(), np.stack(courses)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_model(str(download(MODEL_FILE)))
    ensemble = pd.read_csv(download(ENSEMBLE_FILE)).iloc[:, 1:-1]
    experiments = FITTING_EXPERIMENTS + gdp_fucose.VALIDATION_EXPERIMENTS
    times, _ = gdp_fucose.load_measurements(experiments)
    results = {}
    for experiment in experiments:
        ts, courses = simulate(
            ensemble,
            experiment,
            max(times[experiment].values()),
        )
        results[experiment] = courses
        results[f"{experiment}_time"] = ts
        print(f"Simulated {experiment}")
    np.savez(HUBER_FILE, species=np.array(list(SPECIES)), **results)
    print(f"Saved Huber et al.'s time courses to {HUBER_FILE}")


if __name__ == "__main__":
    main()
