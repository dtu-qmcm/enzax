"""A cell-free enzyme cascade that makes GDP-fucose, followed over time.

The cascade, the data and the experiment design come from Huber et al. (2023),
Metabolic Engineering 81:10-25, https://doi.org/10.1016/j.ymben.2023.10.007.
The model is the one in https://github.com/dtu-qmcm/gdp_fucose_enzax, which
fixes problems with the model as published, without that implementation's
competitive inhibition of FKP_fk by pyrophosphate, and the parameters are that
implementation's prior means. The measurements in gdp_fucose_measurements.csv
are that implementation's preparation of the data in
https://github.com/klamt-lab/GDP-Fucose_OptMet2023_Paper (Apache-2.0): the
adenosine nucleotides are rescaled to add up to the initial ATP, and the rows
at time 0 hold the known initial amounts rather than measurements.

Times are in hours, and amounts in mM in a pot of volume 1, so amounts and
concentrations are the same numbers. The formation energies are eQuilibrator's
estimates, converted from a 1 M to a 1 mM standard state. The pot is closed, so
four conserved moieties follow from the initial amounts. The solver needs
looser tolerances than enzax's defaults to get through the fast reactions:
get_measurables is the time-course solve with tolerances that work.
"""

import csv
import functools
from pathlib import Path

import numpy as np

from enzax.array_types import ParamDict
from enzax.kinetic_model import KineticModel
from enzax.measurements import LognormalMeasurement
from enzax.parameters import pack_parameters
from enzax.reactions import MichaelisMenten
from enzax.time_course import (
    get_initial_state_labelling,
    get_measurables_time_course_single_experiment,
)

DATA_FILE = Path(__file__).parent / "gdp_fucose_measurements.csv"
STOICHIOMETRY = {
    "FKP_fk": {"atp": -1.0, "fucose": -1.0, "adp": 1.0, "fucose1p": 1.0},
    "FKP_gt": {"gtp": -1.0, "fucose1p": -1.0, "pp": 1.0, "gdp_fucose": 1.0},
    "GMPK": {"atp": -1.0, "gmp": -1.0, "adp": 1.0, "gdp": 1.0},
    "PPA": {"pp": -1.0, "p": 2.0},
    "PPK3_A": {"adp": -1.0, "atp": 1.0, "polypnm1": 1.0},
    "PPK3_G": {"gdp": -1.0, "gtp": 1.0, "polypnm1": 1.0},
    "UNKNOWN_HYDROLASE": {"atp": -1.0, "amp": -1.0, "adp": 2.0},
}
ENZYMES = {
    "FKP_fk": "fkp",
    "FKP_gt": "fkp",
    "GMPK": "gmpk",
    "PPA": "ppa",
    "PPK3_A": "ppk3",
    "PPK3_G": "ppk3",
    "UNKNOWN_HYDROLASE": "unknown_hydrolase",
}
BALANCED_SPECIES = [
    "pp",
    "p",
    "atp",
    "adp",
    "amp",
    "fucose",
    "fucose1p",
    "gdp_fucose",
    "gtp",
    "gdp",
    "gmp",
]
MEASURED_SPECIES = ["amp", "adp", "atp", "gmp", "gdp", "gtp", "gdp_fucose"]
FITTING_EXPERIMENTS = [
    "FE13",
    "FE17_1FKP",
    "FE17_07FKP",
    "FE18_1FKP",
    "FE18_07FKP",
]
VALIDATION_EXPERIMENTS = ["FE20_Validation1", "FE20_Validation2"]
ENZYME_LEVELS = {
    "FE13": (0.002933939, 0.003106716, 0.00115141, 0.03306205),
    "FE17_1FKP": (0.00946432, 0.003106716, 0.00115141, 0.006358087),
    "FE17_07FKP": (0.006625024, 0.003106716, 0.00115141, 0.006358087),
    "FE18_1FKP": (0.00946432, 0.03106716, 0.002878526, 0.006358087),
    "FE18_07FKP": (0.006625024, 0.03106716, 0.002878526, 0.006358087),
    "FE20_Validation1": (0.01491, 0.020148, 0.009456, 0.002413),
    "FE20_Validation2": (0.007145, 0.007279, 0.00104, 0.002413),
}
UNKNOWN_HYDROLASE_LEVEL = 0.0001
INITIAL_AMOUNTS = {
    "FE13": {"gmp": 21.5, "fucose": 21.5, "atp": 6.0},
    "FE17_1FKP": {"gmp": 21.0, "fucose": 21.0, "atp": 5.6},
    "FE17_07FKP": {"gmp": 21.0, "fucose": 21.0, "atp": 5.6},
    "FE18_1FKP": {"gmp": 25.7, "fucose": 25.7, "atp": 6.0},
    "FE18_07FKP": {"gmp": 25.7, "fucose": 25.7, "atp": 6.0},
    "FE20_Validation1": {"gmp": 22.0, "fucose": 22.0, "atp": 5.0},
    "FE20_Validation2": {"gmp": 22.0, "fucose": 22.0, "atp": 5.0},
}
KCAT = {
    "FKP_fk": 1404.0,
    "FKP_gt": 36252.0,
    "GMPK": 752400.0,
    "PPA": 558000.0,
    "PPK3_A": 18780.0,
    "PPK3_G": 15048.0,
    "UNKNOWN_HYDROLASE": 10000.0,
}
SATURATION_CONSTANTS = {
    "km|FKP_fk|atp": 1.08,
    "km|FKP_fk|fucose": 0.045,
    "km|FKP_fk|adp": 1e-2,
    "km|FKP_fk|fucose1p": 1e-2,
    "km|FKP_gt|fucose1p": 0.03295,
    "km|FKP_gt|gtp": 0.012,
    "km|FKP_gt|pp": 0.135,
    "km|FKP_gt|gdp_fucose": 0.12,
    "km|GMPK|atp": 0.5,
    "km|GMPK|gmp": 0.156,
    "km|GMPK|adp": 0.017,
    "km|GMPK|gdp": 0.097,
    "km|PPA|pp": 0.003,
    "km|PPK3_A|adp": 2.5,
    "km|PPK3_A|atp": 2.0,
    "km|PPK3_A|polypnm1": 3.0,
    "km|PPK3_G|gdp": 1.2,
    "km|PPK3_G|gtp": 0.1,
    "km|PPK3_G|polypnm1": 3.0,
    "km|UNKNOWN_HYDROLASE|atp": 0.5,
    "km|UNKNOWN_HYDROLASE|amp": 0.5,
    "km|UNKNOWN_HYDROLASE|adp": 0.5,
}
TEMPERATURE = 298.15
DGF = {
    species: dgf - 0.008314 * TEMPERATURE * np.log(1000)
    for species, dgf in {
        "pp": -1943.542247,
        "p": -1055.471107,
        "atp": -2263.627453,
        "adp": -1388.741241,
        "amp": -514.189488,
        "fucose": -236.20833,
        "fucose1p": -1114.762772,
        "gdp_fucose": -1950.101734,
        "gtp": -2771.673718,
        "gdp": -1894.119776,
        "gmp": -1019.27507,
        "polypnm1": 875.5,
    }.items()
}

model = KineticModel(
    reactions={
        reaction: MichaelisMenten(
            stoichiometry=stoichiometry,
            enzyme=ENZYMES[reaction],
            reversible=reaction != "PPA",
        )
        for reaction, stoichiometry in STOICHIOMETRY.items()
    },
    balanced_species=BALANCED_SPECIES,
    moiety_label_species=["p", "amp", "fucose", "gmp"],
)
labelling = model.parameter_labelling | get_initial_state_labelling(model)
get_measurables = functools.partial(
    get_measurables_time_course_single_experiment,
    rtol=1e-6,
    atol=1e-6,
)


def get_parameters(experiment: str) -> ParamDict:
    """Get one experiment's parameters, at the prior means."""
    fkp, ppa, ppk3, gmpk = ENZYME_LEVELS[experiment]
    enzyme_levels = {
        "fkp": fkp,
        "gmpk": gmpk,
        "ppa": ppa,
        "ppk3": ppk3,
        "unknown_hydrolase": UNKNOWN_HYDROLASE_LEVEL,
    }
    initial = INITIAL_AMOUNTS[experiment]
    return pack_parameters(
        labelling,
        {
            "log_saturation_constant": {
                label: np.log(SATURATION_CONSTANTS[label])
                for label in labelling["log_saturation_constant"]
            },
            "log_kcat": {
                reaction: np.log(kcat) for reaction, kcat in KCAT.items()
            },
            "log_enzyme": {
                enzyme: np.log(level) for enzyme, level in enzyme_levels.items()
            },
            "dgf": {species: DGF[species] for species in labelling["dgf"]},
            "log_conc_unbalanced": {"polypnm1": 0.0},
            "moiety_totals": {
                species: 0.0 for species in labelling["moiety_totals"]
            },
            "log_amount_initial": {
                species: np.log(initial[species])
                if species in initial
                else -np.inf
                for species in BALANCED_SPECIES
            },
            "temperature": TEMPERATURE,
        },
    )


def load_measurements(
    experiments: list[str],
    error: float = 0.1,
) -> tuple[dict[str, dict[str, float]], list[LognormalMeasurement]]:
    """Get the experiments' time grids and concentration measurements.

    Each experiment's time points are labelled by the minute they were taken
    at, as in "50min". The rows at time 0 are left out, since they hold the
    initial amounts, and so are values of zero, which a lognormal measurement
    cannot score.

    :param error: the standard deviation of every measurement on the log
        scale.
    """
    with DATA_FILE.open() as f:
        rows = list(csv.DictReader(f))
    times = {}
    measurements = []
    for experiment in experiments:
        grid = {}
        ids = []
        values = []
        for row in rows:
            time = float(row["time"])
            if row["experiment"] != experiment or time == 0.0:
                continue
            label = f"{round(time * 60)}min"
            grid[label] = time
            for species in MEASURED_SPECIES:
                if row[species] and float(row[species]) > 0:
                    ids.append(f"{species}|{label}")
                    values.append(float(row[species]))
        times[experiment] = grid
        measurements.append(
            LognormalMeasurement(
                experiment=experiment,
                ids=ids,
                values=np.array(values),
                errors=np.full(len(values), error),
            ),
        )
    return times, measurements


parameters = get_parameters("FE13")
