import csv
from pathlib import Path

import jax
import numpy as np

from enzax.examples import red_blood_cell_volume as rbcv
from enzax.parameters import get_parameter_position
from enzax.steady_state import get_steady_state_dae

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
SWEEP_FILE = RESULTS_DIR / "ph_sweep.csv"
VARIANTS = {
    "improved": (
        rbcv.improved,
        rbcv.improved_parameters,
        rbcv.improved_initial_state,
    ),
    "faithful": (
        rbcv.faithful,
        rbcv.faithful_parameters,
        rbcv.faithful_initial_state,
    ),
}
REFERENCE_PH = 7.4
PH_E = np.round(np.arange(5.7, 9.6 + 1e-9, 0.05), 2)
GUNN_TEMPERATURE = 273.15
GUNN_MEDIUM = {"cl_e": 150.0, "hco3_e": 1e-6}


def with_log_value(model, parameters, kind, label, value):
    position = get_parameter_position(model.parameter_labelling, kind, label)
    return parameters | {
        kind: parameters[kind].at[position].set(np.log(value)),
    }


def get_gunn_parameters(model, parameters, ph_e):
    gunn = with_log_value(model, parameters, "log_custom", "cu|plasma|ph", ph_e)
    gunn |= {"temperature": np.asarray(GUNN_TEMPERATURE)}
    for species, conc in GUNN_MEDIUM.items():
        gunn = with_log_value(
            model,
            gunn,
            "log_conc_unbalanced",
            species,
            conc,
        )
    return gunn


def sweep(model, parameters, initial_state):
    contents = get_steady_state_dae(model, initial_state, parameters)
    solve = jax.jit(lambda p: model.get_algebraic_variables(contents, p))
    volumes, ratios = [], []
    for ph_e in PH_E:
        volume, psi = solve(get_gunn_parameters(model, parameters, ph_e))
        volumes.append(float(volume))
        ratios.append(float(rbcv.get_donnan_ratio(psi, GUNN_TEMPERATURE)))
    reference = volumes[list(PH_E).index(REFERENCE_PH)]
    return np.array(volumes) / reference, np.array(ratios)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(SWEEP_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "ph_e", "relative_volume", "donnan_ratio"])
        for name, variant in VARIANTS.items():
            volumes, ratios = sweep(*variant)
            for row in zip(PH_E, volumes, ratios, strict=True):
                writer.writerow([name, *row])
    print(f"Saved {SWEEP_FILE}")


if __name__ == "__main__":
    main()
