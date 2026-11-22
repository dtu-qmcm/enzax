import csv
import time

import jax
import jax.numpy as jnp
import numpy as np

from enzax.examples import red_blood_cell_volume as rbcv
from enzax.steady_state import get_steady_state_dae
from ph_sweep import RESULTS_DIR

jax.config.update("jax_enable_x64", True)

SENSITIVITY_FILE = RESULTS_DIR / "sensitivities.csv"
STEP = 1e-4
LOG_KINDS = ["log_k_plus", "log_custom", "log_conc_unbalanced"]
MODEL = rbcv.improved
PARAMETERS = rbcv.improved_parameters


@jax.jit
def get_outputs(parameters, guess):
    state = get_steady_state_dae(MODEL, guess, parameters)
    volume, psi = MODEL.get_algebraic_variables(state, parameters)
    donnan_ratio = rbcv.get_donnan_ratio(psi, parameters["temperature"])
    return jnp.log(jnp.array([volume, donnan_ratio]))


def get_scaled_jacobian(jacobian):
    scaled = {kind: jacobian[kind] for kind in LOG_KINDS}
    scaled["moiety_totals"] = (
        jacobian["moiety_totals"] * PARAMETERS["moiety_totals"]
    )
    return scaled


def perturb(kind, position, step):
    if kind == "moiety_totals":
        new = PARAMETERS[kind].at[position].multiply(np.exp(step))
    else:
        new = PARAMETERS[kind].at[position].add(step)
    return PARAMETERS | {kind: new}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    guess = get_steady_state_dae(
        MODEL,
        rbcv.improved_initial_state,
        PARAMETERS,
    )
    start = time.perf_counter()
    jacobian = get_scaled_jacobian(
        jax.jacrev(get_outputs)(PARAMETERS, guess),
    )
    print(f"Implicit gradient took {time.perf_counter() - start:.1f} s")
    rows = []
    for kind, scaled in jacobian.items():
        labels = MODEL.parameter_labelling[kind]
        for position, label in enumerate(labels):
            up = get_outputs(perturb(kind, position, STEP), guess)
            down = get_outputs(perturb(kind, position, -STEP), guess)
            finite_difference = (up - down) / (2 * STEP)
            rows.append(
                [
                    kind,
                    label,
                    *np.asarray(scaled[:, position]),
                    *np.asarray(finite_difference),
                ],
            )
    print(f"Finite differences done at {time.perf_counter() - start:.1f} s")
    with open(SENSITIVITY_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "kind",
                "label",
                "volume_implicit",
                "donnan_ratio_implicit",
                "volume_finite_difference",
                "donnan_ratio_finite_difference",
            ],
        )
        writer.writerows(rows)
    print(f"Saved {SENSITIVITY_FILE}")


if __name__ == "__main__":
    main()
