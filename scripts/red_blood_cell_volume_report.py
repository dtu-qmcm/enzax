"""Report the variable-volume red cell's steady-state volume and Donnan ratio
at several external pHs, to compare with Joshi and Palsson's Part IV."""

import jax
import numpy as np

from enzax.examples import red_blood_cell_volume as rbcv
from enzax.parameters import get_parameter_position
from enzax.steady_state import get_steady_state_dae

jax.config.update("jax_enable_x64", True)

EXTERNAL_PHS = [6.0, 7.4, 8.9]


def with_external_ph(model, parameters, ph):
    position = get_parameter_position(
        model.parameter_labelling,
        "log_custom",
        "cu|plasma|ph",
    )
    log_custom = parameters["log_custom"].at[position].set(np.log(ph))
    return parameters | {"log_custom": log_custom}


for name, model, parameters, initial_state in [
    (
        "faithful",
        rbcv.faithful,
        rbcv.faithful_parameters,
        rbcv.faithful_initial_state,
    ),
    (
        "improved",
        rbcv.improved,
        rbcv.improved_parameters,
        rbcv.improved_initial_state,
    ),
]:
    for ph in EXTERNAL_PHS:
        at_ph = with_external_ph(model, parameters, ph)
        steady = get_steady_state_dae(model, initial_state, at_ph)
        volume, donnan_ratio = model.get_algebraic_variables(steady, at_ph)
        print(f"{name} pH_e {ph}: V = {volume:.3f}, r = {donnan_ratio:.3f}")
