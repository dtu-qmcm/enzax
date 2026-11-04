"""Write the red blood cell examples' steady states, as enzax solves them, to
`tests/data/red_blood_cell_steady_states.json`.

The red blood cell tests check the model against these stored states
instead of solving it, which takes minutes. Run this script, and check
what changed, after changing the red blood cell examples on purpose.

Run with `uv run python scripts/red_blood_cell_reference_states.py`.
"""

import json
from pathlib import Path

import jax
import numpy as np

from enzax.examples import red_blood_cell as rbc
from enzax.examples import red_blood_cell_volume as rbcv
from enzax.steady_state import get_steady_state_hybrid

jax.config.update("jax_enable_x64", True)

OUT = (
    Path(__file__).parent.parent
    / "tests"
    / "data"
    / "red_blood_cell_steady_states.json"
)
FORWARD_TO_NET_RATIOS = (1e3, 1e4)


def solve(model, guess, parameters, **kwargs) -> list[float]:
    steady = get_steady_state_hybrid(model, guess, parameters, **kwargs)
    if not np.all(np.isfinite(steady)):
        msg = "A steady state solve failed."
        raise RuntimeError(msg)
    return np.asarray(steady).tolist()


def main():
    states = {
        "stiff": solve(rbc.model, rbc.steady_state, rbc.parameters),
        "rapid_equilibrium": solve(
            rbc.rapid_equilibrium_model,
            rbc.rapid_equilibrium_steady_state,
            rbc.rapid_equilibrium_parameters,
        ),
        "faithful": solve(
            rbcv.faithful,
            rbcv.faithful_initial_state,
            rbcv.faithful_parameters,
        ),
        "improved": solve(
            rbcv.improved,
            rbcv.improved_initial_state,
            rbcv.improved_parameters,
        ),
    }
    for ratio in FORWARD_TO_NET_RATIOS:
        states[f"stiff_ratio_{ratio:.0e}"] = solve(
            rbc.model,
            rbc.steady_state,
            rbc.get_parameters(rbc.model, ratio),
            steady_state_rtol=1e-9,
            steady_state_atol=1e-9,
        )
    OUT.write_text(json.dumps(states, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
