"""Plot the improved red cell model's sensitivities for the supplement.

Shows the log sensitivities of volume and Donnan ratio from sensitivities.py
at 37 °C and pH 7.4, for every parameter among the eight with the largest
effect on either output, external pH excluded. Bars show the derivatives
through the DAE steady state and ticks the central finite differences. Saves
results/sensitivities.png and results/sensitivities.pdf.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from figure import MODEL_COLOUR, MUTED, RC_PARAMS, TEXT, style
from ph_sweep import RESULTS_DIR
from sensitivities import SENSITIVITY_FILE

SENSITIVITY_OUTPUTS = {
    "volume": "volume",
    "donnan_ratio": "Donnan ratio",
}
N_SENSITIVITY = 8
EXCLUDED_PARAMETERS = {"cu|plasma|ph"}
PARAMETER_NAMES = {
    "cu|cell|k": "Intracellular K⁺",
    "na_e": "External Na⁺",
    "cl_e": "External Cl⁻",
    "cu|pump|vm": "Na⁺/K⁺ pump $V_{max}$",
    "cu|cell|other": "Other impermeant solutes",
    "cu|leak_Na|vm": "Na⁺ carrier $V_{max}$",
    "cu|leak_Na|k_perm": "Na⁺ permeability",
    "cu|cell|hb": "Haemoglobin",
    "cu|HK|k_mgatp": "Hexokinase $K_{MgATP}$",
    "mg": "Total Mg",
}


def read_sensitivities():
    with open(SENSITIVITY_FILE) as f:
        return [
            row
            for row in csv.DictReader(f)
            if row["label"] not in EXCLUDED_PARAMETERS
        ]


def plot_sensitivities(axes):
    rows = read_sensitivities()
    shown = []
    for output in SENSITIVITY_OUTPUTS:
        ranked = sorted(
            rows,
            key=lambda row: -abs(float(row[f"{output}_implicit"])),
        )
        shown += [row for row in ranked[:N_SENSITIVITY] if row not in shown]
    positions = np.arange(len(shown))[::-1]
    for ax, (output, name) in zip(
        axes,
        SENSITIVITY_OUTPUTS.items(),
        strict=True,
    ):
        ax.barh(
            positions,
            [float(row[f"{output}_implicit"]) for row in shown],
            color=MODEL_COLOUR,
            height=0.6,
            label="Implicit differentiation",
        )
        ax.plot(
            [float(row[f"{output}_finite_difference"]) for row in shown],
            positions,
            color=TEXT,
            linestyle="none",
            marker="|",
            markersize=8,
            label="Finite differences",
        )
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlabel(f"Log sensitivity of {name}")
        style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(
        positions,
        [PARAMETER_NAMES[row["label"]] for row in shown],
    )
    axes[0].legend(frameon=False, loc="lower right")


def main():
    plt.rcParams.update(RC_PARAMS)
    fig, axes = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(7.5, 2.4),
        layout="constrained",
    )
    plot_sensitivities(axes)
    for suffix in ("png", "pdf"):
        path = RESULTS_DIR / f"sensitivities.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
