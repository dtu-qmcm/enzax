"""Plot the red cell figure.

A maps the network, with rapid equilibria in orange. B shows the Donnan ratio
and the volume relative to pH 7.4 against external pH, from ph_sweep.py, for
the improved and faithful variants, with Joshi and Palsson's (1990) model
curves and Gunn et al.'s (1973) regression lines. C shows the time each
solver takes to reach steady state: for the stiff formulation across
forward-to-net flux ratios, and for the rapid equilibrium formulation as
horizontal lines. Below it, C shows the largest relative difference between
the two formulations' steady-state concentrations. D shows the largest log
sensitivities of volume and Donnan ratio from sensitivities.py, with finite
differences as ticks. Saves results/figure.png and results/figure.pdf.

data/joshi_palsson.csv holds Joshi and Palsson's model curves, read off
Figs. 5 and 6 of their 1991 erratum, which replace Part IV's, to within about
2% of each axis. Gunn et al.'s lines come from the regressions in their
Fig. 3, of chloride ratio and of water content on pH over pH 5.7 to 9.6,
with the published standard deviations of intercept and slope. The bands
cover 95% of lines drawn from those estimates, correlated as an even spread
of pH over the range implies. The water content lines are divided by their
values at pH 7.4.
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from benchmark import BENCHMARK_FILE
from network import draw_reactions, plot_network
from ph_sweep import HERE, REFERENCE_PH, RESULTS_DIR, SWEEP_FILE
from sensitivities import SENSITIVITY_FILE

JOSHI_PALSSON_FILE = HERE / "data" / "joshi_palsson.csv"
GUNN_PH_RANGE = (5.7, 9.6)
GUNN_DONNAN_RATIO = {"intercept": (3.18, 0.05), "slope": (-0.312, 0.007)}
GUNN_WATER_CONTENT = {"intercept": (3.56, 0.09), "slope": (-0.241, 0.012)}
N_DRAW = 4000
QUANTILES = (0.025, 0.975)
MODEL_COLOUR = "#2a78d6"
JOSHI_PALSSON_COLOUR = "#eb6834"
GUNN_COLOUR = "#52514e"
VARIANT_STYLES = {"improved": "-", "faithful": "--"}
VARIANT_NAMES = {"improved": "Improved", "faithful": "Faithful"}
BENCHMARK_TOLERANCE = 1e-9
METHODS = {
    "bdf": ("ODE", "#2a78d6"),
    "dae": ("DAE", "#eb6834"),
    "hybrid": ("Hybrid", "#1baf7a"),
}
FORMULATIONS = {
    "stiff": ("Stiff", "-"),
    "rapid equilibrium": ("Rapid equilibrium", "--"),
}
ERROR_GUIDE_RATIO = 1e4
TIME_AXIS_TOP = 1e5
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
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def read_sweep():
    sweep = defaultdict(lambda: defaultdict(list))
    with open(SWEEP_FILE) as f:
        for row in csv.DictReader(f):
            for column in ("ph_e", "relative_volume", "donnan_ratio"):
                sweep[row["variant"]][column].append(float(row[column]))
    return sweep


def read_joshi_palsson():
    curves = defaultdict(lambda: ([], []))
    with open(JOSHI_PALSSON_FILE) as f:
        for row in csv.DictReader(f):
            ph_e, value = curves[row["quantity"]]
            ph_e.append(float(row["ph_e"]))
            value.append(float(row["value"]))
    return curves


def draw_gunn_lines(regression, ph_e, rng):
    low, high = GUNN_PH_RANGE
    mean_ph = (low + high) / 2
    sd_ph = (high - low) / np.sqrt(12)
    correlation = -mean_ph / np.hypot(sd_ph, mean_ph)
    (intercept, sd_intercept), (slope, sd_slope) = regression.values()
    covariance = correlation * sd_intercept * sd_slope
    intercepts, slopes = rng.multivariate_normal(
        [intercept, slope],
        [[sd_intercept**2, covariance], [covariance, sd_slope**2]],
        N_DRAW,
    ).T
    return intercepts[:, None] + slopes[:, None] * ph_e


def plot_gunn(ax, lines, ph_e, label):
    low, high = np.quantile(lines, QUANTILES, axis=0)
    ax.fill_between(ph_e, low, high, color=GUNN_COLOUR, alpha=0.25, lw=0)
    ax.plot(ph_e, np.median(lines, axis=0), color=GUNN_COLOUR, label=label)


def plot_ph_response(ratio_ax, volume_ax):
    sweep = read_sweep()
    joshi_palsson = read_joshi_palsson()
    rng = np.random.default_rng(0)
    ph_e = np.linspace(*GUNN_PH_RANGE, 100)
    gunn_label = "Gunn et al. (1973)"
    plot_gunn(
        ratio_ax,
        draw_gunn_lines(GUNN_DONNAN_RATIO, ph_e, rng),
        ph_e,
        gunn_label,
    )
    water = draw_gunn_lines(
        GUNN_WATER_CONTENT,
        np.append(ph_e, REFERENCE_PH),
        rng,
    )
    plot_gunn(volume_ax, water[:, :-1] / water[:, -1:], ph_e, gunn_label)
    for ax, quantity in (
        (ratio_ax, "donnan_ratio"),
        (volume_ax, "relative_volume"),
    ):
        ax.plot(
            *joshi_palsson[quantity],
            color=JOSHI_PALSSON_COLOUR,
            label="Joshi and Palsson (1990)",
        )
        for name, linestyle in VARIANT_STYLES.items():
            ax.plot(
                sweep[name]["ph_e"],
                sweep[name][quantity],
                color=MODEL_COLOUR,
                linestyle=linestyle,
                label=VARIANT_NAMES[name],
            )
        style(ax)
    ratio_ax.set_ylabel("Donnan ratio")
    volume_ax.set_ylabel(f"Volume, relative to pH {REFERENCE_PH}")
    volume_ax.set_xlabel("External pH")
    volume_ax.legend(frameon=False, loc="upper right")


def read_benchmark():
    runs = defaultdict(list)
    with open(BENCHMARK_FILE) as f:
        for row in csv.DictReader(f):
            if float(row["tolerance"]) != BENCHMARK_TOLERANCE:
                continue
            runs[row["formulation"], row["method"]].append(
                {
                    "ratio": float(row["ratio"]),
                    "seconds": float(row["seconds"]),
                    "success": row["success"] == "True",
                    "conc_error": float(row["conc_error"]),
                },
            )
    return runs


def plot_benchmark(time_ax, error_ax):
    runs = read_benchmark()
    stiff = runs["stiff", "bdf"]
    ratios = [run["ratio"] for run in stiff]
    for method, (_, colour) in METHODS.items():
        [rapid] = runs["rapid equilibrium", method]
        time_ax.axhline(
            rapid["seconds"],
            color=colour,
            linestyle=FORMULATIONS["rapid equilibrium"][1],
        )
        rows = runs["stiff", method]
        solved = [row for row in rows if row["success"]]
        failed = [row for row in rows if not row["success"]]
        time_ax.plot(
            [row["ratio"] for row in solved],
            [row["seconds"] for row in solved],
            color=colour,
            linestyle=FORMULATIONS["stiff"][1],
            marker="o",
            markersize=3,
        )
        time_ax.plot(
            [row["ratio"] for row in failed],
            [row["seconds"] for row in failed],
            color=colour,
            linestyle="none",
            marker="x",
            markersize=5,
        )
    handles = [
        Line2D([], [], color=colour, label=label)
        for label, colour in METHODS.values()
    ] + [
        Line2D([], [], color=MUTED, linestyle=linestyle, label=label)
        for label, linestyle in FORMULATIONS.values()
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=MUTED,
            linestyle="none",
            marker="x",
            label="No steady state",
        ),
    )
    solved = [run for run in stiff if run["success"]]
    error_ax.plot(
        [run["ratio"] for run in solved],
        [run["conc_error"] for run in solved],
        color=MODEL_COLOUR,
        marker="o",
        markersize=3,
    )
    guide = min(
        solved,
        key=lambda run: abs(np.log(run["ratio"] / ERROR_GUIDE_RATIO)),
    )
    guide_ratios = np.array([min(ratios), max(ratios)])
    error_ax.plot(
        guide_ratios,
        guide["conc_error"] * guide["ratio"] / guide_ratios,
        color=MUTED,
        linestyle=":",
        label="Proportional to 1 / ratio",
    )
    error_ax.legend(frameon=False)
    for ax in (time_ax, error_ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(ratios) / 2, max(ratios) * 2)
        style(ax)
    time_ax.set_ylabel("Time (s)")
    error_ax.set_ylabel("Relative error")
    error_ax.set_xlabel("Forward-to-net flux ratio")
    time_ax.set_ylim(top=TIME_AXIS_TOP)
    time_ax.legend(handles=handles, frameon=False, ncols=2, loc="upper left")


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
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.fontsize": 7,
        },
    )
    fig = plt.figure(figsize=(7.5, 8.7), layout="constrained")
    network, top, bottom = fig.subfigures(3, 1, height_ratios=[3.6, 3.1, 2.0])
    network_ax = network.subplots()
    network_texts = plot_network(network_ax)
    left, right = top.subfigures(1, 2)
    ratio_ax, volume_ax = left.subplots(2, 1, sharex=True)
    time_ax, error_ax = right.subplots(2, 1, sharex=True)
    plot_ph_response(ratio_ax, volume_ax)
    plot_benchmark(time_ax, error_ax)
    sensitivity_axes = bottom.subplots(1, 2, sharey=True)
    plot_sensitivities(sensitivity_axes)
    for ax, letter in (
        (network_ax, "A"),
        (ratio_ax, "B"),
        (time_ax, "C"),
        (sensitivity_axes[0], "D"),
    ):
        ax.set_title(letter, loc="left", fontweight="bold")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    draw_reactions(network_ax, network_texts)
    for suffix in ("png", "pdf"):
        path = RESULTS_DIR / f"figure.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
