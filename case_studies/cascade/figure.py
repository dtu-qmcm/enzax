"""Plot the cascade figure.

Panel A maps the network, colouring GMPK's rapid equilibrium, the reaction
added to Huber et al.'s model and the postulated hydrolase. Panel B shows the
fitting runs and panel C the validation runs: medians and 90% posterior
predictive bands of the readings from predict.py, with the median of Huber et
al.'s parameter ensemble from huber.py dashed. Panel D compares the baseline
design, Huber et al.'s O3 design and the optimum from optimise.py: their
GDP-fucose at 24 h, with the probability of reaching the target above each,
their enzyme mass, and their enzyme levels within Huber et al.'s bounds and
the range the runs tested. Saves results/figure.png and results/figure.pdf.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D

from common import FITTING_EXPERIMENTS, HUBER_FILE, RESULTS_DIR
from enzax.examples import gdp_fucose
from network import draw_reactions, plot_network
from optimise import DESIGNS_FILE, LOWER, UPPER, get_mass
from predict import PREDICTIONS_FILE

MODEL_COLOUR = "#2a78d6"
VALIDATION_COLOUR = "#eb6834"
MEASURED_COLOUR = "#0b0b0b"
TEXT = "#0b0b0b"
MUTED = "#52514e"
HUBER_COLOUR = MUTED
GRID = "#e4e3df"
TESTED = "#c3c2b7"
BAND = (5, 95)
TITRE_QUANTILES = (10, 90)
SPECIES_NAMES = {
    "amp": "AMP",
    "adp": "ADP",
    "atp": "ATP",
    "gmp": "GMP",
    "gdp": "GDP",
    "gtp": "GTP",
    "gdp_fucose": "GDP-fucose",
}
EXPERIMENT_NAMES = {
    "FE17_1FKP": "FE17\n1 FKP",
    "FE17_07FKP": "FE17\n0.7 FKP",
    "FE18_1FKP": "FE18\n1 FKP",
    "FE18_07FKP": "FE18\n0.7 FKP",
    "FE20_Validation1": "Validation 1",
    "FE20_Validation2": "Validation 2",
}
DESIGN_NAMES = {
    "baseline": "Baseline",
    "huber": "Huber et al.",
    "optimum": "Optimum",
}
DESIGN_MARKERS = {"baseline": "o", "huber": "s", "optimum": "D"}
ENZYME_NAMES = ["FKP", "PPA", "PPK3"]


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def get_measured(experiments):
    times, measurements = gdp_fucose.load_measurements(experiments)
    measured = {}
    for measurement in measurements:
        grid = times[measurement.experiment]
        for id_, value in zip(measurement.ids, measurement.values):
            species, label = id_.split("|")
            measured.setdefault((measurement.experiment, species), []).append(
                (grid[label], float(value)),
            )
    return measured


def plot_time_courses(axes):
    predictions = np.load(PREDICTIONS_FILE)
    huber = np.load(HUBER_FILE)
    huber_species = list(huber["species"])
    species = list(predictions["species"])
    experiments = FITTING_EXPERIMENTS + gdp_fucose.VALIDATION_EXPERIMENTS
    measured = get_measured(experiments)
    for column, experiment in enumerate(experiments):
        validation = experiment in gdp_fucose.VALIDATION_EXPERIMENTS
        colour = VALIDATION_COLOUR if validation else MODEL_COLOUR
        ts = predictions[f"{experiment}_time"]
        courses = predictions[experiment]
        for row, name in enumerate(species):
            ax = axes[row, column]
            style(ax)
            course = courses[:, :, species.index(name)]
            low, high = np.nanpercentile(course, BAND, axis=0)
            ax.fill_between(ts, low, high, color=colour, alpha=0.25, lw=0)
            ax.plot(ts, np.nanmedian(course, axis=0), color=colour, lw=0.8)
            if name in huber_species:
                ax.plot(
                    huber[f"{experiment}_time"],
                    np.median(
                        huber[experiment][:, :, huber_species.index(name)],
                        axis=0,
                    ),
                    color=HUBER_COLOUR,
                    linestyle="--",
                    lw=0.8,
                )
            points = measured.get((experiment, name))
            if points:
                t, value = zip(*points)
                ax.plot(
                    t,
                    value,
                    linestyle="none",
                    marker="o",
                    markersize=1.8,
                    color=MEASURED_COLOUR,
                )
            if row == 0:
                ax.set_title(EXPERIMENT_NAMES[experiment], fontsize=6)
            if column == 0:
                ax.set_ylabel(f"{SPECIES_NAMES[name]}\n(mM)", fontsize=6)
            if row == len(species) - 1:
                ax.set_xlabel("Time (h)", fontsize=6)
            ax.tick_params(labelsize=5, length=2)


def plot_titres(ax):
    designs = np.load(DESIGNS_FILE)
    names = list(DESIGN_NAMES)
    titres = [designs[f"{name}_titre"] for name in names]
    titres = [t[np.isfinite(t)] for t in titres]
    positions = range(1, len(names) + 1)
    low, high = zip(*(np.percentile(t, TITRE_QUANTILES) for t in titres))
    ax.vlines(positions, low, high, color=MODEL_COLOUR, linewidth=1.2)
    ax.scatter(
        positions,
        [np.median(t) for t in titres],
        color=MODEL_COLOUR,
        s=8,
        zorder=3,
    )
    ax.axhline(designs["target"], color=MUTED, linewidth=0.6, linestyle=":")
    ax.text(
        0.55,
        designs["target"],
        "Target",
        fontsize=5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    for position, t, top in zip(positions, titres, high):
        ax.text(
            position,
            top,
            f"{np.mean(t >= designs['target']):.0%}",
            fontsize=5,
            color=MUTED,
            ha="center",
            va="bottom",
        )
    label_designs(ax, names)
    ax.set_ylabel("GDP-fucose at 24 h (mM)")
    style(ax)


def label_designs(ax, names):
    ax.set_xlim(0.5, len(names) + 0.5)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(
        [DESIGN_NAMES[name] for name in names],
        fontsize=6,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )


def plot_masses(ax):
    designs = np.load(DESIGNS_FILE)
    names = list(DESIGN_NAMES)
    positions = np.arange(1, len(names) + 1)
    masses = [get_mass(designs[f"{name}_design"]) for name in names]
    ax.bar(
        positions,
        masses,
        width=0.5,
        color=[MODEL_COLOUR if name == "optimum" else MUTED for name in names],
    )
    label_designs(ax, names)
    ax.set_ylabel("Enzyme (g/l)")
    style(ax)


def plot_designs(ax):
    designs = np.load(DESIGNS_FILE)
    positions = np.arange(len(ENZYME_NAMES))
    offsets = dict(zip(DESIGN_NAMES, (-0.15, 0.0, 0.15)))
    ax.bar(
        positions,
        UPPER - LOWER,
        bottom=LOWER,
        width=0.5,
        color=GRID,
        label="Bounds",
    )
    levels = np.array(
        [
            gdp_fucose.ENZYME_LEVELS[e][: len(ENZYME_NAMES)]
            for e in EXPERIMENT_NAMES
        ],
    )
    ax.bar(
        positions,
        levels.max(axis=0) - levels.min(axis=0),
        bottom=levels.min(axis=0),
        width=0.5,
        color=TESTED,
        label="Tested",
    )
    for name, offset in offsets.items():
        ax.plot(
            positions + offset,
            designs[f"{name}_design"],
            linestyle="none",
            marker=DESIGN_MARKERS[name],
            markersize=3.5,
            color=MODEL_COLOUR if name == "optimum" else MUTED,
            label=DESIGN_NAMES[name],
        )
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(ENZYME_NAMES, fontsize=6)
    ax.set_ylabel("Enzyme (mM)")
    ax.legend(
        fontsize=5,
        frameon=False,
        handlelength=1,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
    )
    style(ax)


def main():
    plt.rcParams.update(
        {
            "font.size": 7,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
        },
    )
    fig = plt.figure(figsize=(7.5, 8.8), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.1, 6.7])
    network_fig, design_fig = top.subfigures(1, 2, width_ratios=[3.4, 4.1])
    network_ax = network_fig.subplots()
    titre_ax, mass_ax, design_ax = design_fig.subplots(
        1,
        3,
        width_ratios=[1.2, 1.0, 1.1],
    )
    n_species = len(gdp_fucose.MEASURED_SPECIES)
    grid = bottom.subplots(
        n_species,
        7,
        width_ratios=[1, 1, 1, 1, 0.15, 1, 1],
        sharex="col",
        sharey="row",
    )
    for ax in grid[:, 4]:
        ax.set_visible(False)
    axes = np.delete(grid, 4, axis=1)
    texts = plot_network(network_ax)
    plot_time_courses(axes)
    bottom.legend(
        handles=[
            Line2D([], [], color=MODEL_COLOUR, label="Fitted"),
            Line2D([], [], color=VALIDATION_COLOUR, label="Predicted"),
            Line2D(
                [],
                [],
                color=HUBER_COLOUR,
                linestyle="--",
                label="Huber et al.",
            ),
            Line2D(
                [],
                [],
                color=MEASURED_COLOUR,
                linestyle="none",
                marker="o",
                markersize=1.8,
                label="Measured",
            ),
        ],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=6,
    )
    plot_titres(titre_ax)
    plot_masses(mass_ax)
    plot_designs(design_ax)
    for ax, letter in (
        (network_ax, "A"),
        (axes[0, 0], "B"),
        (axes[0, len(FITTING_EXPERIMENTS)], "C"),
        (titre_ax, "D"),
    ):
        ax.text(
            -0.02,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
    fig.canvas.draw()
    fig.set_layout_engine("none")
    draw_reactions(network_ax, texts)
    for suffix in ("png", "pdf"):
        path = RESULTS_DIR / f"figure.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
