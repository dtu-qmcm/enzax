"""Plot the glycolysis regulator-finding figure.

Panel A maps the pathway's reactions, cofactors and known regulation, with
PFKM's candidate inhibitors coloured by hypothesis. Panel B shows sedoheptulose
7-phosphate against external glucose, as predicted by the PEP and lactate
hypotheses' fits to realistic glucose (30 and 4.5 mM), with the fitted and
held-out measurements. Panel C shows the design screen after the fits to
realistic glucose for two pairs of hypotheses, PEP and lactate, and PEP and PEP
plus lactate, leaving out candidate glucose levels of 4.5 mM and above. Panel D
shows how well each hypothesis's fit to realistic glucose predicts each
held-out experiment, and all of them together, as log predictive density
relative to the best hypothesis. Panel E shows how five concentrations change
under DCA x3 for the same two pairs, comparing the predictions from the fits to
realistic glucose, the estimates from the fits to all experiments and the
measurements. Saves results/figure.png and results/figure.pdf.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analyse import get_derived_file, get_pointwise_predictive_density
from common import (
    COLOURS,
    DESIGN,
    HELD_OUT,
    HIGH_GLUCOSE,
    HYPOTHESES,
    MEASUREMENTS_FILE,
    MODEL,
    REFERENCE_EXPERIMENT,
    RESULTS_DIR,
    SERIES,
)
from design import GLUCOSE_MM, REFERENCE_GLUCOSE_MM, get_design_file
from network import draw_reactions, plot_network

HYPOTHESIS_NAMES = {
    "none": "No effector",
    "pep": "PEP",
    "lac": "Lactate",
    "both": "PEP and lactate",
}
FIRST_FIT = "high_glucose"
FULL_FIT = "series_dca"
CONTRASTS = (("pep", "lac"), ("pep", "both"))
HELD_OUT_GROUPS = {
    "Glucose\n1.5 mM": ("glc_1.5",),
    "Glucose\n0.45 mM": ("glc_0.45",),
    "DCA x3": ("dca",),
    "All held\nout": HELD_OUT[FIRST_FIT],
}
SCREEN_HIDDEN = ("Glucose 30 mM", "Glucose 10 mM", "Glucose 4.5 mM")
RESPONSE_SPECIES = "s7p_c"
RESPONSE_TICKS_MM = (0.15, 0.2, 0.3)
DCA_SPECIES = {
    "f6p_c": "F6P",
    "fdp_c": "FBP",
    "pep_c": "PEP",
    "pyr_c": "Pyruvate",
    "lac_c": "Lactate",
}
DCA_CANDIDATE = "DCA x3"
CONTROL_CANDIDATE = f"Glucose {REFERENCE_GLUCOSE_MM:g} mM"
BEFORE_ALPHA = 0.35
TRAINING_COLOUR = "#efeeeb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
HEATMAP = "Blues"
DODGE = 0.08
DCA_OFFSET = 0.15
M_TO_MM = 1000


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def get_glucose_mm(experiment):
    return REFERENCE_GLUCOSE_MM * DESIGN[experiment].get("glc_e", 1.0)


def load_measured(species, experiments):
    with MEASUREMENTS_FILE.open() as f:
        return {
            row["experiment"]: float(row["value"])
            for row in csv.DictReader(f)
            if row["kind"] == "conc"
            and row["id"] == species
            and row["experiment"] in experiments
        }


def plot_glucose_response(ax, design):
    glucose = np.array(GLUCOSE_MM)
    candidates = list(design["candidates"])
    rows = [candidates.index(f"Glucose {mm:g} mM") for mm in glucose]
    column = list(design["observables"]).index(RESPONSE_SPECIES)
    ax.axvspan(
        min(map(get_glucose_mm, HIGH_GLUCOSE)),
        max(map(get_glucose_mm, HIGH_GLUCOSE)),
        color=TRAINING_COLOUR,
        linewidth=0,
    )
    for hypothesis in CONTRASTS[0]:
        conc = M_TO_MM * np.exp(
            design[f"predictions|{hypothesis}"][:, rows, column],
        )
        low, median, high = np.nanquantile(conc, [0.05, 0.5, 0.95], axis=0)
        ax.fill_between(
            glucose,
            low,
            high,
            color=COLOURS[hypothesis],
            alpha=0.2,
            linewidth=0,
        )
        ax.plot(
            glucose,
            median,
            color=COLOURS[hypothesis],
            linewidth=2,
            label=f"{HYPOTHESIS_NAMES[hypothesis]}, first fit",
        )
    measured = load_measured(RESPONSE_SPECIES, SERIES)
    held_out = tuple(e for e in SERIES if e not in HIGH_GLUCOSE)
    for experiments, fill, name in (
        (HIGH_GLUCOSE, TEXT, "fitted"),
        (held_out, "white", "held out"),
    ):
        ax.plot(
            [get_glucose_mm(e) for e in experiments],
            [M_TO_MM * measured[e] for e in experiments],
            "o",
            markersize=5,
            markerfacecolor=fill,
            markeredgecolor=TEXT,
            label=f"Measured, {name}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(glucose, [f"{mm:g}" for mm in glucose])
    ax.set_yticks(RESPONSE_TICKS_MM, [f"{y:g}" for y in RESPONSE_TICKS_MM])
    ax.minorticks_off()
    ax.set_xlabel("External glucose (mM)")
    ax.set_ylabel("Sedoheptulose 7-phosphate (mM)")
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=6,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    style(ax)


def plot_discrimination(ax, design):
    candidates = list(design["candidates"])
    shown = [c for c in candidates if c not in SCREEN_HIDDEN]
    pairs = list(design["pairs"])
    discrimination = design["discrimination"][
        np.ix_(
            [candidates.index(c) for c in shown],
            [pairs.index(f"{a}-{b}") for a, b in CONTRASTS],
        )
    ]
    ax.imshow(
        np.log10(1 + discrimination),
        cmap=HEATMAP,
        aspect="auto",
        vmin=0,
    )
    for (i, j), value in np.ndenumerate(discrimination):
        ax.text(
            j,
            i,
            f"{value:.0f}" if value >= 1 else f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white"
            if np.log10(1 + value) > 0.6 * np.log10(1 + discrimination.max())
            else TEXT,
        )
    ax.set_xticks(
        range(len(CONTRASTS)),
        [
            f"{HYPOTHESIS_NAMES[a]} vs\n{HYPOTHESIS_NAMES[b]}"
            for a, b in CONTRASTS
        ],
        fontsize=7,
    )
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(shown)), shown)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def plot_held_out_score(ax):
    pointwise = {}
    for hypothesis in HYPOTHESES:
        derived = np.load(get_derived_file(f"{hypothesis}_{FIRST_FIT}"))
        pointwise[hypothesis] = get_pointwise_predictive_density(
            derived["held_out_log_likelihood"],
        )[0]
        experiments = np.array(HELD_OUT[FIRST_FIT])[
            derived["held_out_experiment"][0]
        ]
    positions = np.arange(len(HELD_OUT_GROUPS))
    scores = {h: [] for h in HYPOTHESES}
    errors = {h: [] for h in HYPOTHESES}
    for group in HELD_OUT_GROUPS.values():
        mask = np.isin(experiments, group)
        best = max(HYPOTHESES, key=lambda h: pointwise[h][mask].sum())
        for hypothesis in HYPOTHESES:
            difference = pointwise[hypothesis][mask] - pointwise[best][mask]
            scores[hypothesis].append(difference.sum())
            errors[hypothesis].append(
                np.sqrt(len(difference) * difference.var()),
            )
    for offset, hypothesis in zip(
        np.linspace(-1.5, 1.5, len(HYPOTHESES)) * DODGE,
        HYPOTHESES,
        strict=True,
    ):
        ax.errorbar(
            positions + offset,
            scores[hypothesis],
            yerr=errors[hypothesis],
            color=COLOURS[hypothesis],
            linestyle="none",
            elinewidth=2,
            marker="o",
            markersize=6,
            capsize=0,
            label=HYPOTHESIS_NAMES[hypothesis],
        )
    ax.set_xticks(positions, list(HELD_OUT_GROUPS))
    ax.set_xlim(-0.5, positions[-1] + 0.5)
    ax.set_ylabel(
        "Log predictive density of held-out\ndata, relative to the best",
    )
    ax.legend(frameon=False, loc="lower left", fontsize=6, handlelength=1.2)
    style(ax)
    ax.grid(axis="x", visible=False)


def get_percent_change(log_ratio):
    return 100 * (np.exp(log_ratio) - 1)


def get_dca_changes(design, hypothesis):
    candidates = list(design["candidates"])
    observables = list(design["observables"])
    predictions = design[f"predictions|{hypothesis}"]
    dca, control = (
        predictions[:, candidates.index(candidate)]
        for candidate in (DCA_CANDIDATE, CONTROL_CANDIDATE)
    )
    derived = np.load(get_derived_file(f"{hypothesis}_{FULL_FIT}"))
    before, after = {}, {}
    for species in DCA_SPECIES:
        observable = observables.index(species)
        before[species] = get_percent_change(
            dca[:, observable] - control[:, observable],
        )
        position = MODEL.species.index(species)
        after[species] = get_percent_change(
            np.log(derived["conc|dca"][:, position])
            - np.log(derived[f"conc|{REFERENCE_EXPERIMENT}"][:, position]),
        )
    return before, after


def plot_dca_response(axes, design):
    rows = np.arange(len(DCA_SPECIES))[::-1]
    measured = [
        load_measured(species, (REFERENCE_EXPERIMENT, "dca"))
        for species in DCA_SPECIES
    ]
    for ax, contrast in zip(axes, CONTRASTS, strict=True):
        offsets = np.linspace(DCA_OFFSET, -DCA_OFFSET, len(contrast))
        for offset, hypothesis in zip(offsets, contrast, strict=True):
            for changes, alpha in zip(
                get_dca_changes(design, hypothesis),
                (BEFORE_ALPHA, 1.0),
                strict=True,
            ):
                for row, species in zip(rows, DCA_SPECIES, strict=True):
                    low, median, high = np.quantile(
                        changes[species],
                        [0.05, 0.5, 0.95],
                    )
                    ax.plot(
                        [low, high],
                        [row + offset] * 2,
                        color=COLOURS[hypothesis],
                        alpha=alpha,
                        linewidth=2,
                        solid_capstyle="round",
                    )
                    ax.plot(
                        median,
                        row + offset,
                        "o",
                        color=COLOURS[hypothesis],
                        alpha=alpha,
                        markersize=5,
                    )
        ax.plot(
            [
                get_percent_change(np.log(m["dca"] / m[REFERENCE_EXPERIMENT]))
                for m in measured
            ],
            rows,
            "D",
            color=TEXT,
            markersize=4,
            zorder=3,
        )
        a, b = contrast
        ax.set_title(
            f"{HYPOTHESIS_NAMES[a]} vs\n{HYPOTHESIS_NAMES[b]}",
            fontsize=7,
            color=MUTED,
        )
        ax.axvline(0, color=MUTED, linewidth=0.6)
        ax.set_yticks(rows, list(DCA_SPECIES.values()))
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Change under DCA x3 (%)")
        style(ax)
        ax.grid(axis="y", visible=False)
    hypotheses = list(dict.fromkeys(h for c in CONTRASTS for h in c))
    handles = [
        Line2D([], [], color=COLOURS[h], marker="o", linewidth=2)
        for h in hypotheses
    ] + [
        Line2D([], [], color=MUTED, marker="o", alpha=BEFORE_ALPHA),
        Line2D([], [], color=MUTED, marker="o"),
        Line2D([], [], color=TEXT, marker="D", linestyle="none", markersize=4),
    ]
    axes[0].figure.legend(
        handles,
        [
            *(HYPOTHESIS_NAMES[h] for h in hypotheses),
            "Predicted (first fit)",
            "Estimated (fit to all)",
            "Measured",
        ],
        frameon=False,
        loc="outside lower center",
        ncols=3,
        fontsize=6,
        handlelength=1.2,
    )


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
    design = np.load(get_design_file(FIRST_FIT))
    fig = plt.figure(figsize=(7.5, 8.7), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[5.4, 3.3])
    network, side = top.subfigures(1, 2, width_ratios=[4.5, 3.0])
    network_ax = network.subplots()
    network_texts = plot_network(network_ax)
    response_ax, discrimination_ax = side.subplots(
        2,
        1,
        height_ratios=[1.0, 0.8],
    )
    score_fig, dca_fig = bottom.subfigures(1, 2, width_ratios=[1.0, 1.2])
    score_ax = score_fig.subplots()
    dca_axes = dca_fig.subplots(1, 2, sharey=True)
    plot_glucose_response(response_ax, design)
    plot_discrimination(discrimination_ax, design)
    plot_held_out_score(score_ax)
    plot_dca_response(dca_axes, design)
    for ax, letter in zip(
        [network_ax, response_ax, discrimination_ax, score_ax, dca_axes[0]],
        "ABCDE",
        strict=True,
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
