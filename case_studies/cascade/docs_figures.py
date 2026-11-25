"""Plot a fitted time course for the documentation.

Plots GMP, ATP, GTP and GDP-fucose in PLOT_EXPERIMENT against its
measurements, at one or more sets of free parameters on shared axes.
mode.py uses it to save the plots at the prior mean and at the mode.
"""

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from common import get_reading, get_time_course
from enzax.examples import gdp_fucose

PLOT_EXPERIMENT = "FE18_07FKP"
PLOT_SPECIES = ["gmp", "atp", "gtp", "gdp_fucose"]
COLOUR = "#2a78d6"
GRID = "#e4e3df"
INK = "#52514e"


def get_measured(times, measurements, experiment, species):
    """Get an experiment's measured times and values for one species."""
    measurement = next(m for m in measurements if m.experiment == experiment)
    rows = [
        (times[experiment][id_.split("|")[1]], float(value))
        for id_, value in zip(measurement.ids, measurement.values)
        if id_.split("|")[0] == species
    ]
    return zip(*rows)


def plot_time_course(conc, ts, times, measurements, title, y_max):
    """Plot PLOT_EXPERIMENT's time course against its measurements."""
    fig, axes = plt.subplots(
        1,
        len(PLOT_SPECIES),
        figsize=(11, 3),
        sharex=True,
    )
    for ax, species in zip(axes, PLOT_SPECIES):
        ax.plot(
            ts,
            conc[:, gdp_fucose.MEASURED_SPECIES.index(species)],
            color=COLOUR,
            linewidth=2,
            solid_capstyle="round",
            label="model",
        )
        measured_ts, measured = get_measured(
            times,
            measurements,
            PLOT_EXPERIMENT,
            species,
        )
        ax.plot(
            measured_ts,
            measured,
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=COLOUR,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="measured",
        )
        ax.set_title(species, color=INK, fontsize=11)
        ax.set_ylim(0, y_max[species])
        ax.set_xlabel("time (h)", color=INK)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("concentration (mM)", color=INK)
    axes[0].legend(frameon=False, labelcolor=INK)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    return fig


def plot_fit(
    layout,
    split,
    times,
    measurements,
    free_by_label,
    reading_by_label,
):
    """Plot PLOT_EXPERIMENT at each set of free parameters, on shared axes.

    free_by_label maps a title, such as "MAP", to a set of free parameters,
    and reading_by_label maps the same titles to the fraction of GDP-fucose
    read as GMP and the bias of PLOT_EXPERIMENT's measurements, so that the
    curves show what the instrument would read.
    """
    ts = jnp.linspace(0.0, max(times[PLOT_EXPERIMENT].values()), 200)
    measured = [
        gdp_fucose.model.species.index(species)
        for species in gdp_fucose.MEASURED_SPECIES
    ]
    conc_by_label = {
        label: get_reading(
            np.asarray(
                get_time_course(layout, split, free, PLOT_EXPERIMENT, ts),
            )[:, measured],
            *reading_by_label[label],
        )
        for label, free in free_by_label.items()
    }
    y_max = {}
    for species in PLOT_SPECIES:
        i = gdp_fucose.MEASURED_SPECIES.index(species)
        _, measured = get_measured(
            times,
            measurements,
            PLOT_EXPERIMENT,
            species,
        )
        y_max[species] = 1.1 * max(
            *(float(conc[:, i].max()) for conc in conc_by_label.values()),
            *measured,
        )
    return {
        label: plot_time_course(
            conc,
            ts,
            times,
            measurements,
            f"{PLOT_EXPERIMENT} at the {label}",
            y_max,
        )
        for label, conc in conc_by_label.items()
    }
