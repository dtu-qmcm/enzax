"""Plot the performance figure.

Panel A shows time per gradient and gradient error against tolerance for the
three AMICI benchmark problems, from plot.py. Panel B shows the optimisation
ladder on the problem that AMICI can also run, from the summary that
optimisation_ladder.py saves. Saves results/figure.png and results/figure.pdf.
"""

import matplotlib.pyplot as plt

from optimisation_ladder import plot_ladder, read_summary
from plot import RC_PARAMS, plot_work_precision
from work_precision import RESULTS

AMICI_SUFFIX = "_amici"


def main():
    plt.rcParams.update(RC_PARAMS)
    fig = plt.figure(figsize=(7.5, 7.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[4.8, 2.8])
    plot_work_precision(top, RESULTS)
    plot_ladder(bottom.subplots(), read_summary(AMICI_SUFFIX))
    for subfigure, letter in ((top, "A"), (bottom, "B")):
        subfigure.text(
            0,
            1,
            letter,
            fontsize="large",
            fontweight="bold",
            ha="left",
            va="top",
        )
    for suffix in ("png", "pdf"):
        path = RESULTS / f"figure.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
