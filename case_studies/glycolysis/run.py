"""Run the glycolysis regulator-finding case study from start to finish.

Simulates the measurements, then for each fit in common.FITS finds the
posterior mode, samples the posterior and computes derived quantities. Then
screens candidate experiments after the fit to the realistic glucose levels
and plots the figure. There are eight fits, one per hypothesis for each of the
two sets of experiments. Sampling needs four free cores per fit.
"""

import analyse
import design
import figure
import fit
import mode
import simulate
from common import FITS


def main():
    simulate.main()
    for name in FITS:
        mode.main(name)
    fit.main(list(FITS))
    for name in FITS:
        analyse.main(name)
    for phase in design.SCREENED_PHASES:
        design.main(phase)
    figure.main()


if __name__ == "__main__":
    main()
