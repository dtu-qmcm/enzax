"""Run the glycolysis case study from start to finish.

Simulates the measurements, then for each fit in common.FITS finds the
posterior mode, samples the posterior and computes derived quantities, and
finally plots the figure. One fit uses all three experiments; the other
leaves out the high-lactate experiment, to show what it adds. Sampling
needs four free cores, and the whole run takes about an hour and a half.
"""

import analyse
import figure
import fit
import mode
import simulate
from common import FITS


def main():
    simulate.main()
    for name in FITS:
        mode.main(name)
        fit.main(name)
        analyse.main(name)
    figure.main()


if __name__ == "__main__":
    main()
