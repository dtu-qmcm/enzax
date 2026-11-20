"""TODO(TG): module docstring for the regulator-finding design."""

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
