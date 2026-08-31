"""The four Michaelis Menten rate laws.

Each is a `SaturableRateEquation` with `reversible` and `allosteric` set, and
nothing else: the machinery lives in `enzax.rate_equations.saturable` and the
fields are documented there.
"""

from enzax.rate_equations.saturable import SaturableRateEquation


class IrreversibleMichaelisMenten(SaturableRateEquation):
    """A reaction with irreversible Michaelis Menten kinetics.

    Its rate has no thermodynamic driving force, so it is proportional to the
    fraction of enzyme bound to every substrate and nothing else.
    """

    reversible: bool = False


class ReversibleMichaelisMenten(SaturableRateEquation):
    """A reaction with reversible Michaelis Menten kinetics.

    Its rate is that of an irreversible reaction times a driving force
    computed from the reactants' formation energies and concentrations.
    """


class AllostericIrreversibleMichaelisMenten(SaturableRateEquation):
    """Irreversible Michaelis Menten kinetics with a Monod Wyman Changeux term.

    Declare the effectors with `allosteric_inhibitors` and
    `allosteric_activators`, and the number of subunits with `subunits`.
    """

    reversible: bool = False
    allosteric: bool = True


class AllostericReversibleMichaelisMenten(SaturableRateEquation):
    """Reversible Michaelis Menten kinetics with a Monod Wyman Changeux term.

    Declare the effectors with `allosteric_inhibitors` and
    `allosteric_activators`, and the number of subunits with `subunits`.
    """

    allosteric: bool = True
