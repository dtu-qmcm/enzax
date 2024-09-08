"""Module with parameters and parameter sets.

These are not required, but they do provide handy shape checking, for example to ensure that your parameter set has the same number of enzymes and kcat parameters.

"""  # Noqa: E501

import equinox as eqx
from jaxtyping import Array, Float, Scalar, jaxtyped
from typeguard import typechecked

LogKcat = Float[Array, " n_enzyme"]
LogEnzyme = Float[Array, " n_enzyme"]
Dgf = Float[Array, " n_metabolite"]
LogKm = Float[Array, " n_km"]
LogKi = Float[Array, " n_ki"]
LogConcUnbalanced = Float[Array, " n_unbalanced"]
LogDrain = Float[Array, " n_drain"]
LogTransferConstant = Float[Array, " n_allosteric_enzyme"]
LogDissociationConstant = Float[Array, " n_allosteric_effect"]


@jaxtyped(typechecker=typechecked)
class MichaelisMentenParameterSet(eqx.Module):
    """Parameters for a model with Michaelis Menten kinetics.

    This kind of parameter set supports models with the following rate laws:

      - enzax.rate_equations.drain.Drain
      - enzax.rate_equations.michaelis_menten.IrreversibleMichaelisMenten
      - enzax.rate_equations.michaelis_menten.ReversibleMichaelisMenten

    """

    log_kcat: LogKcat
    log_enzyme: LogEnzyme
    dgf: Dgf
    log_km: LogKm
    log_ki: LogKi
    log_conc_unbalanced: LogConcUnbalanced
    temperature: Scalar
    log_drain: LogDrain


class AllostericMichaelisMentenParameterSet(MichaelisMentenParameterSet):
    """Parameters for a model with Michaelis Menten kinetics, with allostery.

    Reactions can be any out of:

      - drain.Drain
      - michaelis_menten.IrreversibleMichaelisMenten
      - michaelis_menten.ReversibleMichaelisMenten
      - generalised_mwc.AllostericIrreversibleMichaelisMenten
      - generalised_mwc.AllostericReversibleMichaelisMenten

    """

    log_transfer_constant: LogTransferConstant
    log_dissociation_constant: LogDissociationConstant
