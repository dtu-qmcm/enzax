from typing import Type
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar

from enzax.rate_equations import (
    RateEquation,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " m n"]
    rate_equation_classes: list[Type[RateEquation]]
    ix_balanced: Int[Array, " n_balanced"]
    ix_unbalanced: Int[Array, " n_unbalanced"]
    ix_substrate: Int[Array, " n _"]
    ix_product: Int[Array, " n _"]
    ix_reactant: Int[Array, " n _"]
    ix_rate_to_km: Int[Array, " n _"]
    ix_mic_to_metabolite: Int[Array, " m"]
    stoich_by_rate: Float[Array, "n _"]


class KineticModelParameters(eqx.Module):
    """Parameters for a kinetic model."""

    log_kcat: Float[Array, " n"]
    log_enzyme: Float[Array, " n"]
    dgf: Float[Array, " n_metabolite"]
    log_km: Float[Array, " n_km"]
    log_conc_unbalanced: Float[Array, " n_unbalanced"]
    temperature: Scalar


class KineticModel(eqx.Module):
    parameters: KineticModelParameters
    structure: KineticModelStructure
    rate_equations: list[RateEquation]

    def __init__(self, parameters, structure):
        self.parameters = parameters
        self.structure = structure
        rate_equations = []
        for r, rec in enumerate(self.structure.rate_equation_classes):
            ix_km = structure.ix_rate_to_km[r]
            ix_dgf = structure.ix_mic_to_metabolite[structure.ix_reactant[r]]
            if rec == ReversibleMichaelisMenten:
                re = ReversibleMichaelisMenten(
                    dgf=parameters.dgf[ix_dgf],
                    log_km=parameters.log_km[ix_km],
                    log_enzyme=parameters.log_enzyme[r],
                    log_kcat=parameters.log_kcat[r],
                    temperature=parameters.temperature,
                    stoich=structure.stoich_by_rate[r],
                    ix_substrate=structure.ix_substrate[r],
                    ix_product=structure.ix_product[r],
                )
                rate_equations.append(re)
            elif rec == IrreversibleMichaelisMenten:
                re = IrreversibleMichaelisMenten(
                    dgf=parameters.dgf[ix_dgf],
                    log_km=parameters.log_km[ix_km],
                    log_enzyme=parameters.log_enzyme[r],
                    log_kcat=parameters.log_kcat[r],
                    temperature=parameters.temperature,
                    stoich=structure.stoich_by_rate[r],
                    ix_substrate=structure.ix_substrate[r],
                )
                rate_equations.append(re)
        self.rate_equations = rate_equations


# @eqx.filter_jit
def get_flux(
    conc_balanced: Float[Array, " m"],
    model: KineticModel,
) -> Float[Array, " n"]:
    structure = model.structure
    conc = jnp.zeros(structure.S.shape[0])
    conc = conc.at[structure.ix_balanced].set(conc_balanced)
    conc = conc.at[structure.ix_unbalanced].set(
        jnp.exp(model.parameters.log_conc_unbalanced)
    )
    return jnp.array(
        [f(conc[structure.ix_reactant[r]]) for r, f in enumerate(model.rate_equations)]
    )


# @eqx.filter_jit
def dcdt(
    t: Scalar, conc: Float[Array, " n_balanced"], args: KineticModel
) -> Float[Array, " n_balanced"]:
    model = args
    v = get_flux(conc, model)
    return (model.structure.S @ v)[model.structure.ix_balanced]
