from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar

from enzax.rate_equations import RateEquation, ReversibleMichaelisMenten


class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " m n"]
    ix_balanced: Int[Array, " n_balanced"]
    ix_unbalanced: Int[Array, " n_unbalanced"]
    ix_substrate: Int[Array, " n _"]
    ix_product: Int[Array, " n _"]
    ix_reactant: Int[Array, " n _"]
    ix_reactant_to_km: Int[Array, " n _"]
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
        self.rate_equations = [
            ReversibleMichaelisMenten(self.parameters, self.structure, r)
            for r in range(self.structure.S.shape[1])
        ]


@eqx.filter_jit
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


@eqx.filter_jit
def dcdt(
    t: Scalar, conc: Float[Array, " n_balanced"], args: KineticModel
) -> Float[Array, " n_balanced"]:
    model = args
    v = get_flux(conc, model)
    return (model.structure.S @ v)[model.structure.ix_balanced]
