from typing import Protocol, Any
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar, ScalarLike


class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " m n"]
    rate_equation_classes: list[Any]
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


class RateEquation(Protocol):

    def __init__(
        self,
        global_parameters: KineticModelParameters,
        global_structure: KineticModelStructure,
        ix: int,
    ): ...

    def __call__(self, conc: Float[Array, " n"]) -> Scalar: ...


class KineticModel(eqx.Module):
    parameters: KineticModelParameters
    structure: KineticModelStructure
    rate_equations: list[RateEquation]

    def __init__(self, parameters, structure):
        self.parameters = parameters
        self.structure = structure
        self.rate_equations = [
            cls(self.parameters, self.structure, ix)
            for ix, cls in enumerate(self.structure.rate_equation_classes)
        ]

    def __call__(self, conc_balanced: Float[Array, " m"]) -> Float[Array, " n"]:
        conc = jnp.zeros(self.structure.S.shape[0])
        conc = conc.at[self.structure.ix_balanced].set(conc_balanced)
        conc = conc.at[self.structure.ix_unbalanced].set(
            jnp.exp(self.parameters.log_conc_unbalanced)
        )
        return jnp.array(
            [
                f(conc[self.structure.ix_reactant[r]])
                for r, f in enumerate(self.rate_equations)
            ]
        )


# @eqx.filter_jit
def dcdt(
    t: ScalarLike, conc: Float[Array, " n_balanced"], args: KineticModel
) -> Float[Array, " n_balanced"]:
    model = args
    return (model.structure.S @ model(conc))[model.structure.ix_balanced]
