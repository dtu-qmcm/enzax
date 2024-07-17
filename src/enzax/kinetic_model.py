from functools import partial
import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar

from enzax.rate_equations import reversible_michaelis_menten


@chex.dataclass
class KineticModelParameters:
    """Parameters for a kinetic model."""

    log_kcat: Float[Array, " n"]
    log_enzyme: Float[Array, " n"]
    dgf: Float[Array, " n_metabolite"]
    log_km: Float[Array, " n_km"]
    log_conc_unbalanced: Float[Array, " n_unbalanced"]


@chex.dataclass
class KineticModel:
    S: Float[Array, " m"]
    temperature: Scalar
    ix_reactant_to_km: Int[Array, " n m"]
    ix_balanced: Int[Array, " n_balanced"]
    ix_unbalanced: Int[Array, " n_unbalanced"]
    ix_substrate: Int[Array, " n"]
    ix_product: Int[Array, " n"]
    ix_reactant: Int[Array, " n"]
    stoich_by_transition: Float[Array, " n"]

    @property
    def m(self):
        return self.S.shape[0]

    @property
    def n(self):
        return self.S.shape[1]


@jax.jit
def get_rmm_input(r, conc, parameters, kinetic_model):
    ix_reactant = kinetic_model.ix_reactant[r]
    ix_km = kinetic_model.ix_reactant_to_km[r]
    return (
        conc[ix_reactant],
        kinetic_model.stoich_by_transition[r],
        kinetic_model.ix_product[r],
        kinetic_model.temperature,
        parameters.dgf[ix_reactant],
        parameters.log_km[ix_km],
        parameters.log_enzyme[r],
        parameters.log_kcat[r],
    )


@jax.jit
def get_flux(
    conc_balanced: Float[Array, " m"],
    parameters: KineticModelParameters,
    kinetic_model: KineticModel,
) -> Float[Array, " n"]:
    conc = jnp.zeros(kinetic_model.m)
    conc = conc.at[kinetic_model.ix_balanced].set(conc_balanced)
    conc = conc.at[kinetic_model.ix_unbalanced].set(
        jnp.exp(parameters.log_conc_unbalanced)
    )
    get_input = partial(
        get_rmm_input, conc=conc, parameters=parameters, kinetic_model=kinetic_model
    )
    return jnp.array(
        [reversible_michaelis_menten(*get_input(r)) for r in range(kinetic_model.n)]
    )


@jax.jit
def dcdt(
    t: Scalar,
    conc: Float[Array, " n_balanced"],
    args: tuple[KineticModelParameters, KineticModel],
) -> Float[Array, " n_balanced"]:
    parameters, kinetic_model = args
    v = get_flux(conc, parameters, kinetic_model)
    return (kinetic_model.S @ v)[kinetic_model.ix_balanced]
