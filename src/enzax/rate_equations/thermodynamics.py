import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Scalar, ScalarLike
from numpy.typing import NDArray


def get_dgr_std(stoichiometry, dgf, temperature, water_stoichiometry):
    RT = temperature * 0.008314
    dgf_water = -150.9
    dgr_std = (
        stoichiometry.T @ dgf + water_stoichiometry * dgf_water
    ).flatten()
    return jnp.exp(-dgr_std / RT)


def get_keq(stoichiometry, dgf, temperature: ScalarLike, water_stoichiometry):
    minus_RT = -0.008314 * temperature
    dgrs = get_dgr_std(stoichiometry, dgf, temperature, water_stoichiometry)
    return jnp.exp(dgrs / minus_RT)


def get_reversibility(
    reactant_conc: Float[Array, " n_reactant"],
    dgf: Float[Array, " n_reactant"],
    temperature: Scalar,
    reactant_stoichiometry: NDArray[np.float64],
    water_stoichiometry: float,
) -> Scalar:
    """Get the reversibility of a reaction.

    Hard coded water dgf is taken from <http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001>.

    The equation is

      1 - exp(((dgr + (RT * quotient)) / RT))

    but it's implemented a bit differently so as to be more numerically stable.
    """
    RT = temperature * 0.008314
    conc_clipped = jnp.clip(reactant_conc, min=1e-9)
    dgf_water = -150.9
    dgr_std = (
        reactant_stoichiometry.T @ dgf + water_stoichiometry * dgf_water
    ).flatten()
    quotient = (reactant_stoichiometry.T @ jnp.log(conc_clipped)).flatten()
    expand = jnp.clip((dgr_std / RT) + quotient, min=-1e2, max=1e2)
    out = -jnp.expm1(expand)[0]
    return eqx.error_if(out, jnp.isnan(out), "Reversibility is nan!")
