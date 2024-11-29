"""A simple linear kinetic model."""

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Scalar

from enzax.kinetic_model import (
    RateEquationKineticModelStructure,
    RateEquationModel,
)
from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


class ParameterDefinition(eqx.Module):
    log_substrate_km: dict[int, Array]
    log_product_km: dict[int, Array]
    log_kcat: dict[int, Scalar]
    log_enzyme: dict[int, Array]
    log_ki: dict[int, Array]
    dgf: Array
    temperature: Scalar
    log_conc_unbalanced: Array
    log_dc_inhibitor: dict[int, Array]
    log_dc_activator: dict[int, Array]
    log_tc: dict[int, Array]


S = np.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]], dtype=np.float64)
reactions = ["r1", "r2", "r3"]
species = ["m1e", "m1c", "m2c", "m2e"]
balanced_species = ["m1c", "m2c"]
rate_equations = [
    AllostericReversibleMichaelisMenten(
        ix_allosteric_activators=np.array([2]), subunits=1
    ),
    AllostericReversibleMichaelisMenten(
        ix_allosteric_inhibitors=np.array([1]), ix_ki_species=np.array([1])
    ),
    ReversibleMichaelisMenten(water_stoichiometry=0.0),
]
structure = RateEquationKineticModelStructure(
    S=S,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    species_to_dgf_ix=np.array([0, 0, 1, 1]),
    rate_equations=rate_equations,
)
parameters = ParameterDefinition(
    log_substrate_km={
        0: jnp.array([0.1]),
        1: jnp.array([0.5]),
        2: jnp.array([-1.0]),
    },
    log_product_km={
        0: jnp.array([-0.2]),
        1: jnp.array([0.0]),
        2: jnp.array([0.5]),
    },
    log_kcat={0: jnp.array(-0.1), 1: jnp.array(0.0), 2: jnp.array(0.1)},
    dgf=jnp.array([-3.0, -1.0]),
    log_ki={0: jnp.array([]), 1: jnp.array([1.0]), 2: jnp.array([])},
    temperature=jnp.array(310.0),
    log_enzyme={
        0: jnp.log(jnp.array(0.3)),
        1: jnp.log(jnp.array(0.2)),
        2: jnp.log(jnp.array(0.1)),
    },
    log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.1])),
    log_tc={0: jnp.array(-0.2), 1: jnp.array(0.3)},
    log_dc_activator={0: jnp.array([-0.1]), 1: jnp.array([])},
    log_dc_inhibitor={0: jnp.array([]), 1: jnp.array([0.2])},
)
true_model = RateEquationModel(structure=structure, parameters=parameters)
steady_state = jnp.array([0.43658744, 0.12695706])
model = RateEquationModel(parameters, structure)
