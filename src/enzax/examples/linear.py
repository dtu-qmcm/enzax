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
    log_substrate_km: dict[str, Array]
    log_product_km: dict[str, Array]
    log_kcat: dict[str, Scalar]
    log_enzyme: dict[str, Array]
    log_ki: dict[str, Array]
    dgf: Array
    temperature: Scalar
    log_conc_unbalanced: Array
    log_dc_inhibitor: dict[str, Array]
    log_dc_activator: dict[str, Array]
    log_tc: dict[str, Array]


stoichiometry = {
    "r1": {"m1e": -1, "m1c": 1},
    "r2": {"m1c": -1, "m2c": 1},
    "r3": {"m2c": -1, "m2e": 1},
}
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
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    species_to_dgf_ix=np.array([0, 0, 1, 1]),
    rate_equations=rate_equations,
)
parameters = ParameterDefinition(
    log_substrate_km={
        "r1": jnp.array([0.1]),
        "r2": jnp.array([0.5]),
        "r3": jnp.array([-1.0]),
    },
    log_product_km={
        "r1": jnp.array([-0.2]),
        "r2": jnp.array([0.0]),
        "r3": jnp.array([0.5]),
    },
    log_kcat={
        "r1": jnp.array(-0.1),
        "r2": jnp.array(0.0),
        "r3": jnp.array(0.1),
    },
    dgf=jnp.array([-3.0, -1.0]),
    log_ki={"r1": jnp.array([]), "r2": jnp.array([1.0]), "r3": jnp.array([])},
    temperature=jnp.array(310.0),
    log_enzyme={
        "r1": jnp.log(jnp.array(0.3)),
        "r2": jnp.log(jnp.array(0.2)),
        "r3": jnp.log(jnp.array(0.1)),
    },
    log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.1])),
    log_tc={"r1": jnp.array(-0.2), "r2": jnp.array(0.3)},
    log_dc_activator={"r1": jnp.array([-0.1]), "r2": jnp.array([])},
    log_dc_inhibitor={"r1": jnp.array([]), "r2": jnp.array([0.2])},
)
true_model = RateEquationModel(structure=structure, parameters=parameters)
steady_state = jnp.array([0.43658744, 0.12695706])
model = RateEquationModel(parameters, structure)
