from enzax.rate_equations import (
    AllostericReversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)
from jax import numpy as jnp
from enzax.kinetic_model import (
    KineticModel,
    KineticModelStructure,
    KineticModelParameters,
    UnparameterisedKineticModel,
)
import pytest


@pytest.fixture
def linear_parameters():
    return KineticModelParameters(
        log_kcat=jnp.array([-0.1, 0.0, 0.1]),
        log_enzyme=jnp.log(jnp.array([0.3, 0.2, 0.1])),
        dgf=jnp.array([-3, -1.0]),
        log_km=jnp.array([0.1, -0.2, 0.5, 0.0, -1.0, 0.5]),
        log_ki=jnp.array([1.0]),
        log_conc_unbalanced=jnp.log(jnp.array([0.5, 0.1])),
        temperature=jnp.array(310.0),
        log_transfer_constant=jnp.array([-0.2, 0.3]),
        log_dissociation_constant=jnp.array([-0.1, 0.2]),
    )


@pytest.fixture
def linear_structure():
    return KineticModelStructure(
        S=jnp.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]]),
        ix_balanced=jnp.array([1, 2]),
        ix_reactant=jnp.array([[0, 1], [1, 2], [2, 3]]),
        ix_substrate=jnp.array([[0], [0], [0]]),
        ix_product=jnp.array([[1], [1], [1]]),
        ix_rate_to_km=jnp.array([[0, 1], [2, 3], [4, 5]]),
        ix_mic_to_metabolite=jnp.array([0, 0, 1, 1]),
        ix_unbalanced=jnp.array([0, 3]),
        stoich_by_rate=jnp.array([[-1, 1], [-1, 1], [-1, 1]]),
        subunits=jnp.array([1, 1, 1]),
        ix_rate_to_tc=[[0], [1], []],
        ix_rate_to_dc_activation=[[0], [], []],
        ix_rate_to_dc_inhibition=[[], [1], []],
        ix_dc_species=jnp.array([2, 1]),
        ix_ki_species=jnp.array([1]),
        ix_rate_to_ki=[[], [0], []],
    )


@pytest.fixture
def unparameterised_linear_model(linear_structure):
    return UnparameterisedKineticModel(
        linear_structure,
        [
            AllostericReversibleMichaelisMenten,
            AllostericReversibleMichaelisMenten,
            ReversibleMichaelisMenten,
        ],
    )


@pytest.fixture
def linear_model(linear_parameters, unparameterised_linear_model):
    return KineticModel(linear_parameters, unparameterised_linear_model)


@pytest.fixture
def rate_equation_data_linear_one(linear_parameters, linear_structure):
    conc = jnp.array([0.5, 0.2, 0.1, 0.3])
    return (linear_parameters, linear_structure, conc, 1)
