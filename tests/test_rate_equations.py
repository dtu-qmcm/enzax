import pytest
from jax import numpy as jnp
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    AllostericReversibleMichaelisMenten,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


@pytest.mark.parametrize(
    ["rate_equation", "expected_rate"],
    [
        (IrreversibleMichaelisMenten, 0.020949269),
        (ReversibleMichaelisMenten, 0.017686484),
        (AllostericIrreversibleMichaelisMenten, 0.019444676),
        (AllostericReversibleMichaelisMenten, 0.017833749),
    ],
)
def test_rate_equations_linear_one(
    rate_equation, expected_rate, linear_parameters, linear_structure
):
    """Check output of different rate equations for rate 1 of the linear model.

    Note that parameters `linear_parameters` and `linear_structure` are not defined in this file. They are fixtures defined in the file `conftest.py`: pytest makes these anywhere in the same directory. See https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files.

    """
    conc = jnp.array([0.5, 0.2, 0.1, 0.3])
    ix = 1
    f = rate_equation(linear_parameters, linear_structure, ix)
    assert jnp.equal(f(conc[linear_structure.ix_reactant[ix]]), expected_rate)
