from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import operator
from typing import Union

from enzax.kinetic_model import (
    RateEquationKineticModelStructure,
    RateEquationModel,
    get_conc,
)
from enzax.steady_state import get_kinetic_model_steady_state
import jax
from jax.flatten_util import ravel_pytree
from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal, norm
from jaxtyping import Array, Float, PyTree, Scalar
import equinox as eqx

FloatArray = Float[Array, " _"]
ParamDict = dict[str, Union[FloatArray, "ParamDict"]]


@dataclass
class Measurement:
    id: str
    value: float
    error_sd: float


@dataclass
class ObservationSet:
    """Measurements from a single experiment."""

    id: str
    conc: list[Measurement]
    flux: list[Measurement]
    enzyme: list[Measurement]


@dataclass
class FreeParamSpec:
    path: tuple[str, ...]
    ix: tuple[tuple[int, ...], ...]
    inits: Float[Array, " _"]


@dataclass
class FixedParamSpec:
    path: tuple[str, ...]
    ix: tuple[tuple[int, ...], ...]
    fixed_values: Float[Array, " _"]


def ind_normal_prior_logdensity(param, prior: Float[Array, "2 _"]):
    """Total log density for an independent normal distribution."""
    return norm.logpdf(param, loc=prior[0], scale=prior[1]).sum()


def mv_normal_prior_logdensity(
    param: Float[Array, " _"],
    prior: tuple[Float[Array, " _"], Float[Array, " _ _"]],
):
    """Total log density for an multivariate normal distribution."""
    return jnp.sum(
        multivariate_normal.logpdf(param, mean=prior[0], cov=prior[1])
    )


def prior_from_truth(
    true_params: ParamDict,
    sd: float,
    is_multivariate: PyTree[bool] | None = None,
) -> PyTree:
    """Get a PyTree of priors from a PyTree of true values and an sd.

    The return value is a pytree of tuples, where the first value is the prior
     location (the same as true_params) and the second value is the prior
    scale (a 1D array with the same length as true_params, filled with sd).

    If is_multivariate is provided, then the prior scale for true leaves is a
    square matrix with sd along the diagonal.

    """

    def prior_leaf_from_truth(
        true_leaf: Float[Array, " k"],
        is_multivariate=False,
    ) -> tuple[Float[Array, " k"], Float[Array, "..."]]:
        sd_arr = jnp.full((len(true_leaf),), sd)
        if is_multivariate:
            sd_arr = jnp.diag(sd_arr)
        return (true_leaf, sd_arr)

    if is_multivariate is None:
        is_multivariate = jax.tree.map(lambda leaf: False, true_params)
    return jax.tree.map(prior_leaf_from_truth, true_params, is_multivariate)


def merge_fixed_and_free_parameters(free: PyTree, fixed: PyTree) -> PyTree:
    """Combine a Pytree of fixed parameters with a Pytree of free parameters.

    Inputs fixed and free must have the same structure, and it is assumed that
    one of the following three cases holds for each leaf:

    Case 1: The leaf of fixed is None. In this case the leaf of free is output.

    Case 2: The leaf of free is None. In this case the leaf of fixed is output.

    Case 3: Both leaves are JAX numpy arrays with dtype float. In this case the
    free array should be non-null, with exactly as many members as there are
    null values in the fixed array. The output is the result of replacing the
    null values of the fixed leaf with the values of the free leaf.

    """

    def merge_leaves(
        free_leaf: Float[Array, " _"] | None,
        fixed_leaf: Float[Array, " _"] | None,
    ) -> Float[Array, " _"]:
        if free_leaf is None and fixed_leaf is None:
            msg = "fixed_leaf and free_leaf cannot both be None"
            raise ValueError(msg)
        elif free_leaf is None:
            return fixed_leaf  # pyright: ignore[reportReturnType]
        elif fixed_leaf is None:
            return free_leaf
        else:
            return fixed_leaf.at[jnp.isnan(fixed_leaf)].set(free_leaf)

    return jax.tree.map(merge_leaves, free, fixed, is_leaf=lambda x: x is None)


def enzax_prior_logdensity(prior: PyTree, parameters: PyTree) -> Scalar:
    def enzax_prior_logdensity_leaf(param_prior, param_leaf):
        loc, scale = param_prior
        return norm.logpdf(param_leaf, loc, scale).sum()

    prior_logdensities = jax.tree.map(
        enzax_prior_logdensity_leaf,
        prior,
        parameters,
        is_leaf=lambda node: isinstance(node, tuple),
    )
    return jax.tree.reduce(operator.add, prior_logdensities)


def enzax_log_likelihood(
    observations: list[ObservationSet], conc, flux, enzyme
) -> Scalar:
    """Stub!"""
    return jnp.array(1.0)


def enzax_log_density(
    free_parameters: PyTree,
    structure: RateEquationKineticModelStructure,
    observations: list[ObservationSet],
    prior: PyTree,
    fixed_parameters: PyTree | None = None,
    guess: Float[Array, " _"] | None = None,
) -> Scalar:
    if guess is None:
        guess = jnp.full((len(structure.balanced_species_ix)), 0.01)
    if fixed_parameters is not None:
        parameters = eqx.combine(free_parameters, fixed_parameters)
    else:
        parameters = free_parameters
    # find the steady state concentration and flux
    model = RateEquationModel(parameters, structure)
    steady = get_kinetic_model_steady_state(model, guess)
    conc = get_conc(steady, parameters["log_conc_unbalanced"], structure)
    flux = model.flux(steady)
    flat_log_enzyme, _ = ravel_pytree(parameters["log_enzyme"])
    enzyme = jnp.exp(flat_log_enzyme)
    log_prior = enzax_prior_logdensity(prior, free_parameters)
    log_likelihood = enzax_log_likelihood(observations, conc, flux, enzyme)
    # likelihood
    return log_prior + log_likelihood


def search(to_search: dict, queries: tuple[str, ...]):
    out = to_search
    for q in queries:
        out = out[q]
    return out


def is_none(x):
    return x is None


def split_given_free(
    params: PyTree,
    free_spec: list[FreeParamSpec],
) -> tuple[PyTree, PyTree]:
    """Split parameters into free and fixed given free_spec.


    Parameters
    ----------
    params_all : PyTree
        The parameters to split.
    free_spec : list[FreeParamSpec]
        A specification of the free parameters. Mutually exclusive with
        fixed_spec. All parameters in params_all except those specified in
         free_spec are considered fixed.

    Returns
    -------
    free : PyTree
        PyTree of free parameters.
    fixed : PyTree
        PyTree of fixed parameters.

    Examples
    --------
    ```python
    from enzax.examples.linear import parameters
    from enzax.statistical_modelling import FreeParamSpec, split_given_free
    import jax.numpy as jnp

    free_spec = [
        FreeParamSpec(path=("log_kcat", "r1"), ix=(), inits=jnp.array(0.3)),
        FreeParamSpec(path=("temperature",), ix=(), inits=jnp.array(3.3)),
        FreeParamSpec(path=("dgf",), ix=(0,), inits=jnp.array(33.3)),
    ]
    free_params, fixed_params = split_given_free(parameters, free_spec)
    print(free_params)
    print(fixed_params)

    ```

    """
    fixed = deepcopy(params)
    free = jax.tree.map(lambda _: None, params)
    for ps in free_spec:
        filter_spec = partial(search, queries=ps.path)
        free = eqx.tree_at(filter_spec, free, replace=ps.inits, is_leaf=is_none)
        fixed = eqx.tree_at(
            filter_spec,
            fixed,
            replace_fn=lambda arr: arr.at[ps.ix].set(jnp.nan),
            is_leaf=is_none,
        )
    return free, fixed


def split_given_fixed(
    params: PyTree,
    fixed_spec: list[FixedParamSpec],
) -> tuple[PyTree, PyTree]:
    """Split parameters into free and fixed given fixed_spec.

    params : PyTree
        The parameters to split.

    fixed_spec : list[FixedParamSpec]
        A specification of the fixed parameters. Mutually exclusive with
        free_spec. All parameters in params_all except those specified in
        fixed_spec are considered free.

    Returns
    -------
    free : PyTree
        PyTree of free parameters.
    fixed : PyTree
        PyTree of fixed parameters.

    Examples
    --------
    ```python

    from enzax.examples.linear import parameters
    from enzax.statistical_modelling import FixedParamSpec, split_given_fixed
    import jax.numpy as jnp

    fixed_spec = [
        FixedParamSpec(
            path=("log_kcat", "r1"),
            ix=(),
            fixed_values=jnp.array(0.3),
        ),
        FixedParamSpec(
            path=("temperature",),
            ix=(),
            fixed_values=jnp.array(3.3),
        ),
        FixedParamSpec(path=("dgf",), ix=(0,), fixed_values=jnp.array(33.3)),
    ]
    free_params, fixed_params = split_given_fixed(parameters, fixed_spec)
    print(free_params)
    print(fixed_params)
    """

    fixed = jax.tree.map(lambda _: None, params)
    free = deepcopy(params)
    for ps in fixed_spec:
        filter = partial(search, queries=ps.path)
        arr = filter(params)
        fixed_p = jnp.full_like(arr, jnp.nan).at[ps.ix].set(ps.fixed_values)  # pyright: ignore[reportArgumentType]
        if len(arr.shape) > 0:  # pyright: ignore[reportAttributeAccessIssue]
            mask = jnp.ones_like(arr, dtype=jnp.int16).at[ps.ix].set(0)  # pyright: ignore[reportArgumentType]
            free_p = arr[mask == 1]
        else:
            free_p = None
        fixed = eqx.tree_at(filter, fixed, replace=fixed_p, is_leaf=is_none)
        free = eqx.tree_at(filter, free, replace=free_p, is_leaf=is_none)
    return free, fixed
