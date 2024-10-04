"""Blackjax implementation of grapevine sampler for equation solver models."""

from typing import Callable, NamedTuple
import jax
from jax import numpy as jnp
from blackjax.types import ArrayTree, ArrayLikeTree, PRNGKey
from blackjax import GenerateSamplingAPI
from blackjax.base import SamplingAlgorithm

from blackjax.mcmc.nuts import NUTSInfo, iterative_nuts_proposal
from blackjax.mcmc.metrics import KineticEnergy, MetricTypes, default_metric
from blackjax.mcmc.integrators import euclidean_momentum_update_fn


class GrapevineState(NamedTuple):
    position: ArrayTree
    logdensity: float
    logdensity_grad: ArrayTree
    guess: ArrayTree


class GrapevineIntegratorState(NamedTuple):
    position: ArrayTree
    momentum: ArrayLikeTree
    logdensity: float
    logdensity_grad: ArrayTree
    guess: ArrayTree


def grapevine_euclidean_position_update_fn(logdensity_fn: Callable):
    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn, has_aux=True)

    def update(
        position: ArrayTree,
        kinetic_grad: ArrayTree,
        step_size: float,
        coef: float,
        guess: ArrayTree,
    ):
        new_position = jax.tree_util.tree_map(
            lambda x, grad: x + step_size * coef * grad,
            position,
            kinetic_grad,
        )
        (
            (logdensity, new_guess),
            logdensity_grad,
        ) = logdensity_and_grad_fn(new_position, guess=guess)
        del guess
        return new_position, logdensity, logdensity_grad, new_guess

    return update


def format_grapevine_euclidean_state_output(
    position,
    momentum,
    logdensity,
    logdensity_grad,
    kinetic_grad,
    position_update_info,
    momentum_update_info,
):
    del kinetic_grad, momentum_update_info
    return GrapevineIntegratorState(
        position,
        momentum,
        logdensity,
        logdensity_grad,
        position_update_info,
    )


def grapevine_generalized_two_stage_integrator(
    operator1: Callable,
    operator2: Callable,
    coefficients: list[float],
    format_output_fn: Callable = lambda x: x,
):
    def one_step(state: GrapevineIntegratorState, step_size: float):
        position, momentum, _, logdensity_grad, guess = state
        # auxiliary infomation generated during integration for diagnostics.
        # It is updated by the operator1 and operator2 at each call.
        momentum_update_info = None
        position_update_info = guess
        for i, coef in enumerate(coefficients[:-1]):
            if i % 2 == 0:
                momentum, kinetic_grad, momentum_update_info = operator1(
                    momentum,
                    logdensity_grad,
                    step_size,
                    coef,
                    auxiliary_info=momentum_update_info,
                    is_last_call=False,
                )
            else:
                (
                    position,
                    logdensity,
                    logdensity_grad,
                    position_update_info,
                ) = operator2(
                    position,
                    kinetic_grad,
                    step_size,
                    coef,
                    guess=position_update_info,
                )
        # Separate the last steps to short circuit the computation of the
        # kinetic_grad.
        momentum, kinetic_grad, momentum_update_info = operator1(
            momentum,
            logdensity_grad,
            step_size,
            coefficients[-1],
            momentum_update_info,
            is_last_call=True,
        )
        return format_output_fn(
            position,
            momentum,
            logdensity,
            logdensity_grad,
            kinetic_grad,
            position_update_info,
            momentum_update_info,
        )

    return one_step


def generate_grapevine_euclidean_integrator(coefficients):
    def euclidean_integrator(
        logdensity_fn: Callable, kinetic_energy_fn: KineticEnergy
    ) -> Callable:
        position_update_fn = grapevine_euclidean_position_update_fn(
            logdensity_fn
        )
        momentum_update_fn = euclidean_momentum_update_fn(kinetic_energy_fn)
        one_step = grapevine_generalized_two_stage_integrator(
            momentum_update_fn,
            position_update_fn,
            coefficients,
            format_output_fn=format_grapevine_euclidean_state_output,
        )
        return one_step

    return euclidean_integrator


def init(position: ArrayTree, logdensity_fn: Callable):
    (logdensity, new_guess), logdensity_grad = jax.value_and_grad(
        logdensity_fn, has_aux=True
    )(position, guess=jnp.full((5,), 0.01))
    return GrapevineState(position, logdensity, logdensity_grad, new_guess)


velocity_verlet_coefficients = [0.5, 1.0, 0.5]
grapevine_velocity_verlet = generate_grapevine_euclidean_integrator(
    velocity_verlet_coefficients
)


def grapevine_build_kernel(
    integrator: Callable = grapevine_velocity_verlet,
    divergence_threshold: int = 1000,
):
    def kernel(
        rng_key: PRNGKey,
        state: GrapevineState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: MetricTypes,
        max_num_doublings: int = 10,
    ) -> tuple[GrapevineState, NUTSInfo]:
        """Generate a new sample with the NUTS kernel."""

        metric = default_metric(inverse_mass_matrix)
        symplectic_integrator = integrator(logdensity_fn, metric.kinetic_energy)
        proposal_generator = iterative_nuts_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            metric.check_turning,
            max_num_doublings,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad, guess = state
        momentum = metric.sample_momentum(key_momentum, position)

        integrator_state = GrapevineIntegratorState(
            position, momentum, logdensity, logdensity_grad, guess
        )
        proposal, info = proposal_generator(
            key_integrator, integrator_state, step_size
        )
        proposal = GrapevineState(
            proposal.position,
            proposal.logdensity,
            proposal.logdensity_grad,
            proposal.guess,
        )
        return proposal, info

    return kernel


def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    inverse_mass_matrix: MetricTypes,
    *,
    max_num_doublings: int = 10,
    divergence_threshold: int = 1000,
) -> SamplingAlgorithm:
    kernel = grapevine_build_kernel(
        integrator=grapevine_velocity_verlet,
        divergence_threshold=divergence_threshold,
    )

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            max_num_doublings,
        )

    return SamplingAlgorithm(init_fn, step_fn)


grapevine_algorithm = GenerateSamplingAPI(
    as_top_level_api,
    init,
    grapevine_build_kernel,
)
