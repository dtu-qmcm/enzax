"""Find the posterior mode and the Hessian of the log density there.

Climbs from the prior mean with adam, then polishes with Newton steps, and
saves the mode, its steady states and the Hessian. fit.py uses these to
whiten the sampler's coordinates.
"""

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from common import (
    get_floored_precision,
    get_initial_guess,
    get_labels,
    RESULTS_DIR,
    get_z_posterior,
)

N_STEP = 3000
LEARNING_RATE = 0.02
MIN_LEARNING_RATE = 1e-5
REPORT_EVERY = 250
HESSIAN_STEP = 1e-5
N_NEWTON = 10
GRADIENT_TOLERANCE = 1e-3
MIN_NEWTON_STEP = 1e-3
N_SHOW = 5


def find_mode(log_posterior, z, guess):
    def negative_log_density(z, guess):
        log_density, steady = log_posterior(z, guess)
        return -log_density, steady

    value_and_grad = jax.jit(
        jax.value_and_grad(negative_log_density, has_aux=True),
    )
    optimiser = optax.inject_hyperparams(optax.adam)(
        learning_rate=LEARNING_RATE,
    )
    state = optimiser.init(z)
    good = (z, state, guess)
    for step in range(N_STEP):
        (value, steady), gradient = value_and_grad(z, guess)
        if not jnp.isfinite(value):
            z, state, guess = good
            state.hyperparams["learning_rate"] /= 2
            if state.hyperparams["learning_rate"] < MIN_LEARNING_RATE:
                print(f"Step {step}: no steady state; giving up.")
                break
            continue
        guess = steady
        good = (z, state, guess)
        if step % REPORT_EVERY == 0:
            print(
                f"Step {step}: log density {-float(value):.2f}, "
                f"gradient norm {float(optax.global_norm(gradient)):.3g}, "
                "learning rate "
                f"{float(state.hyperparams['learning_rate']):.3g}",
            )
        updates, state = optimiser.update(gradient, state, z)
        z = optax.apply_updates(z, updates)
    return good[0], good[2]


def get_hessian(value_and_grad, flat, guess):
    columns = []
    for i in range(len(flat)):
        step = jnp.zeros_like(flat).at[i].set(HESSIAN_STEP)
        _, above = value_and_grad(flat + step, guess)
        _, below = value_and_grad(flat - step, guess)
        columns.append((above - below) / (2 * HESSIAN_STEP))
    hessian = np.asarray(jnp.stack(columns, axis=1))
    return (hessian + hessian.T) / 2


def polish_mode(log_posterior, mode, guess):
    flat, unflatten = ravel_pytree(mode)
    value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda x, guess: log_posterior(unflatten(x), guess),
            has_aux=True,
        ),
    )
    (value, _), gradient = value_and_grad(flat, guess)
    hessian = get_hessian(value_and_grad, flat, guess)
    for step in range(N_NEWTON):
        norm = float(jnp.linalg.norm(gradient))
        print(
            f"Newton step {step}: log density {float(value):.2f}, "
            f"gradient norm {norm:.3g}",
        )
        if norm < GRADIENT_TOLERANCE:
            break
        eigenvectors, precision = get_floored_precision(hessian)
        direction = eigenvectors @ (
            (eigenvectors.T @ np.asarray(gradient)) / precision
        )
        size = 1.0
        while size >= MIN_NEWTON_STEP:
            (new_value, steady), new_gradient = value_and_grad(
                flat + size * direction,
                guess,
            )
            if jnp.isfinite(new_value) and new_value >= value:
                break
            size /= 2
        else:
            print("No Newton step increases the log density; stopping.")
            break
        flat = flat + size * direction
        value, gradient, guess = new_value, new_gradient, steady
        hessian = get_hessian(value_and_grad, flat, guess)
    return unflatten(flat), guess, hessian


def report_laplace(hessian, labels):
    eigenvalues, eigenvectors = np.linalg.eigh(-np.asarray(hessian))
    n_bad = int((eigenvalues <= 0).sum())
    print(
        f"{n_bad} of {len(eigenvalues)} eigenvalues of the negative Hessian "
        "are not positive.",
    )
    print(
        f"Eigenvalues range from {eigenvalues[0]:.3g} to "
        f"{eigenvalues[-1]:.3g}.",
    )
    print("Best determined directions (largest weights):")
    for k in range(1, N_SHOW + 1):
        weights = eigenvectors[:, -k]
        top = np.argsort(-np.abs(weights))[:4]
        terms = ", ".join(f"{weights[j]:+.2f} {labels[j]}" for j in top)
        print(f"  {eigenvalues[-k]:.3g}: {terms}")
    if n_bad == 0:
        laplace_sd = np.sqrt(np.diag(np.linalg.inv(-np.asarray(hessian))))
        print(
            "Laplace sd in z-scores: median "
            f"{np.median(laplace_sd):.2f}, smallest {laplace_sd.min():.3g}",
        )


def main():
    split, _, _, log_posterior, z_zero = get_z_posterior()
    mode, guess = find_mode(
        log_posterior,
        z_zero,
        get_initial_guess(),
    )
    mode, guess, hessian = polish_mode(log_posterior, mode, guess)
    labels = get_labels(split, z_zero)
    report_laplace(hessian, labels)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mode_file = RESULTS_DIR / "mode.npz"
    np.savez(
        mode_file,
        mode=np.asarray(ravel_pytree(mode)[0]),
        guess=np.asarray(guess),
        hessian=np.asarray(hessian),
        labels=np.array(labels),
    )
    print(f"Saved the mode to {mode_file}")


if __name__ == "__main__":
    main()
