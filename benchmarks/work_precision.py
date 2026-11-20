"""Shared code for comparing enzax with AMICI on PEtab problems.

Every tool is timed on the same grid: parameter draws around a problem's
nominal values, times tolerances. One tolerance sets both the integrator's
relative and absolute tolerances and the steady-state stopping criterion, so
that looser settings are less accurate as well as faster. Each gradient is
compared with a reference from AMICI's C++ backend with forward
sensitivities, solved by Newton at the steady state, at tight tolerances.
"""

import csv
import functools
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import petab.v1 as petab

# AMICI removes conservation laws for species that are not constant only
# when this is set. Without it, the steady state's Jacobian is singular for a
# model with a conserved moiety, and AMICI's JAX gradients are wrong.
os.environ["AMICI_EXPERIMENTAL_SBML_NONCONST_CLS"] = "1"

# Every tool runs on one core. AMICI's C++ solver uses one anyway, and on
# problems this small XLA is slower spread over many cores than on one.
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

from amici.importers.petab.v1 import import_petab_problem  # noqa: E402
from amici.sim.jax.petab import SteadyStateEvent, run_simulations  # noqa: E402
from amici.sim.sundials import (  # noqa: E402
    SensitivityMethod,
    SensitivityOrder,
    SteadyStateComputationMode,
    SteadyStateSensitivityMode,
)
from amici.sim.sundials.petab.v1 import LLH, RDATAS, SLLH, simulate_petab  # noqa: E402
from diffrax_bdf import BDF, BDFController  # noqa: E402

from enzax.steady_state import get_steady_state, get_steady_state_hybrid  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
BUILD = HERE / "build"
RESULTS = HERE / "results"
REFERENCE_TOL = 1e-10
REFERENCE_STEADY_STATE_TOL = 1e-12
TOLERANCES = (1e-6, 1e-8, 1e-10)
N_DRAWS = 10
DRAW_SCALE = 0.3
N_REPEATS = 5
MAX_STEPS = 10**4
NEWTON_MAX_STEPS = 10

FIELDS = [
    "problem",
    "tool",
    "solver",
    "tolerance",
    "draw",
    "llh",
    "time_ms",
    "cpu_ms",
    "llh_error",
    "gradient_error",
]

Gradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


def get_estimated(problem: petab.Problem) -> tuple[list[str], list[str]]:
    """Get the estimated parameters' ids and PEtab scales."""
    table = problem.parameter_df
    estimated = table[table[petab.C.ESTIMATE] == 1]
    return list(estimated.index), list(estimated[petab.C.PARAMETER_SCALE])


def get_nominal_theta(problem: petab.Problem) -> np.ndarray:
    """Get the estimated parameters' nominal values on their PEtab scales."""
    ids, scales = get_estimated(problem)
    values = problem.parameter_df.loc[ids, petab.C.NOMINAL_VALUE].to_numpy()
    return np.array(
        [scale_value(v, s) for v, s in zip(values, scales, strict=True)],
    )


def scale_value(value: float, scale: str) -> float:
    if scale == petab.C.LOG:
        return float(np.log(value))
    if scale == petab.C.LOG10:
        return float(np.log10(value))
    return float(value)


def unscale(theta: jax.Array, scales: list[str]) -> jax.Array:
    """Put parameters on their PEtab scales back on their linear scale."""
    return jnp.stack(
        [
            jnp.exp(t)
            if s == petab.C.LOG
            else 10**t
            if s == petab.C.LOG10
            else t
            for t, s in zip(theta, scales, strict=True)
        ],
    )


def get_draws(theta0: np.ndarray) -> list[np.ndarray]:
    """Get N_DRAWS perturbations of the nominal values.

    The nominal values themselves are left out: some are a published
    optimum, where the gradient is close to zero and relative errors in it
    mean little.
    """
    rng = np.random.default_rng(0)
    return [
        theta0 + DRAW_SCALE * rng.standard_normal(theta0.shape)
        for _ in range(N_DRAWS)
    ]


def time_gradient(gradient: Gradient, theta: np.ndarray):
    """Run a gradient once untimed, then N_REPEATS times, and return the
    result and the median wall time in ms."""
    llh, grad = gradient(theta)
    times = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        gradient(theta)
        times.append(time.perf_counter() - start)
    return float(llh), np.asarray(grad), 1e3 * float(np.median(times))


def jax_gradient(llh: Callable[[jax.Array], jax.Array]) -> Gradient:
    value_and_grad = jax.jit(jax.value_and_grad(llh))

    def gradient(theta):
        return jax.block_until_ready(value_and_grad(jnp.asarray(theta)))

    return gradient


def get_enzax_steady_state_options(solver_name: str, tol: float) -> dict:
    """Get keyword arguments for `get_steady_state`, with the same solver,
    step size controller and choice of first step as AMICI's JAX backend."""
    solver, controller = get_diffrax_solver(solver_name, tol)
    return {
        "solver": solver,
        "stepsize_controller": controller,
        "dt0": None,
        "steady_state_rtol": tol,
        "steady_state_atol": tol,
        "max_steps": MAX_STEPS,
    }


def get_enzax_steady_state_solve(solver_name: str, tol: float) -> Callable:
    """Wrap an enzax steady-state solve for a solver name and tolerance.

    A name ending in ", hybrid" gives `get_steady_state_hybrid`, which tries
    Newton before integrating. Any other name gives `get_steady_state`.
    """
    base, _, variant = solver_name.partition(", ")
    options = get_enzax_steady_state_options(base, tol)
    if variant == "hybrid":
        return functools.partial(
            get_steady_state_hybrid,
            **options,
            newton_max_steps=NEWTON_MAX_STEPS,
            newton_rtol=tol,
            newton_atol=tol,
        )
    return functools.partial(get_steady_state, **options)


def get_enzax_time_course_options(solver_name: str, tol: float) -> dict:
    """Get keyword arguments for enzax's time-course solves, with the same
    solver and step size controller as AMICI's JAX backend. Their first step
    is fixed at 1e-6."""
    solver, controller = get_diffrax_solver(solver_name, tol)
    return {
        "solver": solver,
        "stepsize_controller": controller,
        "max_steps": MAX_STEPS,
    }


def get_diffrax_solver(name: str, tol: float):
    """Get a diffrax solver and step size controller by name."""
    if name == "Kvaerno5":
        return diffrax.Kvaerno5(), diffrax.PIDController(rtol=tol, atol=tol)
    if name == "BDF":
        return BDF(), BDFController(rtol=tol, atol=tol, dtmax=1e6)
    msg = f"Unknown solver {name!r}."
    raise ValueError(msg)


class AmiciCpp:
    """AMICI's C++ backend, through its PEtab interface."""

    def __init__(self, problem: petab.Problem, name: str):
        self.problem = problem
        self.ids, _ = get_estimated(problem)
        self.model = import_petab_problem(
            problem,
            output_dir=BUILD / name / "amici_cpp",
        )
        self.default_mode = self.model.get_steady_state_sensitivity_mode()
        self.default_computation_mode = (
            self.model.get_steady_state_computation_mode()
        )
        self.cpu_times: list[float] = []

    def gradient(
        self,
        method: str,
        tol: float,
        steady_state_tol: float | None = None,
        newton_sensitivities: bool = False,
        newton_state: bool = False,
    ) -> Gradient:
        solver = self.model.create_solver()
        solver.set_absolute_tolerance(tol)
        solver.set_relative_tolerance(tol)
        for setter in (
            solver.set_absolute_tolerance_steady_state,
            solver.set_relative_tolerance_steady_state,
            solver.set_absolute_tolerance_steady_state_sensi,
            solver.set_relative_tolerance_steady_state_sensi,
        ):
            setter(tol if steady_state_tol is None else steady_state_tol)
        solver.set_max_steps(MAX_STEPS)
        if newton_state:
            solver.set_newton_max_steps(NEWTON_MAX_STEPS)
        computation_mode = (
            SteadyStateComputationMode.integrateIfNewtonFails
            if newton_state
            else self.default_computation_mode
        )
        solver.set_sensitivity_order(SensitivityOrder.first)
        solver.set_sensitivity_method(
            SensitivityMethod.adjoint
            if method == "adjoint"
            else SensitivityMethod.forward,
        )

        mode = (
            SteadyStateSensitivityMode.newtonOnly
            if newton_sensitivities
            else self.default_mode
        )

        def gradient(theta):
            self.model.set_steady_state_sensitivity_mode(mode)
            self.model.set_steady_state_computation_mode(computation_mode)
            result = simulate_petab(
                petab_problem=self.problem,
                amici_model=self.model,
                solver=solver,
                problem_parameters=dict(zip(self.ids, map(float, theta))),
                scaled_parameters=True,
                scaled_gradients=True,
            )
            self.cpu_times.append(
                sum(r.cpu_time_total for r in result[RDATAS]),
            )
            sllh = result[SLLH] or {}
            return result[LLH], np.array(
                [sllh.get(i, np.nan) for i in self.ids],
            )

        return gradient


class AmiciJax:
    """AMICI's JAX backend."""

    def __init__(self, problem: petab.Problem, name: str):
        self.ids, self.scales = get_estimated(problem)
        self.problem = import_petab_problem(
            problem,
            output_dir=BUILD / name / "amici_jax",
            jax=True,
        )
        self.order = np.array(
            [list(self.problem.parameter_ids).index(i) for i in self.ids],
        )

    def gradient(self, solver_name: str, tol: float) -> Gradient:
        solver, controller = get_diffrax_solver(solver_name, tol)
        event = SteadyStateEvent(rtol=tol, atol=tol)

        def llh(theta):
            values = (
                jnp.asarray(self.problem.parameters)
                .at[self.order]
                .set(
                    unscale(theta, self.scales),
                )
            )
            problem = eqx.tree_at(lambda p: p.parameters, self.problem, values)
            return run_simulations(
                problem,
                solver=solver,
                controller=controller,
                max_steps=MAX_STEPS,
                steady_state_event=event,
            )[0]

        return jax_gradient(llh)


def run_grid(
    name: str,
    problem: petab.Problem,
    enzax_llh: Callable[[str, float], Callable[[jax.Array], jax.Array]],
) -> Path:
    """Time every tool on every draw and tolerance, and write a CSV file.

    `enzax_llh(solver_name, tol)` returns the problem's log likelihood in
    enzax as a function of the estimated parameters on their PEtab scales.
    AMICI's models are compiled afresh, since AMICI reuses a compiled model
    without checking that the SBML it came from is unchanged.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(BUILD / name, ignore_errors=True)
    draws = get_draws(get_nominal_theta(problem))
    cpp = AmiciCpp(problem, name)
    amici_jax = AmiciJax(problem, name)
    reference = cpp.gradient(
        "forward",
        REFERENCE_TOL,
        REFERENCE_STEADY_STATE_TOL,
        newton_sensitivities=True,
    )
    references = [reference(theta) for theta in draws]
    tools = [
        (
            "enzax",
            "Kvaerno5",
            lambda tol: jax_gradient(enzax_llh("Kvaerno5", tol)),
        ),
        ("enzax", "BDF", lambda tol: jax_gradient(enzax_llh("BDF", tol))),
        (
            "enzax",
            "Kvaerno5, hybrid",
            lambda tol: jax_gradient(enzax_llh("Kvaerno5, hybrid", tol)),
        ),
        (
            "enzax",
            "BDF, hybrid",
            lambda tol: jax_gradient(enzax_llh("BDF, hybrid", tol)),
        ),
        (
            "amici_jax",
            "Kvaerno5",
            lambda tol: amici_jax.gradient("Kvaerno5", tol),
        ),
        ("amici_jax", "BDF", lambda tol: amici_jax.gradient("BDF", tol)),
        ("amici_cpp", "adjoint", lambda tol: cpp.gradient("adjoint", tol)),
        (
            "amici_cpp",
            "adjoint, Newton",
            lambda tol: cpp.gradient("adjoint", tol, newton_sensitivities=True),
        ),
        (
            "amici_cpp",
            "adjoint, Newton, hybrid",
            lambda tol: cpp.gradient(
                "adjoint",
                tol,
                newton_sensitivities=True,
                newton_state=True,
            ),
        ),
        ("amici_cpp", "forward", lambda tol: cpp.gradient("forward", tol)),
        (
            "amici_cpp",
            "forward, Newton",
            lambda tol: cpp.gradient("forward", tol, newton_sensitivities=True),
        ),
    ]
    path = RESULTS / f"{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for tool, solver, make in tools:
            for tol in TOLERANCES:
                gradient = make(tol)
                for draw, (theta, (ref_llh, ref_grad)) in enumerate(
                    zip(draws, references, strict=True),
                ):
                    cpp.cpu_times = []
                    try:
                        llh, grad, time_ms = time_gradient(gradient, theta)
                    except Exception as error:  # noqa: BLE001
                        print(f"{name} {tool} {solver} {tol} {draw}: {error}")
                        llh, grad, time_ms = np.nan, np.nan, np.nan
                    scale = np.max(np.abs(ref_grad))
                    writer.writerow(
                        {
                            "problem": name,
                            "tool": tool,
                            "solver": solver,
                            "tolerance": tol,
                            "draw": draw,
                            "llh": llh,
                            "time_ms": time_ms,
                            "cpu_ms": float(np.median(cpp.cpu_times[1:]))
                            if tool == "amici_cpp" and len(cpp.cpu_times) > 1
                            else np.nan,
                            "llh_error": abs(llh - ref_llh),
                            "gradient_error": np.max(np.abs(grad - ref_grad))
                            / scale,
                        },
                    )
                    f.flush()
                print(f"{name}: {tool} {solver} tol {tol} done")
    return path
