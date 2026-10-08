import diffrax
import equinox
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, PartitionSpec

from enzax.shard_map_patch import patch_for_shard_map


def steady_state(decay_rate):
    return diffrax.diffeqsolve(
        terms=diffrax.ODETerm(lambda t, y, args: -args * y + 1.0),
        solver=diffrax.Kvaerno5(),
        t0=0.0,
        t1=jnp.inf,
        dt0=0.01,
        y0=jnp.array(0.1),
        args=decay_rate,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        event=diffrax.Event(diffrax.steady_state_event()),
        adjoint=diffrax.ImplicitAdjoint(),
        max_steps=100000,
        throw=False,
    ).ys[0]


def test_patch_refuses_other_versions(monkeypatch):
    monkeypatch.setattr(equinox, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="0.0.0"):
        patch_for_shard_map()


@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="Requires >= 2 devices (set JAX_NUM_CPU_DEVICES=2)",
)
def test_shard_map_gradient_matches_sequential():
    patch_for_shard_map()
    mesh = Mesh(jax.devices()[:2], axis_names=("batch",))
    decay_rates = jnp.array([1.0, 1.1])
    sharded = jax.jit(
        jax.shard_map(
            jax.vmap(jax.grad(steady_state)),
            mesh=mesh,
            in_specs=(PartitionSpec("batch"),),
            out_specs=PartitionSpec("batch"),
            check_vma=False,
        ),
    )
    sequential = jnp.array(
        [jax.grad(steady_state)(rate) for rate in decay_rates],
    )
    assert jnp.allclose(sharded(decay_rates), sequential)
