import importlib.util
from pathlib import Path

import jax
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "mcmc_demo.py"


@pytest.mark.slow
@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="Requires >= 2 devices (set JAX_NUM_CPU_DEVICES=2)",
)
def test_mcmc_demo_runs(monkeypatch, capsys):
    pytest.importorskip("grapevine")
    patch = pytest.importorskip("teddys_blackjax_utils.shard_map_patch")
    for module, function_name in patch.get_patched_functions():
        monkeypatch.setattr(
            module,
            function_name,
            getattr(module, function_name),
        )
    spec = importlib.util.spec_from_file_location("mcmc_demo", SCRIPT)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    monkeypatch.setattr(demo, "N_WARMUP", 20)
    monkeypatch.setattr(demo, "N_SAMPLE", 20)
    demo.main()
    assert "Divergent transitions" in capsys.readouterr().out
