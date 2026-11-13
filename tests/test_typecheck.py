import importlib
import os
import pkgutil
import sys

import pytest

import enzax


@pytest.mark.skipif(
    os.environ.get("ENZAX_TYPECHECK", "1") != "1",
    reason="type checking is switched off",
)
def test_every_enzax_module_is_type_checked():
    for module_info in pkgutil.walk_packages(enzax.__path__, "enzax."):
        importlib.import_module(module_info.name)
    unchecked = [
        name
        for name, module in sys.modules.items()
        if name.startswith("enzax.")
        and module.__spec__.origin is not None
        and type(module.__spec__.loader).__name__ != "_JaxtypingLoader"
    ]
    assert not unchecked
