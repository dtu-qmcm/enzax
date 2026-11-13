import sys

import pytest

from enzax.utils import import_diffrax_bdf


def test_a_missing_diffrax_bdf_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffrax_bdf", None)
    with pytest.raises(ModuleNotFoundError, match=r"enzax\[dae\]"):
        import_diffrax_bdf()
