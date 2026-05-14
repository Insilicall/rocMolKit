"""Smoke tests - run without ROCm/GPU.

These are pure-Python checks that the package is importable and self-consistent.
Real GPU integration tests live in tests/test_etkdg_parity.py and
test_mmff_parity.py and require a self-hosted ROCm runner.
"""

from __future__ import annotations


def test_import() -> None:
    import rocmolkit

    assert rocmolkit.__version__


def test_version_string() -> None:
    import rocmolkit

    parts = rocmolkit.__version__.split(".")
    assert len(parts) >= 2, f"version should be MAJOR.MINOR[.PATCH], got {rocmolkit.__version__}"
    assert all(p.isdigit() or "-" in p for p in parts), f"non-numeric component in {rocmolkit.__version__}"


def test_has_expected_modules_when_built() -> None:
    """Phase 2+: once C++ extensions are built, these submodules should exist.

    Until then this test only checks the namespace is reachable.
    """
    import rocmolkit

    # Minimum surface always present
    assert hasattr(rocmolkit, "__version__")

    # Optional - present once Phase 2 binds embedMolecules
    # if hasattr(rocmolkit, "embedMolecules"):
    #     assert callable(rocmolkit.embedMolecules.EmbedMolecules)
