"""rocMolKit — GPU-accelerated RDKit operations on AMD GPUs via HIP/ROCm.

Port of NVIDIA nvMolKit. API mirrors nvMolKit / RDKit where possible.

For the direct C++ bindings (zero subprocess overhead but affected by
the open ROCm 7.2.3 + gfx1200 state-leak segfault), import from
``rocmolkit._embedMolecules``. For deterministic embedding via
subprocess + retry, use ``rocmolkit.safe.embed_molecule(s)``.
"""

from . import safe

# The _mmffOptimization / _uffOptimization / _batchedForcefield bindings take a
# BatchHardwareOptions default argument whose boost-python to_python converter is
# registered by _embedMolecules' module init. Importing one of those submodules
# in a process that never imported _embedMolecules raises:
#   TypeError: No to_python converter found for nvMolKit::BatchHardwareOptions
# Importing a submodule runs this package __init__ first, so eagerly importing
# _embedMolecules here makes `from rocmolkit._mmffOptimization import ...` work
# regardless of order. Best-effort: stays importable when the extension isn't
# built (pure-Python/no-GPU environments).
try:
    from . import _embedMolecules as _embedMolecules  # noqa: F401
except ImportError:
    pass

__version__ = "0.4.3"
__all__ = ["safe"]
