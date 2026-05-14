"""rocMolKit — GPU-accelerated RDKit operations on AMD GPUs via HIP/ROCm.

Port of NVIDIA nvMolKit. API mirrors nvMolKit / RDKit where possible.

For the direct C++ bindings (zero subprocess overhead but affected by
the open ROCm 7.2.3 + gfx1200 state-leak segfault), import from
``rocmolkit._embedMolecules``. For deterministic embedding via
subprocess + retry, use ``rocmolkit.safe.embed_molecule(s)``.
"""

from . import safe

__version__ = "0.3.1"
__all__ = ["safe"]
