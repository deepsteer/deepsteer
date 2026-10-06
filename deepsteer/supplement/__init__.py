"""Moved: the paper supplement now lives at ``papers/supplement/`` (LIBRARY_RELEASE_PLAN §C1).

It was never library API (data, provenance and build scripts for the FL/MN papers), so this
shim raises instead of re-exporting. See ``deepsteer/supplement/README.md`` in the repository.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "deepsteer.supplement moved to papers/supplement/ in the repository; it is not part of the "
    "installed library.",
    DeprecationWarning,
    stacklevel=2,
)
raise ImportError(
    "deepsteer.supplement moved to papers/supplement/ in the deepsteer repository "
    "(https://github.com/deepsteer/deepsteer/tree/main/papers/supplement). It is paper "
    "supplement data and scripts, not library API; run its scripts from a clone, e.g. "
    "python3 papers/supplement/scripts/verify.py"
)
