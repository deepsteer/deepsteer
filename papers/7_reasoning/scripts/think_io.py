#!/usr/bin/env python3
"""Back-compat shim. The implementation MOVED to ``deepsteer.reasoning.think_io`` (shared across
Paper 7, Direction 1, and future reasoning work). Paper 7 scripts do ``import think_io`` and call
``think_io.<fn>``; this re-exports everything (public + the private ``_THINK_*`` / ``_HARMONY_*``
markers) so those call sites keep working unchanged.
"""

from __future__ import annotations

from deepsteer.reasoning import think_io as _canonical
from deepsteer.reasoning.think_io import *  # noqa: F401,F403

# Mirror every module attribute (incl. privates like _THINK_CLOSE) onto this shim namespace.
globals().update({k: v for k, v in vars(_canonical).items() if not k.startswith("__")})
