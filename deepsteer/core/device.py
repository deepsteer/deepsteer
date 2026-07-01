"""Device-side helpers shared across scripts: memory cleanup and MPS shims.

Small, dependency-light utilities that many analysis scripts need verbatim. They
lived as copy-pasted ``_clear_memory`` / ``_histc_mps_fallback`` blocks across the
paper scripts (17 and 8 copies respectively); centralized here so there is one
copy to fix.
"""

from __future__ import annotations

import gc

import torch


def clear_memory() -> None:
    """Free GPU/MPS memory: run the GC and empty the CUDA/MPS allocator caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# The original ``torch.histc``, captured once at import (before any patching) so
# the fallback can delegate to it. MPS has no histc kernel, so on MPS (and for
# non-float inputs) we compute on CPU and move the result back.
_orig_histc = torch.histc


def _histc_mps_fallback(input, bins=100, min=0, max=0):  # noqa: A002 (mirror torch.histc signature)
    if input.device.type == "mps" or not input.is_floating_point():
        return _orig_histc(input.cpu().float(), bins, min, max).to(input.device)
    return _orig_histc(input, bins, min, max)


def enable_mps_histc_fallback() -> None:
    """Route ``torch.histc`` through the CPU fallback (MPS lacks a histc kernel).

    Idempotent: repeated calls just re-point ``torch.histc`` at the same fallback.
    """
    torch.histc = _histc_mps_fallback
