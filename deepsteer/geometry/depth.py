"""Depth-fraction layer conventions shared across the cross-model papers.

The cross-model program (Papers 5, 6, 7) compares models with different layer
counts, so "which layer" must be expressed as a depth *fraction* anchored on
OLMo-3 (32 layers) and mapped onto every other model. Encoding that rule once
here keeps the mapping identical across the panel; if two papers computed it
independently, "OLMo looks different" could become a measurement artifact.

Anchors (OLMo-3, Paper 5 Appendix B):
  * stable band = layers 15..31 of 32 -> depth fractions (15/32, 31/32);
  * headline decomposition layer = depth-fraction 0.5 (OLMo L16/32).
"""

from __future__ import annotations

# Canonical band edges as depth fractions (anchored on OLMo-3, 32 layers).
BAND_FRACS: tuple[float, float] = (15 / 32, 31 / 32)  # (0.46875, 0.96875)

# Primary single layer reported for cross-model comparability: depth-fraction
# 0.5 (OLMo L16/32). The headline decomposition layer.
PRIMARY_FRAC: float = 0.5


def band_layers(n_layers: int) -> tuple[int, int]:
    """Inclusive stable band ``(lo, hi)`` at the canonical depth fractions.

    The literal ``round(f * n)`` rule. It reproduces (15, 31) at 32 layers and
    lands the top edge on the last layer for OLMo-3/Qwen2.5/Llama-3.1. See
    :func:`band_to_final_layer` for the "band runs to the final layer inclusive"
    variant Paper 7 needs for its 48-layer model.
    """
    return round(BAND_FRACS[0] * n_layers), round(BAND_FRACS[1] * n_layers)


def band_to_final_layer(n_layers: int) -> tuple[int, int]:
    """Stable band whose high edge is pinned to the last layer ``n - 1``.

    Same low edge as :func:`band_layers`, but ``hi = n_layers - 1`` rather than
    ``round(0.96875 * n)``. These agree for every layer count in Paper 6's panel
    (24->23, 28->27, 32->31, where ``round(0.96875 * n) == n - 1`` already
    holds). They differ only at 48 layers: ``31/32 * 48 == 46.5`` exactly, and
    Python's round-half-to-even sends 46.5 -> 46, which would stop the band one
    short of the last layer (L46 of 0..47). Pinning ``hi = 47`` keeps the
    "band-to-final-layer" property across the whole panel — the terminal residual
    layer feeds the unembedding and is where late linear structure consolidates,
    so it must be included for the depth span to be genuinely identical.
    """
    return band_layers(n_layers)[0], n_layers - 1


def primary_layer(n_layers: int) -> int:
    """Headline single layer at depth-fraction 0.5."""
    return round(PRIMARY_FRAC * n_layers)


def olmo3_full_attention_layers(n_layers: int) -> list[int]:
    """OLMo-3 hybrid attention: every 4th layer is full attention.

    Annotation only (so any 4-layer periodicity in layer-wise plots is
    attributable to attention type, not signal). Standard full-attention models
    (Qwen2.5, Llama-3.1) carry ``None`` for this field.
    """
    return [i for i in range(n_layers) if (i + 1) % 4 == 0]
