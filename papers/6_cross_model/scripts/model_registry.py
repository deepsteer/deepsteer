#!/usr/bin/env python3
"""Paper 6 model registry: the single source of truth for cross-model conventions.

The validity of the cross-model comparison rests entirely on IDENTICAL extraction
conventions across all three families. The same depth-fraction layer rule, the
same contrast sets, the same pooling (mean), the same direction kind (mean-diff
primary, probe reported), and the same input_format must hold for OLMo-3,
Qwen2.5, and Llama-3.1. If conventions differ across models, "OLMo looks
different" becomes a measurement artifact, the exact failure Phase 0b exists to
avoid.

This module encodes those conventions once so every Phase-1/2 driver reads layer
indices and formats from here instead of hardcoding the OLMo-indexed band
[15, 31] / layer 16. The OLMo numbers are the anchor: the fractional rule
reproduces Paper 5's band and layer exactly, and maps the same depth fractions
onto the 28-layer Qwen and 32-layer Llama.

Conventions pinned here:
  * Moral (MFT 6-foundation) and persona directions are extracted on the BASE
    model in ``raw`` text format (Paper 5 Sprint 1 decision; the moral/persona
    collection path pools raw text via ``collect_batch_activations`` and never
    applies a chat template, so Qwen2.5-base shipping a ChatML template does not
    leak into the geometry).
  * The refusal direction is extracted on the INSTRUCT model in ``chat`` format
    (Arditi/Heretic last-input-token diff-of-means). The three instruct chat
    templates inject different system prompts, but refusal = mean-diff(harmful,
    harmless) with the same template on both sides, so the system prompt is
    common-mode and cancels.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Canonical band / layer fractions (anchored on OLMo-3, 32 layers)
# ---------------------------------------------------------------------------
# OLMo-3 stable band (Paper 5 Appendix B) is layers 15..31 of 32 -> depth
# fractions (15/32, 31/32). Mapping ``round(f * n_layers)`` reproduces (15, 31)
# at 32 layers and the top edge always lands on the last layer (n_layers - 1).
BAND_FRACS: tuple[float, float] = (15 / 32, 31 / 32)  # (0.46875, 0.96875)

# Primary single layer reported for cross-model comparability: depth-fraction
# 0.5 (OLMo L16/32). This is the headline decomposition layer.
PRIMARY_FRAC: float = 0.5


def band_layers(n_layers: int) -> tuple[int, int]:
    """Inclusive stable band ``(lo, hi)`` at the canonical depth fractions."""
    return round(BAND_FRACS[0] * n_layers), round(BAND_FRACS[1] * n_layers)


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


# ---------------------------------------------------------------------------
# Per-model specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One base+instruct family in the Paper 6 panel.

    ``n_layers``/``hidden`` are the expected values confirmed at Phase 0a; they
    drive offline band/command construction. Drivers must still read the live
    ``model.info.n_layers`` and call :meth:`assert_matches_model` so a quietly
    re-released model fails loud instead of silently mis-indexing the band.
    """

    key: str           # short id used in output paths: olmo3 / qwen25 / llama31
    family: str        # olmo / qwen / llama
    base_repo: str
    instruct_repo: str
    n_layers: int      # expected (verify against model.info at load)
    hidden: int        # expected
    gated: bool = False
    has_full_attention_pattern: bool = False  # True only for OLMo-3 hybrid attn

    # Pinned extraction conventions (identical across the panel).
    input_format_base: str = "raw"      # moral + persona on the base
    input_format_refusal: str = "chat"  # refusal on the instruct

    @property
    def band(self) -> tuple[int, int]:
        """Inclusive stable band at this model's layer count."""
        return band_layers(self.n_layers)

    @property
    def primary_layer(self) -> int:
        """Headline depth-0.5 layer at this model's layer count."""
        return primary_layer(self.n_layers)

    @property
    def full_attention_layers(self) -> list[int] | None:
        """Hybrid-attention annotation, or ``None`` for full-attention models."""
        if not self.has_full_attention_pattern:
            return None
        return olmo3_full_attention_layers(self.n_layers)

    @property
    def base_out(self) -> str:
        """Output subdirectory holding the base moral/persona caches."""
        return f"{self.key}_base"

    @property
    def instruct_out(self) -> str:
        """Output subdirectory holding the instruct refusal/persona caches."""
        return f"{self.key}_instruct"

    def assert_matches_model(self, n_layers_live: int, hidden_live: int | None = None) -> None:
        """Fail loud if a live-loaded model disagrees with the expected geometry."""
        if n_layers_live != self.n_layers:
            raise RuntimeError(
                f"{self.key}: expected {self.n_layers} layers but loaded model has "
                f"{n_layers_live}. A re-release changed the geometry; update the "
                f"registry before trusting the band/layer mapping."
            )
        if hidden_live is not None and hidden_live != self.hidden:
            raise RuntimeError(
                f"{self.key}: expected hidden {self.hidden} but loaded {hidden_live}."
            )


SPECS: dict[str, ModelSpec] = {
    "olmo3": ModelSpec(
        key="olmo3", family="olmo",
        base_repo="allenai/Olmo-3-1025-7B",
        instruct_repo="allenai/Olmo-3-7B-Instruct",
        n_layers=32, hidden=4096,
        has_full_attention_pattern=True,
    ),
    "qwen25": ModelSpec(
        key="qwen25", family="qwen",
        base_repo="Qwen/Qwen2.5-7B",
        instruct_repo="Qwen/Qwen2.5-7B-Instruct",
        n_layers=28, hidden=3584,
    ),
    "llama31": ModelSpec(
        key="llama31", family="llama",
        base_repo="meta-llama/Llama-3.1-8B",
        instruct_repo="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32, hidden=4096, gated=True,
    ),
}

# Anchor first, then comparisons (display / iteration order).
PANEL_ORDER: list[str] = ["olmo3", "qwen25", "llama31"]


def get(key: str) -> ModelSpec:
    """Look up a spec by short key, with a clear error on typos."""
    if key not in SPECS:
        raise KeyError(f"Unknown model key {key!r}; choices: {PANEL_ORDER}")
    return SPECS[key]


def all_specs() -> list[ModelSpec]:
    """Specs in panel order (anchor first)."""
    return [SPECS[k] for k in PANEL_ORDER]


if __name__ == "__main__":
    # Quick human-readable dump of the per-model conventions.
    print(f"{'key':8s} {'family':6s} {'n_L':>3s} {'hid':>5s} {'band':>9s} "
          f"{'primary':>7s} {'gated':>5s}  base / instruct")
    for s in all_specs():
        print(f"{s.key:8s} {s.family:6s} {s.n_layers:3d} {s.hidden:5d} "
              f"{str(list(s.band)):>9s} {s.primary_layer:7d} {str(s.gated):>5s}  "
              f"{s.base_repo} / {s.instruct_repo}")
