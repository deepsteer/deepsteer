#!/usr/bin/env python3
"""Paper 7 model registry: reasoning-model panel, extending Paper 6's conventions.

Paper 7 adds three REASONING models to the cross-model program. The validity of
every cross-model claim still rests on IDENTICAL extraction conventions, so this
module REUSES Paper 6's machinery rather than forking it:

  * the fractional depth -> layer rule (band low edge + primary depth-0.5 layer)
    is imported verbatim from ``papers/6_cross_model/scripts/model_registry.py``;
  * ``assert_matches_model`` keeps the loud-fail-on-drift discipline and extends
    it with an architecture (``model_type``) and MoE-expert-count check;
  * the moral/persona subspace is collected in ``raw`` text, exactly as Paper 6.

Two things are genuinely NEW in Paper 7 and are encoded here once:

  1. A ``think`` input format — the chat template WITH the reasoning trace opened
    (``add_generation_prompt=True`` puts these models at the start of their CoT).
    This sits alongside Paper 6's base-``raw`` and instruct-``chat``.

  2. TWO extraction sites per reasoning model (mirrors Yamaguchi App. G.2):
       * ``END_OF_PROMPT`` — the last input token, before any reasoning is
         generated (the reflexive boundary decision; this is the SAME position
         Paper 6 used for refusal, so the validity anchor compares like-for-like).
       * ``COT`` — token positions INSIDE the generated reasoning trace (the
         deliberative decision; Phase 2 localizes these to "decision sentences").

What stays identical across the panel is the METHOD (last-input-token
diff-of-means at END_OF_PROMPT; decision-sentence-localized diff-of-means at COT;
moral-subspace projection; the depth-fraction band/layer). What necessarily
differs is the CoT DELIMITER: DeepSeek-R1 distills emit ``<think>...</think>``;
GPT-OSS-20B emits an "analysis" channel in the harmony format. That is a parsing
detail (``CoTFormat``), not a method difference, and is recorded per spec so the
Phase 2 sentence splitter branches on it without changing the geometry.

Panel (verified 2026-06-23 against HF primary sources; see PAPER_PLAN.md ledger):
  * gpt_oss_20b   openai/gpt-oss-20b                       gpt_oss  24L  2880  MoE 32/4
  * ds_r1_llama8b deepseek-ai/DeepSeek-R1-Distill-Llama-8B llama    32L  4096  (base Llama-3.1-8B)
  * ds_r1_qwen14b deepseek-ai/DeepSeek-R1-Distill-Qwen-14B qwen2    48L  5120  (base Qwen2.5-14B, general)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Reuse the shared depth-fraction conventions. Papers 5/6/7 read one copy from
# ``deepsteer.geometry.depth`` instead of importing Paper 6's registry by file
# path. Paper 7's band pins its high edge to the last layer; that variant lives
# in core as ``band_to_final_layer`` (see below for why the 48-layer model needs
# it). ``BAND_FRACS``/``PRIMARY_FRAC``/``primary_layer`` re-export unchanged.
# ---------------------------------------------------------------------------
from deepsteer.geometry.depth import (  # noqa: E402
    BAND_FRACS,
    PRIMARY_FRAC,
    band_to_final_layer,
    primary_layer,
)


def band_layers(n_layers: int) -> tuple[int, int]:
    """Paper 7 stable band: low edge at depth 0.46875, high edge pinned to the
    last layer via ``deepsteer.geometry.depth.band_to_final_layer``. The pin
    matters only for the 48-layer Qwen-14B distill (``31/32 * 48 == 46.5`` rounds
    down to 46 under the bare rule); see that function for the full rationale.
    """
    return band_to_final_layer(n_layers)


# ---------------------------------------------------------------------------
# New Paper 7 enums
# ---------------------------------------------------------------------------


class ExtractionSite(str, Enum):
    """Where the refusal direction is extracted along the reasoning rollout."""

    END_OF_PROMPT = "end_of_prompt"  # last input token, before any CoT (reflexive)
    COT = "cot"                      # positions inside the generated reasoning (deliberative)


# CoTFormat moved to the shared deepsteer.reasoning.think_io (one source of truth); re-exported
# here so `from model_registry import CoTFormat` keeps working.
from deepsteer.reasoning.think_io import CoTFormat  # noqa: E402,F401


class Provenance(str, Enum):
    """How the model's reasoning behavior was trained (the functional/imitated axis)."""

    RL_DELIBERATIVE = "rl_deliberative"  # reasoning learned via RL (GPT-OSS deliberative alignment)
    DISTILLED_R1 = "distilled_r1"        # SFT-imitated from DeepSeek-R1 reasoning traces


# Both extraction sites, in the order Phase 1 iterates them.
DEFAULT_SITES: tuple[ExtractionSite, ...] = (ExtractionSite.END_OF_PROMPT, ExtractionSite.COT)


# ---------------------------------------------------------------------------
# Per-model specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasoningModelSpec:
    """One reasoning model in the Paper 7 panel.

    Unlike Paper 6's base+instruct pairing, the object we probe is a single
    reasoning model (``reasoning_repo``). ``base_repo`` is optional: the two R1
    distills have a public base (used only for the supplementary base-shared
    longitudinal probe), GPT-OSS-20B has none. ``n_layers``/``hidden`` are the
    Phase-0a-verified values; drivers must still read the live
    ``model.info.n_layers`` and call :meth:`assert_matches_model`.
    """

    key: str                 # output-path id: gpt_oss_20b / ds_r1_llama8b / ds_r1_qwen14b
    family: str              # gpt_oss / llama / qwen
    reasoning_repo: str      # the model we actually probe
    n_layers: int            # expected (verify against model.info at load)
    hidden: int              # expected
    provenance: Provenance
    expected_model_type: str  # HF config model_type, checked at load
    cot_format: CoTFormat

    base_repo: str | None = None  # public base, where one exists (distills only)
    teacher: str | None = None    # distillation teacher (DeepSeek-R1 for both distills)
    gated: bool = False
    is_moe: bool = False
    n_experts: int | None = None         # total experts (MoE only)
    n_experts_active: int | None = None  # active experts per token (MoE only)
    moe_quant: str | None = None         # e.g. "mxfp4" — Phase 3b must handle dequant

    # CoT delimiters (informational; harmony differs from think-tags). The COT
    # extraction/parsing branches on ``cot_format``, not on these strings.
    cot_open: str = "<think>"
    cot_close: str = "</think>"

    # Pinned extraction conventions (identical across the panel).
    #  * moral (MFT) + persona directions are collected on the REASONING model in
    #    ``raw`` text (Paper 6 pooled raw text on the base; GPT-OSS-20B has no
    #    base, so the only panel-identical choice is the reasoning model itself).
    #  * refusal is extracted in ``think`` format at BOTH sites.
    input_format_subspace: str = "raw"
    input_format_refusal: str = "think"
    extraction_sites: tuple[ExtractionSite, ...] = DEFAULT_SITES

    @property
    def band(self) -> tuple[int, int]:
        """Inclusive stable band at this model's layer count."""
        return band_layers(self.n_layers)

    @property
    def primary_layer(self) -> int:
        """Headline depth-0.5 layer at this model's layer count."""
        return primary_layer(self.n_layers)

    @property
    def out(self) -> str:
        """Primary output subdirectory (reasoning-model caches + decompositions)."""
        return self.key

    @property
    def base_out(self) -> str | None:
        """Base-model cache subdirectory for the longitudinal probe, or None."""
        return f"{self.key}_base" if self.base_repo else None

    def assert_matches_model(
        self,
        n_layers_live: int,
        hidden_live: int | None = None,
        model_type_live: str | None = None,
        n_experts_live: int | None = None,
    ) -> None:
        """Fail loud if a live-loaded model disagrees with the verified geometry.

        Extends Paper 6's layer/hidden check with an architecture (``model_type``)
        check and, for the MoE primary, an expert-count check — the Phase 0a
        "do not assume" guards encoded so a quietly re-released or mis-pinned repo
        fails before it can mis-index a band or a router.
        """
        if n_layers_live != self.n_layers:
            raise RuntimeError(
                f"{self.key}: expected {self.n_layers} layers but loaded {n_layers_live}. "
                f"A re-release changed the geometry; update the registry before trusting "
                f"the band/layer mapping."
            )
        if hidden_live is not None and hidden_live != self.hidden:
            raise RuntimeError(
                f"{self.key}: expected hidden {self.hidden} but loaded {hidden_live}."
            )
        if model_type_live is not None and model_type_live != self.expected_model_type:
            raise RuntimeError(
                f"{self.key}: expected model_type {self.expected_model_type!r} but loaded "
                f"{model_type_live!r}. Wrong repo or a base swap (e.g. a Math variant); "
                f"loud-fail rather than mis-attribute the geometry."
            )
        if self.is_moe and n_experts_live is not None and n_experts_live != self.n_experts:
            raise RuntimeError(
                f"{self.key}: expected {self.n_experts} experts but loaded {n_experts_live}; "
                f"the per-expert orthogonalization (Phase 3b) would mis-target."
            )


# ---------------------------------------------------------------------------
# The panel (primary first, then the two distilled contrasts)
# ---------------------------------------------------------------------------

SPECS: dict[str, ReasoningModelSpec] = {
    "gpt_oss_20b": ReasoningModelSpec(
        key="gpt_oss_20b", family="gpt_oss",
        reasoning_repo="openai/gpt-oss-20b",
        n_layers=24, hidden=2880,
        provenance=Provenance.RL_DELIBERATIVE,
        expected_model_type="gpt_oss",
        cot_format=CoTFormat.HARMONY_ANALYSIS,
        cot_open="<|channel|>analysis<|message|>", cot_close="<|end|>",
        base_repo=None, teacher=None,
        is_moe=True, n_experts=32, n_experts_active=4, moe_quant="mxfp4",
    ),
    "ds_r1_llama8b": ReasoningModelSpec(
        key="ds_r1_llama8b", family="llama",
        reasoning_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        n_layers=32, hidden=4096,
        provenance=Provenance.DISTILLED_R1,
        expected_model_type="llama",
        cot_format=CoTFormat.THINK_TAGS,
        base_repo="meta-llama/Llama-3.1-8B",  # = Paper 6 Llama anchor (base-shared)
        teacher="deepseek-ai/DeepSeek-R1",
        gated=False,  # the distill itself is ungated; the base is gated (probe is optional)
    ),
    "ds_r1_qwen14b": ReasoningModelSpec(
        key="ds_r1_qwen14b", family="qwen",
        reasoning_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        n_layers=48, hidden=5120,
        provenance=Provenance.DISTILLED_R1,
        expected_model_type="qwen2",
        cot_format=CoTFormat.THINK_TAGS,
        base_repo="Qwen/Qwen2.5-14B",  # general (NOT Math); verified at Phase 0a
        teacher="deepseek-ai/DeepSeek-R1",
    ),
}

# Primary first (GPT-OSS-20B), then the two distilled contrasts.
PANEL_ORDER: list[str] = ["gpt_oss_20b", "ds_r1_llama8b", "ds_r1_qwen14b"]

# Phase 3a Shairah extended-refusal comparator base — MEASUREMENT-ONLY (built to
# compare against, never released). Kept out of PANEL_ORDER on purpose.
SHAIRAH_COMPARATOR_BASE: str = "meta-llama/Llama-3.1-8B-Instruct"  # = Paper 6 anchor instruct


def get(key: str) -> ReasoningModelSpec:
    """Look up a spec by short key, with a clear error on typos."""
    if key not in SPECS:
        raise KeyError(f"Unknown model key {key!r}; choices: {PANEL_ORDER}")
    return SPECS[key]


def all_specs() -> list[ReasoningModelSpec]:
    """Specs in panel order (primary first)."""
    return [SPECS[k] for k in PANEL_ORDER]


def lightest_distill() -> ReasoningModelSpec:
    """The cheapest model to smoke (fewest params): the 8B Llama distill."""
    return SPECS["ds_r1_llama8b"]


if __name__ == "__main__":
    print(f"{'key':14s} {'family':8s} {'prov':16s} {'n_L':>3s} {'hid':>5s} "
          f"{'band':>9s} {'prim':>4s} {'moe':>9s} {'cot':>16s}  repo")
    for s in all_specs():
        moe = f"{s.n_experts}/{s.n_experts_active}" if s.is_moe else "dense"
        print(f"{s.key:14s} {s.family:8s} {s.provenance.value:16s} {s.n_layers:3d} "
              f"{s.hidden:5d} {str(list(s.band)):>9s} {s.primary_layer:4d} {moe:>9s} "
              f"{s.cot_format.value:>16s}  {s.reasoning_repo}")
