#!/usr/bin/env python3
"""Phase 0b local gate: no-GPU checks that must pass before any RunPod work.

Verifies, without loading a single model, the things that make the Paper 7
reasoning-model comparison valid:

  1. The fractional band/layer rule is REUSED from Paper 6 (not forked) and maps
     sensibly onto the new layer counts (24 / 32 / 48), including the 48-layer
     top-edge pin (the one place the bare round() rule and the documented "top =
     last layer" intent diverge).
  2. The validity anchor: the Llama-8B distill shares Llama-3.1-8B's geometry, so
     its band/primary layer must EXACTLY equal Paper 6's ``llama31`` band/primary.
     (Phase 0c then checks the END-OF-PROMPT decomposition lands near Paper 6's
     Llama numbers on the GPU.)
  3. Per-model specs + pinned conventions are identical across the panel
     (subspace=raw, refusal=think, both extraction sites), and the Phase-0a
     provenance facts are encoded (model_type, MoE 32/4, distill bases/teacher).
  4. ``assert_matches_model`` fails loud on layer / hidden / architecture / expert
     drift.
  5. The shared Paper 6 / Paper 5 / Paper 3 tooling imports cleanly and the real
     contrast sets (Arditi refusal prompts, MFT probing v2) are present.

Run:
    python papers/7_reasoning/scripts/local_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PAPER7 = _THIS.parent.parent
_REPO = _PAPER7.parent.parent
_P6_SCRIPTS = _REPO / "papers" / "6_cross_model" / "scripts"
_P5_SCRIPTS = _REPO / "papers" / "5_moral_alignment" / "scripts"
_P3_SCRIPTS = _REPO / "papers" / "3_moral_geometry" / "scripts"

# Reused Paper 5 / Paper 6 dirs for the tooling-import check, with Paper 7's
# scripts inserted LAST so it sits at index 0: ``import model_registry`` must
# resolve to THIS paper's registry, not Paper 6's same-named module.
sys.path.insert(0, str(_P5_SCRIPTS))
sys.path.insert(0, str(_P6_SCRIPTS))
sys.path.insert(0, str(_THIS.parent))

import model_registry as reg  # noqa: E402  (Paper 7 registry)

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _FAILS.append(msg)


def test_band_math() -> None:
    print("\n[1] fractional band / primary-layer mapping (reused from Paper 6)")
    reg6 = reg.reg6
    # Reuse parity: Paper 6 anchors are reproduced exactly.
    check(reg.band_layers(32) == (15, 31), "32L band == (15, 31) [Paper 5/6 anchor]")
    check(reg.primary_layer(32) == 16, "32L primary == L16 (depth 0.5)")
    # New layer counts.
    check(reg.band_layers(24) == (11, 23), "GPT-OSS 24L band == (11, 23)")
    check(reg.primary_layer(24) == 12, "GPT-OSS 24L primary == L12")
    check(reg.band_layers(48) == (22, 47), "Qwen-14B 48L band == (22, 47) [band-to-final-layer]")
    check(reg.primary_layer(48) == 24, "Qwen-14B 48L primary == L24")
    # Convention = "band runs to the final layer inclusive" (the terminal residual
    # state feeding the unembedding). The bare round() rule breaks this ONLY at
    # 48L: 31/32*48 == 46.5 exactly, and round-half-to-even sends it to 46, which
    # would make Qwen-14B the lone model stopping short of its last layer. Pinning
    # hi=47 keeps band-to-final-layer identical across the whole panel.
    check(reg6.band_layers(48) == (22, 46),
          "bare Paper-6 rule gives (22, 46) at 48L (31/32*48==46.5, round-half-to-even)")
    check(reg.band_layers(48)[1] == 47 and reg6.band_layers(48)[1] == 46,
          "pin hi=47 (last layer) is MORE convention-faithful than the literal round()")
    # Top edge always lands on the last layer across the panel's counts.
    for n in (24, 32, 48):
        lo, hi = reg.band_layers(n)
        check(hi == n - 1, f"{n}L band top == last layer ({n - 1})")
        check(0 < lo < reg.primary_layer(n) < hi, f"{n}L: 0 < lo < primary < hi monotone")


def test_validity_anchor() -> None:
    print("\n[2] validity anchor: Llama-8B distill shares Paper 6 Llama geometry")
    reg6 = reg.reg6
    llama_p6 = reg6.get("llama31")
    distill = reg.get("ds_r1_llama8b")
    check(distill.n_layers == llama_p6.n_layers == 32, "both 32 layers")
    check(distill.hidden == llama_p6.hidden == 4096, "both hidden 4096")
    check(distill.band == llama_p6.band == (15, 31),
          "distill band == Paper 6 llama31 band (15, 31)")
    check(distill.primary_layer == llama_p6.primary_layer == 16,
          "distill primary == Paper 6 llama31 primary L16")
    check(distill.base_repo == llama_p6.base_repo == "meta-llama/Llama-3.1-8B",
          "distill base_repo == Paper 6 llama31 base (base-shared longitudinal probe)")


def test_specs() -> None:
    print("\n[3] per-model specs + pinned conventions + Phase-0a provenance")
    specs = reg.all_specs()
    check(len(specs) == 3, "panel has 3 reasoning models")
    check([s.key for s in specs] == ["gpt_oss_20b", "ds_r1_llama8b", "ds_r1_qwen14b"],
          "primary-first order (GPT-OSS, then the two distills)")
    for s in specs:
        check(s.input_format_subspace == "raw", f"{s.key}: moral/persona subspace format == raw")
        check(s.input_format_refusal == "think", f"{s.key}: refusal format == think")
        check(tuple(s.extraction_sites) == (reg.ExtractionSite.END_OF_PROMPT,
                                            reg.ExtractionSite.COT),
              f"{s.key}: both extraction sites (end-of-prompt, CoT)")
        check(s.band[0] < s.band[1], f"{s.key}: band well-ordered {list(s.band)}")

    gpt = reg.get("gpt_oss_20b")
    check(gpt.provenance == reg.Provenance.RL_DELIBERATIVE, "GPT-OSS = RL-deliberative")
    check(gpt.expected_model_type == "gpt_oss", "GPT-OSS model_type == gpt_oss")
    check(gpt.is_moe and gpt.n_experts == 32 and gpt.n_experts_active == 4,
          "GPT-OSS MoE 32 experts / 4 active")
    check(gpt.moe_quant == "mxfp4", "GPT-OSS experts mxfp4 (Phase 0c precision gate)")
    check(gpt.base_repo is None and gpt.teacher is None, "GPT-OSS has no base / no teacher")
    check(gpt.cot_format == reg.CoTFormat.HARMONY_ANALYSIS, "GPT-OSS CoT == harmony analysis")

    for k, mt, base, nL, hid in [
        ("ds_r1_llama8b", "llama", "meta-llama/Llama-3.1-8B", 32, 4096),
        ("ds_r1_qwen14b", "qwen2", "Qwen/Qwen2.5-14B", 48, 5120),
    ]:
        s = reg.get(k)
        check(s.provenance == reg.Provenance.DISTILLED_R1, f"{k} = distilled-R1")
        check(s.expected_model_type == mt, f"{k} model_type == {mt}")
        check(s.base_repo == base, f"{k} base == {base} (general, Phase-0a verified)")
        check(s.n_layers == nL and s.hidden == hid, f"{k} geometry {nL}L/{hid}")
        check(s.teacher == "deepseek-ai/DeepSeek-R1", f"{k} teacher == DeepSeek-R1 (shared)")
        check(s.cot_format == reg.CoTFormat.THINK_TAGS, f"{k} CoT == <think> tags")
        check(not s.is_moe, f"{k} is dense (only GPT-OSS is MoE)")

    # Shared teacher across the two distills -> 3c transfer probe is well-posed.
    check(reg.get("ds_r1_llama8b").teacher == reg.get("ds_r1_qwen14b").teacher,
          "both distills share the DeepSeek-R1 teacher (3c shared-teacher probe)")
    # Shairah comparator base kept OUT of the panel (measurement-only).
    check("shairah" not in [s.key for s in specs],
          "Shairah comparator not in PANEL_ORDER (measurement-only)")
    check(reg.SHAIRAH_COMPARATOR_BASE == "meta-llama/Llama-3.1-8B-Instruct",
          "Shairah comparator base == Paper 6 anchor instruct")


def test_assert_matches() -> None:
    print("\n[4] assert_matches_model fails loud on drift")
    gpt = reg.get("gpt_oss_20b")
    qwen = reg.get("ds_r1_qwen14b")

    def raises(fn) -> bool:
        try:
            fn()
            return False
        except RuntimeError:
            return True

    check(raises(lambda: gpt.assert_matches_model(99)), "raises on layer drift")
    check(raises(lambda: gpt.assert_matches_model(24, 9999)), "raises on hidden drift")
    check(raises(lambda: qwen.assert_matches_model(48, 5120, model_type_live="llama")),
          "raises on architecture mismatch (wrong base family)")
    check(raises(lambda: gpt.assert_matches_model(24, 2880, n_experts_live=8)),
          "raises on MoE expert-count drift")
    # Correct geometry must NOT raise.
    gpt.assert_matches_model(24, 2880, model_type_live="gpt_oss", n_experts_live=32)
    qwen.assert_matches_model(48, 5120, model_type_live="qwen2")
    check(True, "passes on the Phase-0a-verified geometry")


def test_tooling_imports() -> None:
    print("\n[5] shared Paper 6 / Paper 5 / Paper 3 tooling imports cleanly")
    check(reg.reg6 is not None and hasattr(reg.reg6, "ModelSpec"),
          "Paper 6 registry loaded by path (reuse, not fork)")
    try:
        import direction_utils as du  # noqa: F401
        check(hasattr(du, "cosine") and hasattr(du, "transfer_metrics"),
              "direction_utils (cosine / transfer_metrics) ok")
        check(hasattr(du, "load_directions") and hasattr(du, "save_directions"),
              "direction_utils npz load/save present")
    except Exception as e:  # noqa: BLE001
        check(False, f"import direction_utils: {e}")
    try:
        import extract_refusal as er  # noqa: F401  (Paper 6 — reused verbatim in Phase 1)
        check(hasattr(er, "linear_separability_gap"), "extract_refusal.linear_separability_gap ok")
        check(hasattr(er, "consolidation_at_layer"), "extract_refusal.consolidation_at_layer ok")
        check(hasattr(er, "subspace_projection_fraction"),
              "extract_refusal.subspace_projection_fraction ok")
    except Exception as e:  # noqa: BLE001
        check(False, f"import extract_refusal: {e}")
    try:
        import measure_refusal_decomposition as mrd  # noqa: F401
        check(hasattr(mrd, "decompose_layer"), "measure_refusal_decomposition.decompose_layer ok")
    except Exception as e:  # noqa: BLE001
        check(False, f"import measure_refusal_decomposition: {e}")
    try:
        from moral_dependency import build_subspace_basis  # noqa: F401
        check(True, "moral_dependency.build_subspace_basis importable")
    except Exception as e:  # noqa: BLE001
        check(False, f"import build_subspace_basis: {e}")
    check((_P3_SCRIPTS / "exp1_2_3_framework_geometry.py").exists(),
          "Paper 3 exp1 producer (exp1_2_3_framework_geometry.py) present")
    check((_P6_SCRIPTS / "random_ablation_control.py").exists(),
          "Paper 6 random_ablation_control.py present (reused in Phase 3b)")


def test_contrast_sets() -> None:
    print("\n[6] real contrast sets present (not _FALLBACK_ placeholders)")
    refusal = _REPO / "papers" / "5_moral_alignment" / "refusal_prompts.json"
    check(refusal.exists(), "Arditi/Heretic refusal_prompts.json present")
    if refusal.exists():
        d = json.loads(refusal.read_text())
        nh, ns = len(d.get("harmful", [])), len(d.get("harmless", []))
        check(nh >= 50 and ns >= 50,
              f"refusal set is real (harmful={nh}, harmless={ns}, not the 5-item fallback)")
    moral = _REPO / "deepsteer" / "datasets" / "moral_probing_v2.json"
    check(moral.exists(), "MFT moral_probing_v2.json present")


def main() -> int:
    print("Paper 7 Phase 0b local gate (reasoning-model registry)")
    print("=" * 64)
    print("registry dump:")
    for s in reg.all_specs():
        moe = f"MoE {s.n_experts}/{s.n_experts_active}" if s.is_moe else "dense"
        print(f"  {s.key:14s} {s.family:8s} {s.n_layers}L hid={s.hidden} "
              f"band={list(s.band)} primary=L{s.primary_layer} {moe} "
              f"prov={s.provenance.value}")
    test_band_math()
    test_validity_anchor()
    test_specs()
    test_assert_matches()
    test_tooling_imports()
    test_contrast_sets()
    print("\n" + "=" * 64)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for m in _FAILS:
            print(f"  - {m}")
        return 1
    print("ALL CHECKS PASSED — local gate clear. Next is the Phase 0c GPU smoke +")
    print("validity anchor + GPT-OSS precision gate, which need the human gate first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
