#!/usr/bin/env python3
"""Phase 0b local gate: no-GPU checks that must pass before any RunPod work.

Verifies the three things that make the cross-model comparison valid without
loading a single model:

  1. The fractional band/layer rule reproduces the OLMo anchor (15..31 / L16) and
     maps sensibly onto the 28-layer Qwen and 32-layer Llama.
  2. Extraction conventions are pinned and IDENTICAL across the panel (base=raw,
     refusal=chat; same direction tooling).
  3. The shared Paper 5 / Paper 3 tooling imports cleanly from the Paper 6 script
     dir, and the real contrast sets (Arditi refusal prompts, MFT probing v2)
     are present, not the ``_FALLBACK_*`` placeholders.

Run:
    python papers/6_cross_model/scripts/local_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PAPER6 = _THIS.parent.parent
_REPO = _PAPER6.parent.parent
_PAPER5_SCRIPTS = _REPO / "papers" / "5_moral_alignment" / "scripts"
_PAPER3_SCRIPTS = _REPO / "papers" / "3_moral_geometry" / "scripts"

sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_PAPER5_SCRIPTS))

import model_registry as reg  # noqa: E402

_FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _FAILS.append(msg)


def test_band_math() -> None:
    print("\n[1] fractional band / primary-layer mapping")
    # Anchor must reproduce Paper 5 exactly.
    check(reg.band_layers(32) == (15, 31), "OLMo 32L band == (15, 31) [Paper 5 Appendix B]")
    check(reg.primary_layer(32) == 16, "OLMo 32L primary == L16 (depth 0.5)")
    # Qwen 28L and Llama 32L mappings.
    check(reg.band_layers(28) == (13, 27), "Qwen 28L band == (13, 27)")
    check(reg.primary_layer(28) == 14, "Qwen 28L primary == L14")
    check(reg.band_layers(32) == (15, 31), "Llama 32L band == (15, 31)")
    # Top edge always lands on the last layer regardless of count.
    for n in (28, 32, 40, 24):
        lo, hi = reg.band_layers(n)
        check(hi == n - 1, f"{n}L band top == last layer ({n - 1})")
        check(0 < lo < reg.primary_layer(n) < hi, f"{n}L: 0 < lo < primary < hi monotone")


def test_specs() -> None:
    print("\n[2] per-model specs + pinned conventions")
    specs = reg.all_specs()
    check(len(specs) == 3, "panel has 3 families")
    check([s.key for s in specs] == ["olmo3", "qwen25", "llama31"], "anchor-first order")
    for s in specs:
        check(s.input_format_base == "raw", f"{s.key}: base moral/persona format == raw")
        check(s.input_format_refusal == "chat", f"{s.key}: refusal format == chat")
        check(s.band[0] < s.band[1], f"{s.key}: band well-ordered {list(s.band)}")
    olmo = reg.get("olmo3")
    check(olmo.full_attention_layers == [3, 7, 11, 15, 19, 23, 27, 31],
          "OLMo-3 full-attention layers == every 4th")
    check(reg.get("qwen25").full_attention_layers is None, "Qwen full-attn annotation None")
    check(reg.get("llama31").full_attention_layers is None, "Llama full-attn annotation None")
    check(reg.get("llama31").gated is True, "Llama-3.1 flagged gated")
    check(reg.get("qwen25").gated is False and olmo.gated is False, "OLMo/Qwen ungated")
    # assert_matches_model fails loud on drift.
    try:
        olmo.assert_matches_model(99)
        check(False, "assert_matches_model raises on layer drift")
    except RuntimeError:
        check(True, "assert_matches_model raises on layer drift")
    olmo.assert_matches_model(32, 4096)  # must not raise
    check(True, "assert_matches_model passes on expected geometry")


def test_tooling_imports() -> None:
    print("\n[3] shared Paper 5 / Paper 3 tooling imports cleanly")
    try:
        import direction_utils as du  # noqa: F401
        check(hasattr(du, "extract_pair_directions"), "du.extract_pair_directions ok")
        check(hasattr(du, "effective_dimensionality"), "du.effective_dimensionality ok")
        check(hasattr(du, "load_directions") and hasattr(du, "save_directions"),
              "direction_utils npz load/save present")
    except Exception as e:  # noqa: BLE001
        check(False, f"import direction_utils: {e}")
    try:
        from moral_dependency import build_subspace_basis  # noqa: F401
        check(True, "moral_dependency.build_subspace_basis importable")
    except Exception as e:  # noqa: BLE001
        check(False, f"import build_subspace_basis: {e}")
    try:
        import measure_refusal_decomposition as mrd  # noqa: F401
        check(hasattr(mrd, "decompose_layer"), "measure_refusal_decomposition.decompose_layer ok")
    except Exception as e:  # noqa: BLE001
        check(False, f"import measure_refusal_decomposition: {e}")
    check((_PAPER3_SCRIPTS / "exp1_2_3_framework_geometry.py").exists(),
          "Paper 3 exp1 producer (exp1_2_3_framework_geometry.py) present")


def test_contrast_sets() -> None:
    print("\n[4] real contrast sets present (not _FALLBACK_ placeholders)")
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
    print("Paper 6 Phase 0b local gate")
    print("=" * 60)
    print("registry dump:")
    for s in reg.all_specs():
        print(f"  {s.key:8s} {s.family:6s} {s.n_layers}L hid={s.hidden} "
              f"band={list(s.band)} primary=L{s.primary_layer} gated={s.gated}")
    test_band_math()
    test_specs()
    test_tooling_imports()
    test_contrast_sets()
    print("\n" + "=" * 60)
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for m in _FAILS:
            print(f"  - {m}")
        return 1
    print("ALL CHECKS PASSED — local gate clear; safe to proceed to Phase 0c (GPU smoke).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
