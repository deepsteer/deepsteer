#!/usr/bin/env python3
"""Phase 2 deliverable: cross-model comprehension-dissociation table.

For each family, reads the instruct vs Heretic-ablated comprehension battery and
computes the discriminator: does ablating refusal damage comprehension?

  per state (instruct / ablated):
    moral_probe_acc   mean over foundations of the per-foundation peak
                      fresh_probe_acc        (pipeline_study/moral_probing.json)
    eff_dim_layer     framework eff-dim at the headline layer (geometry.json)
    moral_judgment    behavioral overall_accuracy (behavioral_baseline.json)
    compliance        persona-shift baseline compliance (behavioral_baseline.json)
    dependency        moral_dependency_score (moral_dependency.json)
  delta = instruct - ablated  (comprehension columns): ~0 => clean dissociation
  compliance_gain = ablated - instruct: > 0 confirms refusal was stripped

Reads outputs/{key}/battery/{instruct,ablated}/*.json and
outputs/{key}/heretic/refusal_morality_geometry.json. Missing models are skipped.

Usage:
    python papers/6_cross_model/scripts/phase2_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as reg  # noqa: E402

_OUT = Path(__file__).resolve().parent.parent / "outputs"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _moral_probe_acc(state_dir: Path) -> float | None:
    """Mean over foundations of the per-foundation peak fresh_probe_acc."""
    mp = _load(state_dir / "moral_probing.json")
    if not mp or "per_foundation" not in mp:
        return None
    accs = [max((v.get("fresh_probe_acc", 0) for v in per_layer.values()), default=0)
            for per_layer in mp["per_foundation"].values()]
    return round(sum(accs) / len(accs), 4) if accs else None


def _eff_dim(state_dir: Path, layer: int) -> float | None:
    g = _load(state_dir / "geometry.json")
    if not g or "per_layer" not in g:
        return None
    at = g["per_layer"].get(str(layer), {}).get("effective_dimensionality")
    return at


def _behavioral(state_dir: Path) -> tuple[float | None, float | None]:
    """(moral_judgment_overall_accuracy, persona_shift baseline compliance)."""
    bb = _load(state_dir / "behavioral_baseline.json")
    if not bb:
        return None, None
    res = bb.get("results", {})
    mj = (res.get("moral_foundations") or {}).get("overall_accuracy")
    ps = res.get("persona_shift") or {}
    # PersonaShiftDetector reports baseline compliance under a few possible keys.
    comp = (ps.get("baseline_compliance_rate")
            or ps.get("baseline_compliance")
            or ps.get("compliance_rate"))
    return mj, comp


def _dependency(state_dir: Path) -> float | None:
    dep = _load(state_dir / "moral_dependency.json")
    if not dep:
        return None
    return (dep.get("metrics") or {}).get("moral_dependency_score")


def _state_metrics(state_dir: Path, layer: int) -> dict:
    mj, comp = _behavioral(state_dir)
    return {
        "moral_probe_acc": _moral_probe_acc(state_dir),
        "eff_dim_layer": _eff_dim(state_dir, layer),
        "moral_judgment": mj,
        "compliance": comp,
        "dependency": _dependency(state_dir),
    }


def _delta(a, b):
    return round(a - b, 4) if (a is not None and b is not None) else None


def collect(spec: reg.ModelSpec) -> dict | None:
    battery = _OUT / spec.key / "battery"
    instruct = _state_metrics(battery / "instruct", spec.primary_layer)
    ablated = _state_metrics(battery / "ablated", spec.primary_layer)
    if all(v is None for v in instruct.values()) and all(v is None for v in ablated.values()):
        return None
    geom = _load(_OUT / spec.key / "heretic" / "refusal_morality_geometry.json") or {}
    comp_keys = ("moral_probe_acc", "eff_dim_layer", "moral_judgment", "dependency")
    return {
        "key": spec.key, "headline_layer": spec.primary_layer,
        "instruct": instruct, "ablated": ablated,
        "comprehension_delta": {k: _delta(instruct[k], ablated[k]) for k in comp_keys},
        "compliance_gain": _delta(ablated["compliance"], instruct["compliance"]),
        "refusal_moral_proj_frac": geom.get("moral_subspace_projection_fraction"),
        "refusal_mean_abs_cos": geom.get("mean_abs_cosine"),
    }


def _f(v) -> str:
    if v is None:
        return "  n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_table(rows: list[dict]) -> None:
    keys = [r["key"] for r in rows]
    hdr = f"{'metric':<26s}" + "".join(f"{k:>14s}" for k in keys)
    print("\n" + "=" * len(hdr))
    print("Phase 2: comprehension dissociation (instruct vs Heretic-ablated)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    def line(label, getter):
        print(f"{label:<26s}" + "".join(f"{_f(getter(r)):>14s}" for r in rows))

    print("- refusal ablation -")
    line("  refusal moral proj frac", lambda r: r["refusal_moral_proj_frac"])
    line("  compliance: instruct", lambda r: r["instruct"]["compliance"])
    line("  compliance: ablated", lambda r: r["ablated"]["compliance"])
    line("  compliance gain (strip)", lambda r: r["compliance_gain"])
    print("- comprehension: instruct -> ablated (delta) -")
    for k, lab in [("moral_probe_acc", "probe acc"), ("eff_dim_layer", "eff-dim"),
                   ("moral_judgment", "moral judgment"), ("dependency", "dependency")]:
        line(f"  {lab}: instruct", lambda r, k=k: r["instruct"][k])
        line(f"  {lab}: ablated", lambda r, k=k: r["ablated"][k])
        line(f"  {lab}: DELTA", lambda r, k=k: r["comprehension_delta"][k])


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 6 Phase 2 comprehension-dissociation table")
    ap.parse_args()
    rows = []
    for spec in reg.all_specs():
        r = collect(spec)
        if r is None:
            print(f"(no Phase-2 outputs for {spec.key}; skipping)")
            continue
        rows.append(r)
    if not rows:
        print("No Phase-2 outputs under papers/6_cross_model/outputs/. Run Phase 2 first.")
        return
    print_table(rows)
    out = _OUT / "phase2_cross_model.json"
    out.write_text(json.dumps({"models": rows}, indent=2))
    print(f"\nWrote {out}")
    print("\nRead: comprehension DELTAs ~0 => clean dissociation (refusal not "
          "comprehension-load-bearing); a large positive delta => ablating refusal "
          "damaged comprehension in that family.")


if __name__ == "__main__":
    main()
