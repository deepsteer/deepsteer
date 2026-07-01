#!/usr/bin/env python3
"""Tier 2 / Task 2.2: malleability analysis (Measurements 2, 3, 4).

Pure-analysis pass (no model). Combines the freshly-extracted proto-refusal
contrasts (Task 2.1, ``outputs/measurement/stage3/<label>/``) with the cached
per-checkpoint MFT + persona directions (``outputs/pipeline/<label>/
probe_directions.npz``, raw, base-matched) and the back-transferred Instruct
persona (``outputs/olmo3_instruct/persona_directions.npz``).

  * Measurement 2 (malleability curve): proto-refusal projection onto each
    checkpoint's own MFT subspace vs. step. A window where the projection RISES
    (proto-refusal becomes less orthogonal to MFT) is the candidate intervention
    checkpoint.
  * Measurement 3: persona <-> proto-refusal alignment across checkpoints, both
    for the fresh per-checkpoint toxic-voice persona and the back-transferred
    Instruct persona. Tests whether persona carries refusal before SFT wires the
    gate. (Tier-1 M1 already found refusal ~orthogonal to persona at Instruct;
    this checks the pre-training trajectory.)
  * Measurement 4 (anti-artifact baseline): proto-refusal->MFT projection at base
    (== final pre-training). A FUTURE post-intervention projection >= 0.40 only
    means "moved refusal into morality" if this baseline is well below 0.40 to
    begin with. Reported explicitly with the threshold.

Projection convention matches Tier-1 M1 and the dependency metric: the MFT
subspace is the SVD-orthonormal span of the 6 probe foundation directions
(``build_subspace_basis``); ``proj_fraction`` is the norm-ratio
(``heretic_ablation.subspace_projection_fraction``, the §4.4 number) and
``proj_energy`` = its square.

Output: ``outputs/measurement/malleability_summary.json`` + a 2-panel figure
``outputs/measurement/malleability_curve.{png,pdf}`` in the Paper 5 style.

Usage:
    python papers/5_moral_alignment/scripts/malleability_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepsteer.directions import extraction as du  # noqa: E402
from dependency_figures import _save, _style, short_label  # noqa: E402
from heretic_ablation import subspace_projection_fraction  # noqa: E402
from moral_dependency import build_subspace_basis  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_DEF_STAGE3 = _PAPER_ROOT / "outputs/measurement/stage3"
_DEF_PIPELINE = _PAPER_ROOT / "outputs/pipeline"
_DEF_INSTRUCT_PERSONA = _PAPER_ROOT / "outputs/olmo3_instruct/persona_directions.npz"
_DEF_GRID = _PAPER_ROOT / "checkpoint_grid.json"
_DEF_OUT = _PAPER_ROOT / "outputs/measurement"
_COUPLING_THRESHOLD = 0.40  # §6.3: below this, ablating refusal can't damage comprehension


def _step(label: str) -> int:
    if label == "olmo3_base":
        return -1  # plotted as the M4 baseline, not on the step axis
    import re
    m = re.search(r"step(\d+)", label)
    return int(m.group(1)) if m else 10**9


def _band_mean(per_layer: dict[int, float], band: list[int]) -> float:
    vals = [per_layer[L] for L in band if L in per_layer]
    return round(float(np.mean(vals)), 6) if vals else float("nan")


def analyse_state(label, stage3_dir, pipeline_dir, persona_bt, band, headline):
    """Per-state M2/M3 numbers, or None if the proto-refusal npz is missing."""
    proto_path = Path(stage3_dir) / label / "proto_refusal_directions.npz"
    pipe_path = Path(pipeline_dir) / label / "probe_directions.npz"
    if not proto_path.exists():
        logger.warning("[%s] no proto_refusal_directions.npz; skipping", label)
        return None
    if not pipe_path.exists():
        logger.warning("[%s] no pipeline probe_directions.npz; skipping", label)
        return None

    proto = du.load_directions(proto_path)["proto_refusal"]
    pipe = du.load_directions(pipe_path)
    foundations = [f for f in FOUNDATION_ORDER if f in pipe]
    persona_fresh = pipe.get("persona", {})
    n_layers = max(proto) + 1
    basis_by_layer, _, _ = build_subspace_basis(pipe, kind="probe", n_layers=n_layers)

    proj_norm: dict[int, float] = {}
    cos_fresh: dict[int, float] = {}
    cos_bt: dict[int, float] = {}
    for L in sorted(proto):
        if L in basis_by_layer:
            basis = [pipe[f][L] for f in foundations if L in pipe[f]]
            proj_norm[L] = subspace_projection_fraction(proto[L], basis)
        if L in persona_fresh:
            cos_fresh[L] = du.cosine(proto[L], persona_fresh[L])
        if persona_bt is not None and L in persona_bt:
            cos_bt[L] = du.cosine(proto[L], persona_bt[L])

    return {
        "label": label,
        "step": _step(label),
        "proj_norm_headline": round(proj_norm.get(headline, float("nan")), 6),
        "proj_energy_headline": round(proj_norm.get(headline, 0.0) ** 2, 6),
        "proj_norm_band_mean": _band_mean(proj_norm, band),
        "cos_persona_fresh_headline": round(cos_fresh.get(headline, float("nan")), 6),
        "cos_persona_fresh_band_mean": _band_mean({L: abs(v) for L, v in cos_fresh.items()}, band),
        "cos_persona_bt_headline": round(cos_bt.get(headline, float("nan")), 6)
        if cos_bt else None,
        "cos_persona_bt_band_mean": _band_mean({L: abs(v) for L, v in cos_bt.items()}, band)
        if cos_bt else None,
        "per_layer_proj_norm": {str(L): round(v, 6) for L, v in sorted(proj_norm.items())},
    }


def make_figure(stage3_rows, base_row, band, headline, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    x = [r["step"] / 1000.0 for r in stage3_rows]
    xt = [short_label(r["label"]) for r in stage3_rows]

    # --- M2: proto-refusal -> MFT projection vs step ---
    ax1.plot(x, [r["proj_norm_band_mean"] for r in stage3_rows], "o-",
             color="#1565C0", lw=1.8, ms=5, label=f"band {band[0]}-{band[-1]} mean")
    ax1.plot(x, [r["proj_norm_headline"] for r in stage3_rows], "s--",
             color="#6A1B9A", lw=1.4, ms=4, label=f"layer {headline}")
    ax1.axhline(_COUPLING_THRESHOLD, color="#C62828", ls=":", lw=1.3,
                label=f"coupling threshold {_COUPLING_THRESHOLD:g}")
    if base_row is not None:
        ax1.axhline(base_row["proj_norm_band_mean"], color="0.45", ls="--", lw=1,
                    label="base (M4 baseline)")
    ax1.set_ylabel("proto-refusal -> MFT\nprojection fraction")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", ncol=2)
    ax1.set_title("Malleability: how MFT-shaped is the proto-refusal contrast "
                  "across stage-3 pre-training")

    # --- M3: persona <-> proto-refusal alignment vs step ---
    ax2.plot(x, [r["cos_persona_fresh_band_mean"] for r in stage3_rows], "o-",
             color="#2E7D32", lw=1.8, ms=5, label="fresh per-ckpt persona |cos|")
    if any(r["cos_persona_bt_band_mean"] is not None for r in stage3_rows):
        ax2.plot(x, [r["cos_persona_bt_band_mean"] for r in stage3_rows], "^--",
                 color="#EF6C00", lw=1.4, ms=4, label="back-transferred Instruct persona |cos|")
    ax2.set_ylabel("persona <-> proto-refusal\n|cosine|")
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("stage-3 step (thousands)")
    ax2.legend(loc="upper left")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xt, rotation=45, ha="right")

    fig.tight_layout()
    _save(fig, Path(out_dir), "malleability_curve")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Malleability analysis (M2/M3/M4).")
    ap.add_argument("--stage3-dir", default=str(_DEF_STAGE3))
    ap.add_argument("--pipeline-dir", default=str(_DEF_PIPELINE))
    ap.add_argument("--instruct-persona-npz", default=str(_DEF_INSTRUCT_PERSONA))
    ap.add_argument("--grid", default=str(_DEF_GRID))
    ap.add_argument("--headline-layer", type=int, default=16)
    ap.add_argument("--band", type=int, nargs=2, default=[15, 31])
    ap.add_argument("--output-dir", default=str(_DEF_OUT))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    band = list(range(args.band[0], args.band[1] + 1))

    grid = json.load(open(args.grid))
    labels = [g["label"] for g in grid
              if "pretrain_stage3" in g["label"] or g["label"] == "olmo3_base"]

    persona_bt = None
    if Path(args.instruct_persona_npz).exists():
        persona_bt = du.load_directions(args.instruct_persona_npz).get("persona")
    else:
        logger.warning("instruct persona npz missing (%s); back-transfer skipped",
                       args.instruct_persona_npz)

    rows = []
    for lab in labels:
        r = analyse_state(lab, args.stage3_dir, args.pipeline_dir, persona_bt,
                          band, args.headline_layer)
        if r is not None:
            rows.append(r)
    if not rows:
        raise RuntimeError("No states analysed; run extract_proto_refusal.py first.")

    base_row = next((r for r in rows if r["label"] == "olmo3_base"), None)
    stage3_rows = sorted([r for r in rows if r["label"] != "olmo3_base"],
                         key=lambda r: r["step"])

    # M2: intervention window = max projection across stage-3.
    window = max(stage3_rows, key=lambda r: r["proj_norm_band_mean"]) if stage3_rows else None

    # M4: anti-artifact interpretation, anchored on the base baseline.
    baseline_src = base_row or (stage3_rows[-1] if stage3_rows else {})
    baseline_proj = baseline_src.get("proj_norm_band_mean")
    m4 = {
        "baseline_state": "olmo3_base" if base_row else (
            stage3_rows[-1]["label"] if stage3_rows else None),
        "baseline_proj_norm_band_mean": baseline_proj,
        "coupling_threshold": _COUPLING_THRESHOLD,
        "baseline_below_threshold": (baseline_proj is not None
                                     and baseline_proj < _COUPLING_THRESHOLD),
        "interpretation": (
            "A future post-intervention proto-refusal->MFT projection >= 0.40 "
            "means 'moved refusal into morality' ONLY because the baseline "
            f"projection ({baseline_proj}) is well below 0.40, i.e. the "
            "pre-intervention contrast is mostly orthogonal to MFT."),
    }

    payload = {
        "analysis": "malleability_summary",
        "headline_layer": args.headline_layer,
        "band": [args.band[0], args.band[1]],
        "n_stage3_analysed": len(stage3_rows),
        "measurement_2_intervention_window": {
            "label": window["label"] if window else None,
            "step": window["step"] if window else None,
            "proj_norm_band_mean": window["proj_norm_band_mean"] if window else None,
            "note": "Highest proto-refusal->MFT projection across stage-3; the "
                    "least-orthogonal checkpoint, candidate intervention point.",
        },
        "measurement_4_anti_artifact": m4,
        "base": base_row,
        "stage3_trajectory": stage3_rows,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "malleability_summary.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {out/'malleability_summary.json'} ({len(stage3_rows)} stage-3 states)")
    if window:
        print(f"  M2 intervention window: {window['label']} "
              f"(proj {window['proj_norm_band_mean']:.4f})")
    print(f"  M4 baseline proj ({m4['baseline_state']}): {baseline_proj} "
          f"(< {_COUPLING_THRESHOLD}: {m4['baseline_below_threshold']})")

    if not args.no_figure and stage3_rows:
        make_figure(stage3_rows, base_row, band, args.headline_layer, out)
        print(f"  wrote {out/'malleability_curve.png'}")


if __name__ == "__main__":
    main()
