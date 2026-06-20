#!/usr/bin/env python3
"""Phase 1 gate deliverable: combined cross-model decomposition table + figure.

Reads each family's Phase-1 artifacts and assembles the one table the human gate
needs (does the refusal/moral decomposition replicate across families?):

  per model, at the depth-0.5 headline layer (+ stable-band mean):
    * refusal energy partition: mft_frac / persona_unique_frac / residual_frac
    * refusal<->morality: mean|cos| to foundations, moral-subspace projection frac
    * consolidation: margin d', single-direction AUC, across-band eff-rank / PR
    * the §6.4-style verdict string

Inputs (per key in the registry):
  outputs/{key}/refusal_decomposition.json        (measure_refusal_decomposition)
  outputs/{key}_instruct/refusal_extraction.json  (extract_refusal)

Writes outputs/phase1_cross_model.json (the figure's data source, depth-fraction
x-axis so layers align across 28/32-layer models) and, if matplotlib is present,
outputs/phase1_cross_model.pdf/.png. Missing models are skipped with a note, so
this runs after a partial pass (e.g. Llama gated out).

Usage:
    python papers/6_cross_model/scripts/phase1_table.py
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


def collect(spec: reg.ModelSpec) -> dict | None:
    decomp = _load(_OUT / spec.key / "refusal_decomposition.json")
    extract = _load(_OUT / spec.instruct_out / "refusal_extraction.json")
    if decomp is None and extract is None:
        return None
    row: dict = {"key": spec.key, "family": spec.family, "n_layers": spec.n_layers,
                 "headline_layer": spec.primary_layer, "band": list(spec.band),
                 "depth_headline": round(spec.primary_layer / spec.n_layers, 3)}
    if decomp:
        hd = decomp.get("headline", {})
        bs = decomp.get("band_summary", {})
        row["decomp"] = {
            "mft_frac": hd.get("mft_frac"),
            "persona_unique_frac": hd.get("persona_unique_frac"),
            "residual_frac": hd.get("residual_frac"),
            "cos_refusal_persona": hd.get("cos_refusal_persona"),
            "mean_abs_cos_to_foundations": hd.get("mean_abs_cos_to_foundations"),
            "band_mft_frac": bs.get("mft_frac_mean"),
            "band_residual_frac": bs.get("residual_frac_mean"),
            "verdict": (decomp.get("verdict") or {}).get("target"),
        }
        # per-layer for the figure (depth fraction x mft/residual fraction)
        per = decomp.get("per_layer", {})
        row["per_layer"] = {
            "depth": [round(int(L) / spec.n_layers, 4) for L in sorted(per, key=int)],
            "mft_frac": [per[L]["mft_frac"] for L in sorted(per, key=int)],
            "residual_frac": [per[L]["residual_frac"] for L in sorted(per, key=int)],
        }
    if extract:
        geom = extract.get("refusal_morality_geometry") or {}
        cons = extract.get("consolidation") or {}
        hd = cons.get("headline") or {}
        ab = cons.get("across_band_refusal_directions") or {}
        row["consolidation"] = {
            "margin_dprime": hd.get("margin_dprime"),
            "single_dir_auc": hd.get("single_dir_auc"),
            "single_dir_auc_cv": hd.get("single_dir_auc_cv"),
            "full_auc_cv": hd.get("full_auc_cv"),
            "auc_gap": hd.get("auc_gap"),
            "band_mean_auc_gap": cons.get("band_mean_auc_gap"),
            "across_band_effrank": ab.get("effrank_90pct_uncentered"),
            "across_band_PR": ab.get("participation_ratio"),
            "geom_mean_abs_cos": geom.get("mean_abs_cosine"),
            "geom_subspace_frac": geom.get("moral_subspace_projection_fraction"),
        }
    return row


def _fmt(v) -> str:
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    s = str(v)
    return s if len(s) <= 13 else s[:10] + "..."


def print_table(rows: list[dict]) -> None:
    keys = [r["key"] for r in rows]
    print("\n" + "=" * (22 + 14 * len(keys)))
    print("Cross-model refusal decomposition (headline = depth-0.5 layer)")
    print("=" * (22 + 14 * len(keys)))
    hdr = f"{'metric':<28s}" + "".join(f"{k:>14s}" for k in keys)
    print(hdr)
    print("-" * len(hdr))

    def line(label: str, getter) -> None:
        print(f"{label:<28s}" + "".join(f"{_fmt(getter(r)):>14s}" for r in rows))

    line("headline layer", lambda r: r["headline_layer"])
    line("depth fraction", lambda r: r["depth_headline"])
    print("- refusal energy partition -")
    line("  mft_frac", lambda r: r.get("decomp", {}).get("mft_frac"))
    line("  persona_unique_frac", lambda r: r.get("decomp", {}).get("persona_unique_frac"))
    line("  residual_frac", lambda r: r.get("decomp", {}).get("residual_frac"))
    line("  band residual_frac", lambda r: r.get("decomp", {}).get("band_residual_frac"))
    print("- refusal <-> morality -")
    line("  mean|cos| foundations",
         lambda r: r.get("decomp", {}).get("mean_abs_cos_to_foundations"))
    line("  subspace proj frac", lambda r: r.get("consolidation", {}).get("geom_subspace_frac"))
    line("  cos(refusal,persona)", lambda r: r.get("decomp", {}).get("cos_refusal_persona"))
    print("- consolidation -")
    line("  margin d' (headline)", lambda r: r.get("consolidation", {}).get("margin_dprime"))
    line("  single-dir AUC (cv)", lambda r: r.get("consolidation", {}).get("single_dir_auc_cv"))
    line("  full-rank AUC (cv)", lambda r: r.get("consolidation", {}).get("full_auc_cv"))
    line("  AUC gap (full-single)", lambda r: r.get("consolidation", {}).get("auc_gap"))
    line("  across-band eff-rank", lambda r: r.get("consolidation", {}).get("across_band_effrank"))
    line("  across-band PR", lambda r: r.get("consolidation", {}).get("across_band_PR"))
    print("- verdict -")
    line("  target", lambda r: r.get("decomp", {}).get("verdict"))
    print("=" * len(hdr))


def make_figure(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(matplotlib unavailable, skipping figure: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in rows:
        pl = r.get("per_layer")
        if not pl:
            continue
        axes[0].plot(pl["depth"], pl["mft_frac"], marker=".", label=r["key"])
        axes[1].plot(pl["depth"], pl["residual_frac"], marker=".", label=r["key"])
    axes[0].set_title("refusal energy in moral subspace")
    axes[0].set_ylabel("mft_frac")
    axes[1].set_title("refusal residual energy")
    axes[1].set_ylabel("residual_frac")
    for ax in axes:
        ax.set_xlabel("depth fraction (layer / n_layers)")
        ax.axvline(0.5, color="grey", ls=":", lw=0.8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print(f"  figure -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 6 Phase 1 cross-model table + figure")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    rows = []
    for spec in reg.all_specs():
        r = collect(spec)
        if r is None:
            print(f"(no Phase-1 outputs for {spec.key}; skipping)")
            continue
        rows.append(r)
    if not rows:
        print("No Phase-1 outputs found under papers/6_cross_model/outputs/. Run Phase 1 first.")
        return

    print_table(rows)
    out_json = _OUT / "phase1_cross_model.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"models": rows}, indent=2))
    print(f"\nWrote {out_json}")
    if not args.no_figure:
        make_figure(rows, _OUT / "phase1_cross_model.pdf")


if __name__ == "__main__":
    main()
