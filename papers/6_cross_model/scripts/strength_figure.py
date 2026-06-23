#!/usr/bin/env python3
"""Render the Llama refusal-ablation strength sweep (Phase 2d) as a dose-response
figure: moral judgment (with bootstrap CIs) + harmful refusal rate + probe acc vs
ablation strength alpha, with the magnitude-matched random null band overlaid.

Reads outputs/llama31/ablation_strength_sweep.json (1:1 with the figure).
Usage: python papers/6_cross_model/scripts/strength_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_OUT = Path(__file__).resolve().parent.parent / "outputs" / "llama31"


def main() -> None:
    src = _OUT / "ablation_strength_sweep.json"
    if not src.exists():
        print(f"missing {src}; run the strength sweep first.")
        sys.exit(1)
    d = json.loads(src.read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm = sorted(d["refusal_arm"], key=lambda r: r["alpha"])
    a = [r["alpha"] for r in arm]
    judg = [r["judgment"] for r in arm]
    # judgment 95% band from the paired bootstrap CI on the drop vs clean
    clean = d["clean_judgment"]
    j_lo = [clean - r["boot"]["ci95"][1] for r in arm]
    j_hi = [clean - r["boot"]["ci95"][0] for r in arm]
    refus = [r["refusal_rate"] for r in arm]
    probe = [r["probe_acc"] for r in arm]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.fill_between(a, j_lo, j_hi, color="C0", alpha=0.18, label="judgment 95% CI (bootstrap)")
    ax.plot(a, judg, "o-", color="C0", label="moral judgment")
    ax.plot(a, refus, "s--", color="C3", label="harmful refusal rate")
    ax.plot(a, probe, "^:", color="C2", label="fresh probe acc (representation)")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.text(max(a), 0.51, "chance", fontsize=8, color="grey", ha="right")

    for nl in d.get("random_null_arm", []):
        m, s = nl["mean"], nl["std"]
        ax.errorbar(nl["alpha"], m, yerr=s, fmt="D", color="0.4", ms=6, capsize=3,
                    label="matched-random null" if nl is d["random_null_arm"][0] else None)

    ax.set_xlabel("ablation strength  alpha  (fraction of refusal direction removed)")
    ax.set_ylabel("rate / accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Llama-3.1: moral judgment degrades with refusal removal, "
                 "direction-specifically", fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(_OUT / f"ablation_strength_sweep.{ext}", dpi=160, bbox_inches="tight")
    print(f"Wrote {_OUT}/ablation_strength_sweep.pdf/.png")


if __name__ == "__main__":
    main()
