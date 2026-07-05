#!/usr/bin/env python3
"""Build MANIFEST.json for the DeepSteer paper supplement.

Scans deepsteer/supplement/{figure_data,cells} and writes a manifest that
indexes every distilled artifact with a content hash and its metadata
(description, provenance, which paper figures/tables cite it, and schema for
CSVs). The manifest is the single index both papers cite.

This script does not touch raw activations. The distilled cell JSONs are copied
in once by the author from the (gitignored) run outputs; see PROVENANCE.md. Run
from anywhere:

    python3 deepsteer/supplement/scripts/build.py

Deterministic: no timestamps, no randomness, sorted keys, so re-running on an
unchanged tree reproduces byte-identical MANIFEST.json.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

# Per-artifact metadata, keyed by path relative to the supplement root.
# cited_by lists "paper:location" so a reviewer can walk from any figure to its
# numbers. produced_by names the run artifact or regen script the values come
# from. Shared artifacts are used by both papers and live here once.
META: dict[str, dict] = {
    "figure_data/bottleneck_pr.csv": {
        "description": "Participation ratio at the decision site by model, plus decision-token, content-position, and raw->standardized geometric-cell PRs. The control-token bottleneck profile.",
        "produced_by": "cells/olmo3_decision_anatomy.json + per-model D2 position-gate sessions",
        "shared": True,
        "cited_by": ["flagship:Fig bottleneck-PR (sec 5)", "methods-note:Fig 1 + Table 1 (sec 2)"],
    },
    "figure_data/depth_asymmetry.csv": {
        "description": "Read-vs-commit asymmetry statistic A by model and layer with bootstrap CIs; the read-layer value collapses to the depth-matched layer-12 value.",
        "produced_by": "cells/olmo3_depth_L12.json, cells/llama31_depth_L12.json",
        "shared": True,
        "cited_by": ["flagship:Fig depth-collapse (sec 8)", "methods-note:Fig 3 (sec 5)"],
    },
    "figure_data/calibration_ladder_permodel.csv": {
        "description": "Per-model calibrated ladder: moral-family band [min,max], refusal projection, persona reference, covariance-null q95. Every refusal point lands below its band.",
        "produced_by": "D1 calibration sessions (a1_ladder*.json)",
        "shared": False,
        "cited_by": ["flagship:Fig calibration-ladder (sec 6), Table 3 (App B)"],
    },
    "figure_data/calibration_ladder_position.csv": {
        "description": "Two-position validity ladder: at the decision site the positive-control band sits BELOW the covariance null (position-invalid for content); at a content position it sits above (valid).",
        "produced_by": "D2 position-gate session (OLMo-3-Instruct)",
        "shared": False,
        "cited_by": ["methods-note:Fig 2 (sec 2.1)"],
    },
    "figure_data/rank_sweep.csv": {
        "description": "Nested interchange rank sweep on OLMo-3 (n=23 request-twins): restricted-transfer R_judgment and R_refusal over k in {1,3,8,16}, random-direction null, and the one-knob harm-ceiling fit.",
        "produced_by": "cells/olmo3_rank_sweep.json",
        "shared": False,
        "cited_by": ["flagship:Fig one-knob (sec 7)"],
    },
    "figure_data/crystallization.csv": {
        "description": "Moral-subspace crystallization (checkpoint-to-final cosine 0.869->0.999, then a single SFT rotation to 0.757) against the flat proto-refusal->gate cosine 0.155 (single measurement, no per-checkpoint trajectory).",
        "produced_by": "D1 phase-2 base/instruct direction extraction",
        "shared": False,
        "cited_by": ["flagship:Fig crystallization (sec 3, sec 4)"],
    },
    "figure_data/reversibility.csv": {
        "description": "GPT-OSS graded exculpatory/inculpating prefill flips with Wilson CIs: inculpating benign->refuse 7/7, exculpating violating->comply 6/10 (the reversible-reader result).",
        "produced_by": "cells/gpt_oss_tier1.json (graded_disengage / deliberation)",
        "shared": False,
        "cited_by": ["flagship:Fig reversibility (sec 8.2)"],
    },
    "figure_data/head_attribution.csv": {
        "description": "OLMo-3 Stage-1 per-head write attribution at the decision channel: top-10 heads by channel-matched specificity with cumulative-specificity fraction (led by L16H23 at 11.7%; ~67 heads reach 80%; MLP write fraction 0.384).",
        "produced_by": "cells/olmo3_decision_anatomy.json (top_heads, sparsity_curve, mlp)",
        "shared": False,
        "cited_by": ["flagship:sec 7 + App C (per-head attribution)"],
    },
    "cells/olmo3_decision_anatomy.json": {
        "description": "OLMo-3 decisive-cell session: request-twin cells, transport positive control, channel-matched nulls, MDEs, ratio-of-ratios, per-head write attribution, behavioral generate-under-patch. Distilled summary; no raw activations.",
        "produced_by": "D3 C1 session (RMSNorm-folded, reconstruction 0.9999)",
        "shared": False,
        "cited_by": ["flagship:sec 6, sec 7, App C"],
    },
    "cells/olmo3_rank_sweep.json": {
        "description": "OLMo-3 one-knob rank-sweep coefficients: R_refusal/R_judgment/harm_capture by k, harm-rank-1 ceiling 0.3131, plateau RMSE 0.0356, PC1 causal-inertia, harm-partialed R 0.1597.",
        "produced_by": "D3 rank sweep",
        "shared": False,
        "cited_by": ["flagship:sec 7"],
    },
    "cells/olmo3_depth_L12.json": {
        "description": "OLMo-3 depth-matched (layer 12) decision-anatomy session for the read-vs-commit asymmetry recomputation.",
        "produced_by": "D3 depth-matched session",
        "shared": False,
        "cited_by": ["flagship:sec 8", "methods-note:sec 5"],
    },
    "cells/llama31_decision_anatomy.json": {
        "description": "Llama-3.1 decision-anatomy session: the broad-moral read (refusal transfer ~0.85 ~= judgment ~0.79 by interchange at matched depth).",
        "produced_by": "D3 C1 session (Llama-3.1)",
        "shared": False,
        "cited_by": ["flagship:sec 8"],
    },
    "cells/llama31_depth_L12.json": {
        "description": "Llama-3.1 depth-matched (layer 12) session; the read-layer asymmetry collapses to the depth-matched value (paired with olmo3_depth_L12).",
        "produced_by": "D3 depth-matched session (Llama-3.1)",
        "shared": False,
        "cited_by": ["flagship:sec 8", "methods-note:sec 5"],
    },
    "cells/gpt_oss_tier1.json": {
        "description": "GPT-OSS-20B Tier-1 session: harmony-decision-token position gate (PR 12.8), band-existence gate, consequential deliberation (benign->refuse 7/7), graded disengage (violating->comply 6/10, monotone projection). Tier-2 causal C1-MoE held.",
        "produced_by": "D3 Tier-1 session (Amendment 12)",
        "shared": False,
        "cited_by": ["flagship:sec 8.2"],
    },
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_schema(path: str) -> list[str]:
    with open(path, newline="") as f:
        return next(csv.reader(f))


def build() -> dict:
    artifacts = []
    missing_meta = []
    for sub in ("figure_data", "cells"):
        d = os.path.join(ROOT, sub)
        for name in sorted(os.listdir(d)):
            rel = f"{sub}/{name}"
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            meta = META.get(rel)
            if meta is None:
                missing_meta.append(rel)
                continue
            entry = {
                "path": rel,
                "bytes": os.path.getsize(p),
                "sha256": sha256(p),
                "description": meta["description"],
                "produced_by": meta["produced_by"],
                "shared_across_papers": meta.get("shared", False),
                "cited_by": meta["cited_by"],
            }
            if name.endswith(".csv"):
                entry["columns"] = csv_schema(p)
            artifacts.append(entry)
    if missing_meta:
        raise SystemExit(f"No META entry for: {missing_meta} (add it to build.py)")
    return {
        "supplement": "DeepSteer: refusal reads a slice of moral content — distilled artifacts",
        "policy": "Distilled artifacts only (per-head contributions, nulls, PR profiles, "
        "ladders, sweep outcomes). Raw activations are not included; they are "
        "available on reviewer request (see README.md).",
        "shared_artifacts_live_once": True,
        "artifact_count": len(artifacts),
        "artifacts": sorted(artifacts, key=lambda a: a["path"]),
    }


def main() -> int:
    manifest = build()
    out = os.path.join(ROOT, "MANIFEST.json")
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote MANIFEST.json: {manifest['artifact_count']} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
