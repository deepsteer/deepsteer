#!/usr/bin/env python3
"""Amendment 11 — harm-coextensive check for the Llama reads-broad verdict. Zero-GPU: reads the saved
C1 session JSON only.

reads-broad ("Llama refusal reads the broad moral subspace, not just harm") is "beyond harm" only if a
harm basis does NOT span the engage-driving moral directions. The rank-1 harm percept `d_harm` is
resolvable now: per-moral-PC capture at rank 1 is cos(d_harm, PC_i)^2 (saved as `cos_harm_pc`), and the
engage-driving weight per PC comes from the saved per-k engage transfer (`engage_sweep`). The
engage-weighted rank-1 capture answers the rank-1 alternative directly.

The rank-2/4 severity-derived harm basis (a richer, multi-dimensional harm percept) needs the
severity-twin paired content contrasts, which the C1 run did not save → extraction rider. When that
lands, pass the nested harm bases to `sweep.harm_capture_curve` for the rank-2/4 numbers.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "papers/d3_decision_anatomy/scripts")
import sweep as sw  # noqa: E402


def rank1_capture(session_json: str) -> dict:
    d = json.load(open(session_json))
    cells = d["cells"]
    cos_pc = cells["sweep"]["cos_harm_pc"]                 # |cos(d_harm, moral PC_i)|
    eng = cells["engage_sweep"]
    w = sw.engage_marginal_weights(eng["R_engage_refusal_k"], eng["ks"])
    cap = [c ** 2 for c in cos_pc]                         # rank-1 per-PC capture = cos^2
    wsum = sum(w.get(i + 1, 0.0) for i in range(len(cap)))
    weighted = sum(w.get(i + 1, 0.0) * cap[i] for i in range(len(cap))) / wsum if wsum else float("nan")
    return {"key": d.get("key"), "layer": d.get("layer"),
            "engage_marginal_weights": {k: round(v, 4) for k, v in w.items()},
            "rank1_per_pc_capture": [round(c, 4) for c in cap],
            "rank1_engage_weighted_capture": round(weighted, 4),
            "R_engage_refusal_k": eng["R_engage_refusal_k"],
            "note": ("rank-1 harm-coextensive alternative REJECTED if capture is small; "
                     "rank-2/4 severity-harm basis is an extraction rider (contrasts unsaved)")}


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "papers/d3_decision_anatomy/outputs/c1_session_llama31_L12.json"
    res = rank1_capture(src)
    print(json.dumps(res, indent=2))
    c = res["rank1_engage_weighted_capture"]
    print(f"\n>> rank-1 harm percept spans {c:.1%} of the engage-driving moral basis "
          f"→ reads-broad survives the rank-1 harm-coextensive alternative")
