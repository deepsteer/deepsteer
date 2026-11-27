#!/usr/bin/env python3
"""The next adapter build's hold-out index (GPC_PLAN.md §19.2): every word 8-gram of the transfer test's text, so no
corpus the build writes shares it.
- the three depth scenarios' own files, both variants, at the pinned ODCV-Bench clone (prompts, scorers, data);
- A6's 48 renders (`data/a6_renders_v1.json`);
- A7's 48 states (`outputs/_sync/a7_states_8gram_v1.npy`, already hashed).

The hashes are one way (blake2b-64, `overlap_hashes`'s rule), and no text is written. The corpus check takes this
index with the eval index (`eval_8gram_hashes_v2.npy`).

    python c2_holdout.py [--clone DIR]   # writes outputs/_sync/c2_holdout_8gram_v1.npy and data/c2_holdout_v1.json
    python c2_holdout.py --v2            # v1 plus the held-out families' text (§19.14): outputs/_sync/c2_holdout_8gram_v2.npy

**v2 (2026-11-27, §19.14):** the held-out families' own text (the held-out area's manipulation families and the
counterparty split) joins the index before any further trained family is written, so no trained family shares an
8-gram with a held-out one. The held-out families are themselves checked against v1 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "papers/cst_constitutional_steering/scripts")]
import overlap_hashes as OH  # noqa: E402

GPC = HERE.parent
OUT = GPC / "outputs/_sync/c2_holdout_8gram_v1.npy"
REC = GPC / "data/c2_holdout_v1.json"
DEPTH = ("Ai-Education-Assistant", "Ai-Legal-Reviewer-Conflict", "Healthcare-Management")
ODCV_PIN = "6924dafb3a3f84fc320fbe9f0a08203d67d1d488"


def scenario_texts(d: Path) -> list[str]:
    """Every text file under the scenario's directory (checked on the parts inside it, so a clone under ~/.cache is
    read; `z_index_v2.texts_from` skips any path with `.cache` in it). Binary files (a NUL byte) are skipped."""
    out = []
    for f in sorted(d.rglob("*")):
        if not f.is_file() or ".git" in f.relative_to(d).parts:
            continue
        raw = f.read_bytes()
        if b"\x00" in raw[:4096] or len(raw) > 4_000_000:
            continue
        out.append(raw.decode("utf-8", errors="ignore"))
    return out


def build(clone: Path) -> dict:
    parts, counts = [], {}
    for v in ("mandated", "incentivized"):
        for s in DEPTH:
            d = clone / f"{v}_scenarios" / s
            if not d.is_dir():
                raise SystemExit(f"BAIL: {d} is missing (the clone at {ODCV_PIN})")
            ts = scenario_texts(d)
            h = np.unique(np.concatenate([OH.hashes(t) for t in ts if t] or [np.zeros(0, np.uint64)]))
            counts[f"odcv/{v}/{s}"] = {"texts": len(ts), "hashes": int(len(h))}
            parts.append(h)
    rs = json.loads((GPC / "data/a6_renders_v1.json").read_text())["renders"]
    ts = [str(r.get(k, "")) for r in rs for k in ("system_message", "scene", "goal_option", "other_option",
                                                  "goal_sentence")]
    h = np.unique(np.concatenate([OH.hashes(t) for t in ts if t]))
    counts["a6_renders"] = {"texts": len(ts), "hashes": int(len(h))}
    parts.append(h)
    a7 = np.load(GPC / "outputs/_sync/a7_states_8gram_hashes_v1.npy")
    counts["a7_states"] = {"hashes": int(len(a7))}
    parts.append(a7)
    idx = np.unique(np.concatenate(parts))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, idx)
    rec = {"file": str(OUT.relative_to(GPC)), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
           "n_hashes": int(len(idx)), "per_source": counts, "odcv_pin": ODCV_PIN,
           "hash": "blake2b-64 of the space-joined word 8-gram (overlap_hashes)",
           "rule": "GPC_PLAN.md §19.2: every kept training turn is checked against this index and the eval index"}
    REC.write_text(json.dumps(rec, indent=1) + "\n")
    return rec



def build_v2() -> dict:
    import c2_families as F

    v1 = np.load(OUT)
    held = [f for f in F.ROOM if f.split in ("pilot_heldout", "pilot_counterparty")]
    texts = [t for f in held for t in (f.system, f.user, f.twin_user, f.named_shortcut, f.named_constrained,
                                         *f.files.values(), *f.twin_files.values()) if t]
    fam = np.unique(np.concatenate([OH.hashes(t) for t in texts]))
    idx = np.unique(np.concatenate([v1, fam]))
    out2 = GPC / "outputs/_sync/c2_holdout_8gram_v2.npy"
    np.save(out2, idx)
    rec = {"file": str(out2.relative_to(GPC)), "sha256": hashlib.sha256(out2.read_bytes()).hexdigest(),
           "n_hashes": int(len(idx)), "v1_hashes": int(len(v1)), "held_out_families": [f.fid for f in held],
           "family_hashes": int(len(fam)), "rule": "GPC_PLAN.md §19.14: v1 plus the held-out families' own text"}
    (GPC / "data/c2_holdout_v2.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", type=Path, default=Path.home() / ".cache/deepsteer/gpc4_local/ODCV-Bench")
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build_v2() if a.v2 else build(a.clone), indent=1))
