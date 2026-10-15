#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 7: per-rollout option probabilities for release.

    python3 papers/kdg_panel/scripts/export_rollout_option_probs.py [--out DIR]

Walks every saved cell under ``papers/kdg_panel/outputs/<pod>/`` (dry runs and ``*_partial`` files
skipped) and writes one row per rollout: pod, model, cell (the condition), variant (primary or
pressure-removed frame), scenario id with its role and primary link (``twin_of`` for harm twins),
rollout and seed index, the letter-to-option order, the parsed option, and the decision-position
probabilities of the displayed options. Two renormalizations are written, because the paper uses
both: ``p_options`` over the letter tokens the cell saved (letter-only and decision-position cells),
and ``p_options_both`` summing each letter's bare and space-prefixed tokens (the forced-answer arms
of the dose cells, as ``analyze_phase1_session_a.row_masses``). Rows with no saved distribution
(reconstructed GPT-OSS action rows, judgment traces) carry the parsed option only.

Output: ``kdg_rollout_option_probs.csv.gz`` plus a manifest with row counts per file and the
output's SHA-256. The output is a release artifact and is not committed (repository policy: no
bundled datasets); the paper's Appendix F names where it is published.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

LETTERS = "ABCDE"
TOKENIZERS = {  # model-key prefix -> tokenizer repo (read from the local HF cache only)
    "olmo3": "allenai/Olmo-3-7B-Instruct",
    "llama31": "allenai/Llama-3.1-Tulu-3-8B",
    "tulu3": "allenai/Llama-3.1-Tulu-3-8B",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "gpt_oss": "openai/gpt-oss-20b",
}
COLUMNS = ["pod", "model", "cell", "variant", "scenario_id", "role", "twin_of", "family",
           "generator", "rollout", "seed", "order", "parsed_option", "parse_method",
           "parsed_norm_status", "forced", "letter_mass", "p_options", "p_violating",
           "p_options_both", "p_violating_both", "source"]


def letter_ids(model: str) -> tuple[dict, dict]:
    from transformers import AutoTokenizer

    repo = next(r for k, r in TOKENIZERS.items() if model.startswith(k))
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    bare = {L: tok.encode(L, add_special_tokens=False) for L in LETTERS}
    sp = {L: tok.encode(" " + L, add_special_tokens=False) for L in LETTERS}
    # "a letter is one token in both spellings": a multi-token letter would be read wrongly
    assert all(len(v) == 1 for v in list(bare.values()) + list(sp.values())), model
    return {k: v[0] for k, v in bare.items()}, {k: v[0] for k, v in sp.items()}


def scenario_index() -> dict[str, dict]:
    files = sorted(A.DATA.glob("*_scenarios_*.json"))
    files.append(A.DATA / "harm_divergence" / "hdiv_set.json")
    scen, _ = load_scenario_dir(files)
    return {s.id: {"role": s.role, "twin_of": s.twin_of or "", "family": s.family,
                   "generator": s.generator,
                   "status": {o.option_id: o.norm_status for o in s.options}} for s in scen}


def model_of(path: Path) -> str:
    """The model key: the cell's directory, or its parent for GPT-OSS's h1 sub-runs."""
    d = path.parent.name
    return path.parent.parent.name if d.startswith("h1") else d


def renorm(order: dict, logp: dict[str, float], status: dict) -> tuple[dict, float]:
    letters = sorted(order)
    m = np.exp(np.asarray([logp[L] for L in letters], float))
    m = m / m.sum()
    p = {order[L]: float(x) for L, x in zip(letters, m)}
    return p, sum(x for o, x in p.items() if status.get(o) == "violating")


def export(out: Path) -> dict:
    idx = scenario_index()
    ids: dict[str, tuple[dict, dict]] = {}
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "kdg_rollout_option_probs.csv.gz"
    counts: dict[str, int] = {}
    skipped: Counter = Counter()
    with gzip.open(dest, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for f in sorted(A.OUT.glob("*/**/*.jsonl")):
            rel = f.relative_to(A.OUT)
            if any(x.startswith("_dry") for x in rel.parts) or f.stem.endswith("_partial"):
                continue
            rows = A._rows(f)
            if not rows or "scenario_id" not in rows[0]:
                skipped["not_rollout_rows"] += 1
                continue
            model = model_of(f)
            npz = f.with_suffix(".npz")
            lp = oid = None
            if npz.exists() and "logp_decision.npy" in np.load(npz).files:
                z = np.load(npz)
                lp, oid = z["logp_decision"], z["option_token_ids"]
                assert lp.shape[0] == len(rows), (str(rel), lp.shape, len(rows))
                if model not in ids:
                    ids[model] = letter_ids(model)
            n = 0
            for k, r in enumerate(rows):
                sc = idx.get(r["scenario_id"])
                if sc is None:
                    skipped["unknown_scenario"] += 1
                    continue
                order = r.get("order") or {}
                rec = {"pod": rel.parts[0], "model": model, "cell": r.get("cell", f.stem),
                       "variant": r.get("variant", ""), "scenario_id": r["scenario_id"],
                       "role": sc["role"], "twin_of": sc["twin_of"], "family": sc["family"],
                       "generator": sc["generator"], "rollout": r.get("rollout", ""),
                       "seed": r.get("seed", ""), "order": json.dumps(order, sort_keys=True),
                       "parsed_option": r.get("option_id") or "",
                       "parse_method": r.get("parse_method", ""),
                       "parsed_norm_status": sc["status"].get(r.get("option_id"), ""),
                       "forced": f.stem.endswith("_forced"), "source": str(rel)}
                if order and "option_logps" in r:
                    p, v = renorm(order, r["option_logps"], sc["status"])
                    rec.update(letter_mass=r.get("option_mass", ""), p_options=json.dumps(p),
                               p_violating=v)
                elif order and lp is not None:
                    vec = lp[k].astype(np.float64)
                    assert vec.ndim == 1 and vec.size > 1000, (str(rel), k, vec.shape)
                    bare, sp = ids[model]
                    saved = dict(zip(LETTERS, oid[k]))
                    one = {L: float(vec[int(saved[L])]) for L in order}
                    p, v = renorm(order, one, sc["status"])
                    both = {L: float(np.logaddexp(vec[bare[L]], vec[sp[L]])) for L in order}
                    pb, vb = renorm(order, both, sc["status"])
                    rec.update(letter_mass=float(np.exp(list(one.values())).sum()),
                               p_options=json.dumps(p), p_violating=v,
                               p_options_both=json.dumps(pb), p_violating_both=vb)
                w.writerow(rec)
                n += 1
            counts[str(rel)] = n
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    man = {"rule": "KDG_PHASE1_SPEC P1-A17 item 7", "file": dest.name, "sha256": sha,
           "rows": sum(counts.values()), "files": counts, "skipped": dict(skipped),
           "columns": COLUMNS}
    (out / "kdg_rollout_option_probs.manifest.json").write_text(json.dumps(man, indent=1))
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=A.OUT / "release")
    a = ap.parse_args()
    m = export(a.out)
    print(json.dumps({k: v for k, v in m.items() if k != "files"}, indent=1))
