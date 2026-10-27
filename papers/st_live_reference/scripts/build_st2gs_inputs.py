#!/usr/bin/env python3
"""The GPT-OSS site sweep's inputs (ST_PLAN Amendment 51 and its additions). Local; no model.

    python3 papers/st_live_reference/scripts/build_st2gs_inputs.py <outputs/t11/gpt_oss_20b>

From T1.1's bank and rollouts:
- **sites** (item 2, first addition item 1): the ten bank sites (blocks 6, 9, 12, 15, 18 x
  ``onset``, ``clause_end``), each with its median residual norm, its PR at 21 samples of 4,000 rows
  (seed 0, Amendment 12's convention, and seeds 1-20), eligibility (PR >= 30 at the median of the
  21; block 12's ``clause_end`` excluded as read) and its doses (0.4x and 0.3x the median norm);
- **fill draws** (item 6): ten covariance-matched unit draws per site (seed 3; Cell 1's recipe,
  ``steering_gate.fill_draws`` on a 4,000-row sample of the site's class);
- **traces** (item 4): T1.1's first two rollouts with an answer letter (by rollout index) of each
  of Cell 1's 45 engaged scenarios, teacher-forced through prompt and trace to the answer letter;
  each trace's class positions by the steering harness's rule (``steering_gate.Tracker``, the
  rule the sweep's arms steer by), checked against T1.1's banked positions; the violating and
  consistent options' letter ids; T1.1's option log-probs for the parity read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts"), str(HERE.parents[2])]
import constancy_read as CR  # noqa: E402
import steering_gate as SG  # noqa: E402

DATA = HERE.parent / "data"
OUT = HERE.parent / "outputs" / "st2gs_inputs" / "gpt_oss_20b"
KEY = "gpt_oss_20b"
N_FILL, FILL_SEED, N_SAMPLE, PR_SEEDS, PR_BAR = 10, 3, 4000, range(21), 30.0
DOSES = {"04": 0.4, "03": 0.3}
READ = ("12", "clause_end")  # Amendment 50's site of record: excluded from the sweep as read
PER_SCENARIO = 2


def site_name(block: int, cls: str) -> str:
    return f"b{block}_{cls}"


def class_positions(fmt, tok, punct: np.ndarray, gen: list[int], upto: int, prompt_len: int,
                    cls: str) -> list[int]:
    """Absolute positions the harness steers in ``cls`` for a trace ``gen`` fed up to (not
    including) generated index ``upto``: the tracker's state after each token is appended, the
    position being the token's own (``steering_gate.generate``'s rule, replayed)."""
    trk = SG.Tracker(fmt, tok, punct)
    out = []
    for i in range(upto):
        g = gen[: i + 1]
        trk.update(g)
        if trk.in_class(g, cls):
            out.append(prompt_len + i)
    return out


def sites(bank: Path) -> tuple[dict, dict]:
    """Per site: median norm, the 21 PRs, eligibility and doses; and its ten fill draws."""
    files = sorted(bank.glob("bank_*.npz"))
    meta, fill = {}, {}
    for cls, code in CR.CLASSES.items():
        parts = []
        for f in files:
            z = np.load(f)
            parts.append(z["resid"][(z["cls"] & code) > 0])
        xs = np.concatenate(parts)
        for s, b in enumerate(CR.BANK_BLOCKS):
            x = xs[:, s]
            prs = [CR.participation_ratio(x[np.random.default_rng(seed).choice(
                len(x), N_SAMPLE, replace=False)].astype(np.float32)) for seed in PR_SEEDS]
            med = float(np.median(np.linalg.norm(x.astype(np.float32), axis=-1)))
            rng = np.random.default_rng(FILL_SEED)
            samp = x[rng.choice(len(x), min(len(x), N_SAMPLE), replace=False)].astype(np.float32)
            name = site_name(b, cls)
            fill[name] = SG.fill_draws(samp, 1.0, N_FILL, FILL_SEED).astype(np.float32)
            pr_med = float(np.median(prs))
            read = (str(b), cls) == READ
            meta[name] = {"block": b, "cls": cls, "slot": s, "hidden_index": b + 1,
                          "n_rows": int(len(x)), "median_norm": med, "pr_seed0": prs[0],
                          "pr_seeds_1_20": prs[1:], "pr_median_21": pr_med,
                          "n_ge_bar_of_21": int(sum(p >= PR_BAR for p in prs)),
                          "eligible": bool(pr_med >= PR_BAR and not read), "read": read,
                          "alpha": {r: f * med for r, f in DOSES.items()}}
            print(f">> {name}: PR median {pr_med:.2f} ({meta[name]['n_ge_bar_of_21']} of 21 >= "
                  f"30), norm {med:.1f}, eligible {meta[name]['eligible']}", flush=True)
        del xs, parts
    return meta, fill


def traces(t11: Path) -> list[dict]:
    """The 90 teacher-forced traces with their class positions and letter ids (item 4)."""
    import pod_st_screen as P
    from reasoning_formats import build_format, letter_token_ids
    from transformers import AutoTokenizer

    from deepsteer.kdg.schema import load_scenario_dir

    spec = P.registry()[KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(KEY, tok, spec)
    punct = SG.punct_mask(tok)
    lid = {L: i for i, L in letter_token_ids(tok).items()}
    H = {s.id: s for s in load_scenario_dir([P.H1_SET])[0]}
    engaged = json.loads((DATA / "st2_engaged_train_v1.json").read_text())["ids"]
    rows = [json.loads(x) for x in (t11 / "gen" / "h1_rollouts_read.jsonl").open()]
    bank = {}
    for f in sorted((t11 / "bank").glob("bank_*.npz")):
        z = np.load(f)
        for r, p, c in zip(z["row"], z["pos"], z["cls"]):
            bank.setdefault(int(r), []).append((int(p), int(c)))
    out = []
    for sid in engaged:
        cand = sorted((r["rollout"], i) for i, r in enumerate(rows)
                      if r["scenario_id"] == sid and r.get("answer_idx") is not None
                      and r.get("letter") in (r.get("order") or {}))
        for _, i in cand[:PER_SCENARIO]:
            r = rows[i]
            role = {o.option_id: o.norm_status for o in H[sid].options}
            by = {role[oid]: L for L, oid in r["order"].items()}
            n, a = len(r["prompt_ids"]), r["answer_idx"]
            pos = {c: class_positions(fmt, tok, punct, r["gen_ids"], a, n, c)
                   for c in CR.CLASSES}
            banked = {c: sorted(p for p, k in bank.get(i, []) if k & code and p < n + a)
                      for c, code in CR.CLASSES.items()}
            out.append({"scenario_id": sid, "rollout": r["rollout"], "t11_row": i,
                        "ids": r["prompt_ids"] + r["gen_ids"][:a], "prompt_len": n,
                        "answer_idx": a, "positions": pos,
                        "bank_match": {c: pos[c] == banked[c] for c in CR.CLASSES},
                        "violating_letter": by["violating"], "consistent_letter": by["consistent"],
                        "violating_id": lid[by["violating"]],
                        "consistent_id": lid[by["consistent"]],
                        "t11_option_logps": r["option_logps"], "t11_letter": r["letter"],
                        "t11_norm_status": r.get("norm_status")})
    return out


def main() -> None:
    t11 = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    tr = traces(t11)
    with (OUT / "traces.jsonl").open("w") as f:
        for t in tr:
            f.write(json.dumps(t) + "\n")
    match = {c: sum(t["bank_match"][c] for t in tr) for c in CR.CLASSES}
    print(f">> traces: {len(tr)} over {len({t['scenario_id'] for t in tr})} scenarios; positions "
          f"equal to T1.1's banked ones: {match}", flush=True)
    meta, fill = sites(t11 / "bank")
    np.savez(OUT / "fill_sites.npz", **fill)
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (OUT / "fill_sites.npz", OUT / "traces.jsonl")}
    doc = {"rule": "ST_PLAN Amendment 51 and its additions", "sites": meta,
           "candidates": [k for k, v in meta.items() if v["eligible"]],
           "fill": {"seed": FILL_SEED, "n_draws": N_FILL, "sample_n": N_SAMPLE},
           "traces": {"n": len(tr), "per_scenario": PER_SCENARIO, "bank_match": match},
           "sha256": sha}
    (OUT / "sites.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f">> candidates: {doc['candidates']}")


if __name__ == "__main__":
    main()
