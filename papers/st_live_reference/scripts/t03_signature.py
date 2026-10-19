#!/usr/bin/env python3
"""T0.3 (author, 2026-10-19): GPT-OSS-20B only, descriptive, in the co-located decision-token cell.

    python3 papers/st_live_reference/scripts/t03_signature.py

Cell: harmony END_OF_PROMPT token (``<|start|>assistant``), hidden-state index 13 (block 12's
output, the W4 sources' "layer 12"). Directions there (§4 as corrected: sources from
``d1_vmoral_v1``, harm from ``refusal_prompts.json``; no re-extraction):

- ``V_moral``: QR of the three W4 decision-token source mean-differences (moral_stories, fables,
  ethics; 60 pairs each; ``moral`` member minus ``neutral`` member, moral pole positive), rank 3;
- ``V_harm``: harmful minus harmless (``refusal_prompts.json``, 64/64) at ``t_inst`` (FL's harm
  slice, primary; a different token from this cell, recorded) and at this END_OF_PROMPT token
  (``refusal_t_post_inst``; co-located; FL's refusal direction by construction; second column);
- ``V_care``: ``V_moral`` with its in-span harm component removed, rank 2 (per harm variant);
- the anchor axis (T0.6, author): the normalized sum of the harm-removed signed sources.

Rows: KDG's banked dose-0 letter cells at this token: the panel (``p2c`` ``dl_chat_neutral``,
split by norm status) and H1 (``p2m`` H1 ``dl_chat_neutral``, ``H1_train`` only, split protect /
rule / defer). Per split: the signed projection onto the anchor axis (sample-centred, in SDs of the
sample along it) and the projection fractions onto ``V_moral`` / ``V_care`` (centred residuals),
with scenario-cluster bootstrap CIs, against covariance-matched random subspaces of the same rank
drawn from the same sample. The position's PR is flagged (W4: 9.40 raw, below 30). Diagnostic:
the anchor axis against the protect-minus-rule orientation in ``V_care`` on H1_train. No L* verdict.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
W4 = ROOT / "papers/d3_decision_anatomy/outputs/w4/gpt_oss_20b"
POS = ROOT / "papers/7_reasoning/outputs/gpt_oss_20b/position_directions.npz"
PANEL = ROOT / "papers/kdg_panel/outputs/p2c/gpt_oss_20b/gpt_oss_20b/dl_chat_neutral"
H1 = ROOT / "papers/kdg_panel/outputs/p2m/h1/gpt_oss_20b/h1/dl_chat_neutral"
SPLIT = ROOT / "papers/st_live_reference/split_v1.json"
OUT = ROOT / "papers/st_live_reference/outputs/t03"
DIRS = ROOT / "papers/st_live_reference/data/directions"
LAYER, HIDX = 12, 13
SOURCES = ("moral_stories", "fables", "ethics")
N_NULL, N_BOOT, SEED = 500, 2000, 0
ACTION = {"violating": "protect", "consistent": "rule", "neutral": "defer"}  # H1 labels


def unit(v):
    return v / np.linalg.norm(v)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def directions() -> dict:
    src = np.load(W4 / "vmoral_sources_decision_token.npz")
    S = np.stack([unit(src[k].astype(np.float64)) for k in SOURCES])  # [3, H], moral pole +
    B = np.linalg.qr(S.T)[0].T  # [3, H] orthonormal rows
    pos = np.load(POS)
    harms = {"t_inst": unit(pos[f"harmfulness_t_inst_layer{LAYER}"].astype(np.float64)),
             "end_of_prompt": unit(pos[f"refusal_t_post_inst_layer{LAYER}"].astype(np.float64))}
    out = {"S": S, "V_moral": B, "harm": harms, "care": {}, "axis": {}, "in_span": {}}
    for k, h in harms.items():
        a = B @ h  # coordinates of h's in-span component
        out["in_span"][k] = float(np.linalg.norm(a))  # |P_V h| for unit h
        hv = unit(B.T @ a)  # the in-span harm direction
        P = np.eye(3) - np.outer(unit(a), unit(a))
        w, U = np.linalg.eigh(P)
        out["care"][k] = (U[:, w > 0.5].T @ B)  # [2, H], orthonormal, in V_moral, perp to hv
        r = S - np.outer(S @ hv, hv)  # harm-removed signed sources
        out["axis"][k] = unit(r.sum(0))
    return out


def load_cell(stem: Path, ids: set[str] | None):
    z = np.load(f"{stem}.npz")
    rows = [json.loads(x) for x in open(f"{stem}.jsonl")]
    keep = [i for i, r in enumerate(rows) if ids is None or r["scenario_id"] in ids]
    X = z["resid_decision"][keep, HIDX, :].astype(np.float64)
    lab = [rows[i]["norm_status"] for i in keep]
    sid = [rows[i]["scenario_id"] for i in keep]
    return X, np.array(lab), np.array(sid)


def pr(X: np.ndarray) -> float:
    lam = np.linalg.eigvalsh(np.cov(X.T, bias=False)) if X.shape[1] <= X.shape[0] else \
        np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def cluster_boot(vals: np.ndarray, sid: np.ndarray, rng) -> list[float]:
    groups = {}
    for v, s in zip(vals, sid):
        groups.setdefault(s, []).append(v)
    g = list(groups.values())
    means = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(g), len(g))
        means.append(np.mean(np.concatenate([g[i] for i in pick])))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def frac(Xc: np.ndarray, V: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Xc @ V.T, axis=1) / np.linalg.norm(Xc, axis=1).clip(1e-9)


def null_bases(Xc: np.ndarray, k: int, rng) -> list[np.ndarray]:
    out = []
    for _ in range(N_NULL):
        Z = rng.standard_normal((k, Xc.shape[0])) @ Xc  # covariance-matched draws
        out.append(np.linalg.qr(Z.T)[0].T)
    return out


def signature(X, lab, sid, D, harm: str, groups, rng) -> dict:
    mu = X.mean(0)
    Xc = X - mu
    ax = D["axis"][harm]
    proj = Xc @ ax
    sd = proj.std()
    res: dict = {"n": int(len(X)), "pr_centred": pr(X), "sd_along_axis": float(sd),
                 "splits": {}}
    nulls = {k: null_bases(Xc, k, rng) for k in (1, 2, 3)}
    for g in groups:
        m = lab == g
        if not m.any():
            continue
        z = proj[m] / sd
        fm, fc = frac(Xc[m], D["V_moral"]), frac(Xc[m], D["care"][harm])
        res["splits"][g] = {
            "n_rows": int(m.sum()), "n_scenarios": int(len(set(sid[m]))),
            "axis_z_mean": float(z.mean()), "axis_z_ci95": cluster_boot(z, sid[m], rng),
            "frac_V_moral": float(fm.mean()), "frac_V_care": float(fc.mean())}
    res["null"] = {f"rank{k}_frac_q50_q95": [float(np.quantile([frac(Xc, V).mean() for V in vs],
                                                                q)) for q in (0.5, 0.95)]
                   for k, vs in nulls.items()}
    res["null"]["observed_frac_V_moral"] = float(frac(Xc, D["V_moral"]).mean())
    res["null"]["observed_frac_V_care"] = float(frac(Xc, D["care"][harm]).mean())
    a, b = groups[-1], groups[0]  # the departure side minus the rule side
    if a in res["splits"] and b in res["splits"]:
        d_obs = res["splits"][a]["axis_z_mean"] - res["splits"][b]["axis_z_mean"]
        ma, mb = lab == a, lab == b
        d_null = []
        for V in nulls[1]:
            p = Xc @ V[0]
            d_null.append((p[ma].mean() - p[mb].mean()) / p.std())
        res["contrast"] = {"name": f"{a} - {b} on the anchor axis (SD units)", "observed": d_obs,
                           "null_abs_q95": float(np.quantile(np.abs(d_null), 0.95)),
                           "note": "descriptive; the split conditions on the model's dose-0 "
                                   "choice, so it is correlational (Amendment 1 item 4)"}
    return res


def diagnostic(X, lab, sid, D, harm: str, rng) -> dict:
    """T0.6 author item (1): the anchor axis against the protect-minus-rule orientation inside
    V_care on H1_train at this cell; anti-alignment (cos < 0) goes to ANOMALIES."""
    C = D["care"][harm]

    def orient(m_p, m_r):
        d = X[m_p].mean(0) - X[m_r].mean(0)
        return unit(C.T @ (C @ d))
    o = orient(lab == "violating", lab == "consistent")
    cos = float(o @ D["axis"][harm])
    scen = np.array(sorted(set(sid)))
    boots = []
    for _ in range(N_BOOT):
        pick = set(rng.choice(scen, len(scen)))
        m = np.isin(sid, list(pick))
        mp, mr = m & (lab == "violating"), m & (lab == "consistent")
        if mp.any() and mr.any():
            boots.append(float(orient(mp, mr) @ D["axis"][harm]))
    return {"cos_axis_protect_minus_rule": cos,
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "anti_aligned": cos < 0,
            "note": "cos of two unit vectors in the rank-2 V_care plane"}


def type_blocks(D: dict, pr_panel: float) -> dict:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=ROOT).stdout.strip()
    common = {"model": "openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee",
              "format": "harmony (think format)", "layer": "block 12 output (hidden index 13)",
              "participation_ratio": {"w4_raw_of_record": 9.40, "w4_std_of_record": 12.79,
                                      "kdg_panel_sample_centred": pr_panel,
                                      "flag": "PR < 30: position-invalid flag (construct-audit)"},
              "extraction_commit": commit}
    return {
        "V_moral": {**common, "contrast_semantics": "stimulus/content (moral-salience: `moral` "
                    "member minus `neutral` member)", "rank": 3,
                    "source_dataset": "deepsteer/datasets/d1_vmoral_v1.json @3b9bab7 "
                                      "(moral_stories, fables, ethics; 60 pairs/source)",
                    "position_class": "END_OF_PROMPT (harmony <|start|>assistant)",
                    "source_file": {"path": str((W4 / "vmoral_sources_decision_token.npz")
                                                .relative_to(ROOT)),
                                    "sha256": sha(W4 / "vmoral_sources_decision_token.npz"),
                                    "provenance": "d3 W4 unit 14.3 (PREREGISTRATION 14.3; "
                                                  "W4_RESULTS §14.3); extraction script not in "
                                                  "the repo; orientation inferred moral-pole "
                                                  "positive (cos with the pooled phase-2 sources "
                                                  "0.05-0.11, all positive)"},
                    "known_covariates": "narrative vs declarative register; source genre",
                    "outcome_variable": "none (descriptive T0.3); anchor axis -> §5.2 target"},
        "V_harm": {**common, "contrast_semantics": "stimulus/content (harmful minus harmless "
                   "requests)", "rank": 1,
                   "source_dataset": "papers/5_moral_alignment/refusal_prompts.json @0269799 "
                                     "(64/64)",
                   "variants": {"t_inst": "last instruction token (FL's harm slice; primary; "
                                          "not this cell's token)",
                                "end_of_prompt": "<|start|>assistant (co-located; FL's refusal "
                                                 "direction by construction)"},
                   "source_file": {"path": str(POS.relative_to(ROOT)), "sha256": sha(POS)},
                   "in_span_fraction": D["in_span"],
                   "known_covariates": "AdvBench vs Alpaca style, topic, length",
                   "outcome_variable": "none (removed from V_care)"},
        "V_care": {**common, "contrast_semantics": "stimulus/content, derived", "rank": 2,
                   "construction": "V_moral minus its in-span harm component (per harm variant)",
                   "outcome_variable": "§5.2 anchor (via the anchor axis)"},
        "anchor_axis": {**common, "construction": "normalized sum of the three harm-removed "
                        "signed sources (moral pole positive); author 2026-10-19", "rank": 1},
    }


def main() -> int:
    rng = np.random.default_rng(SEED)
    D = directions()
    split = json.loads(SPLIT.read_text())
    Xp, lp, sp = load_cell(PANEL, None)
    Xh, lh, sh = load_cell(H1, set(split["H1_train"]))
    res = {"cell": "harmony END_OF_PROMPT, hidden index 13", "harm_variants": {}}
    for harm in ("t_inst", "end_of_prompt"):
        res["harm_variants"][harm] = {
            "panel": signature(Xp, lp, sp, D, harm, ["consistent", "neutral", "violating"], rng),
            "h1_train": signature(Xh, lh, sh, D, harm, ["consistent", "neutral", "violating"],
                                  rng),
            "diagnostic_h1_train": diagnostic(Xh, lh, sh, D, harm, rng)}
    res["h1_label_map"] = ACTION
    tb = type_blocks(D, res["harm_variants"]["t_inst"]["panel"]["pr_centred"])
    res["type_blocks"] = tb
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t03_signature.json").write_text(json.dumps(res, indent=1))
    DIRS.mkdir(parents=True, exist_ok=True)
    np.savez(DIRS / "gpt_oss_20b_eop_L12.npz", V_moral=D["V_moral"].astype(np.float32),
             sources=D["S"].astype(np.float32),
             **{f"V_harm_{k}": v.astype(np.float32) for k, v in D["harm"].items()},
             **{f"V_care_{k}": v.astype(np.float32) for k, v in D["care"].items()},
             **{f"anchor_axis_{k}": v.astype(np.float32) for k, v in D["axis"].items()})
    (DIRS / "gpt_oss_20b_eop_L12.json").write_text(json.dumps(tb, indent=1))
    print(json.dumps({h: {c: (v.get("splits"), v.get("contrast")) if c != "diagnostic_h1_train"
                          else v for c, v in r.items()} for h, r in res["harm_variants"].items()},
                     indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
