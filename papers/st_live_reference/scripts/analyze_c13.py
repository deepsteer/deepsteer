#!/usr/bin/env python3
"""§13's evaluation read and cell A′ (ST_PLAN Amendments 36 and 40; zero GPU).

    python3 papers/st_live_reference/scripts/analyze_c13.py --c13 <outputs/c13> \
        [--author <answers.json> --key <key.json>] [--subagent <ratings.json>]
    python3 papers/st_live_reference/scripts/analyze_c13.py --c13 <outputs/c13> --amendment42

``--amendment42`` runs ST-A26's standardized read alone (Amendment 42: σ from the pooled sample at
the site, every object re-read in standardized coordinates, OLMo-3-Think's invariance check, the
branch; the raw read and the top-k projection-out variant beside); it never touches the family
read. ``--amendment44`` runs ST-A26's document-contrast read alone (Amendment 44: each document's
direction with the other's removed, a held-out specificity AUC, the positive control on the
C_spec-specific direction; standardized of record, raw beside).

Per model, at its headline layer (d1's conventions): ``V_moral`` = d1's three sources' mean
moral − neutral differences, orthonormalized (rank 3); ``V_const`` per document = the mean m − n of
its training pairs. Overlap of record: the norm fraction ‖Qᵀv̂‖ (d1's projection fraction), with the
principal angle; a pair bootstrap; the held-out pairs' share with proj(m) > proj(n). Ladder: the
covariance-matched null (Xcᵀg, g ~ N(0, I), Xc the centered pooled text means; 10,000 draws), the
held-one-out band [min, max]; per band layer beside. Normalized overlap u = (PF − null q50) /
(band min − null q50). Positive control (Amendment 40 item 3): u(GPT-OSS, C_spec) −
u(OLMo-3-Think, C_spec), bootstrap lower bound above 0. The family read (item 5) and cell A′'s
branch (Amendment 36 item 4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import t13_driver as T13  # noqa: E402

DATA = HERE.parent / "data"
MODELS = ("gpt_oss_20b", "olmo3_think")
SOURCES = ("moral_stories_train", "fables_train_full", "ethics_train_full")
N_BOOT, N_NULL, SEED = 1000, 10000, 0
TOPK_SHARE = 0.05
RELIABILITY_BAR = 0.8


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def ortho(dirs: list[np.ndarray]) -> np.ndarray:
    return np.linalg.qr(np.stack(dirs, axis=1))[0]


def frac(Q: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(Q.T @ v) / (np.linalg.norm(v) + 1e-12))


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def diffs(keys: list[str], X: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """``{pair id: m − n}`` for keys ``<prefix>|<id>|m`` and ``...|n``."""
    idx = {k: i for i, k in enumerate(keys)}
    out = {}
    for k in keys:
        if k.startswith(prefix + "|") and k.endswith("|m"):
            stem = k[:-2]
            if stem + "|n" in idx:
                out[stem.split("|")[-1]] = X[idx[k]] - X[idx[stem + "|n"]]
    return out


def ladder(X: np.ndarray, src: dict[str, np.ndarray], rng, n_null: int) -> dict:
    dirs = {s: unit(v) for s, v in src.items()}
    Q = ortho(list(dirs.values()))
    held = {s: frac(ortho([dirs[o] for o in dirs if o != s]), dirs[s]) for s in dirs}
    Xc = X - X.mean(0, keepdims=True)
    G = rng.standard_normal((n_null, Xc.shape[0]))
    fr = np.array([frac(Q, Xc.T @ g) for g in G])
    return {"Q": Q, "heldout": held, "band": [min(held.values()), max(held.values())],
            "null_q50": float(np.percentile(fr, 50)), "null_q95": float(np.percentile(fr, 95))}


def transform(X: np.ndarray, mode: str) -> np.ndarray:
    """The pooled sample in the read's coordinates. ``raw`` (Amendment 40); ``std``, each dimension
    divided by its standard deviation over the pooled sample at the site, plus 1e-12 (Amendment 42
    item 1; d2's in-format convention); ``topk``, the dimensions holding more than 5% of the raw
    variance removed (Amendment 42 item 6, beside). Each is linear and per-dimension, so every
    difference, mean and null drawn from the transformed sample is the transformed object."""
    if mode == "raw":
        return X
    if mode == "std":
        return X / (X.std(0) + 1e-12)
    if mode == "topk":
        v = X.var(0)
        return X[:, v / v.sum() <= TOPK_SHARE]
    raise ValueError(mode)


def transform_like(Z: np.ndarray, X0: np.ndarray, mode: str) -> np.ndarray:
    """Texts extracted later (Amendment 47's C_spec2025 pairs) in the coordinates of the saved
    pooled sample X0: its σ, or its top-k mask (Amendment 47 item 3)."""
    if mode == "raw":
        return Z
    if mode == "std":
        return Z / (X0.std(0) + 1e-12)
    if mode == "topk":
        v = X0.var(0)
        return Z[:, v / v.sum() <= TOPK_SHARE]
    raise ValueError(mode)


def top_share(X: np.ndarray) -> float:
    v = X.var(0)
    return float(v.max() / v.sum())


def model_read(c13: Path, model: str, mode: str = "raw") -> dict:
    from deepsteer.geometry.participation import participation_ratio

    z = np.load(c13 / model / "extract.npz")
    keys = [str(k) for k in z["keys"]]
    X0 = z["headline"].astype(np.float64)
    X = transform(X0, mode)
    rng = np.random.default_rng(SEED)
    src = {s: np.mean(list(diffs(keys, X, f"d1|{s}").values()), axis=0) for s in SOURCES}
    lad = ladder(X, src, rng, N_NULL)
    out: dict = {"headline": int(z["headline_index"]), "n_texts": len(keys), "mode": mode,
                 "participation_ratio": float(participation_ratio(X)),
                 "participation_ratio_raw": float(participation_ratio(X0)),
                 "top_dim_share": top_share(X), "top_dim_share_raw": top_share(X0),
                 "n_dims": int(X.shape[1]),
                 "heldout_band": lad["band"], "heldout": lad["heldout"],
                 "null_q50": lad["null_q50"], "null_q95": lad["null_q95"], "documents": {}}
    vc = {}
    for doc in ("spec", "anth"):
        pairs = json.loads((DATA / f"st_c13_pairs_{doc}_v1.json").read_text())["pairs"]
        d = diffs(keys, X, f"c13|{doc}")
        tr = [d[str(i)] for i, p in enumerate(pairs) if p["split"] == "train" and str(i) in d]
        te = [d[str(i)] for i, p in enumerate(pairs) if p["split"] == "test" and str(i) in d]
        v = vc[doc] = np.mean(tr, axis=0)
        pf = frac(lad["Q"], v)
        boot = np.array([frac(lad["Q"], np.mean([tr[j] for j in
                                                 rng.integers(0, len(tr), len(tr))], axis=0))
                         for _ in range(N_BOOT)])
        denom = lad["band"][0] - lad["null_q50"]
        u = (pf - lad["null_q50"]) / denom if denom > 0 else None
        ub = (boot - lad["null_q50"]) / denom if denom > 0 else None
        valid = lad["band"][0] > lad["null_q95"]  # instrument-calibration: band above the null
        out["position_valid"] = valid
        out["documents"][doc] = {
            "pf": pf,
            "pf_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "angle_deg": float(np.degrees(np.arccos(min(1.0, pf)))), "u": u,
            "u_boot": None if ub is None else [float(x) for x in ub],
            "heldout_sign_share": float(np.mean([float(t @ unit(v) > 0) for t in te]))
            if te else None, "n_train": len(tr), "n_test": len(te),
            "ladder_position": ("position invalid: the band sits below the null q95" if not valid
                                else "above the band floor" if pf >= lad["band"][0] else
                                "above the null q95, below the band" if pf > lad["null_q95"]
                                else "within the null")}
    out["cos_documents"] = float(unit(vc["spec"]) @ unit(vc["anth"]))
    per_layer = {}
    B = z["band"].astype(np.float64)
    for j, h in enumerate(z["hidden"]):
        Xl = transform(B[:, j], mode)
        sl = {s: np.mean(list(diffs(keys, Xl, f"d1|{s}").values()), axis=0) for s in SOURCES}
        ll = ladder(Xl, sl, np.random.default_rng(SEED + int(h)), 2000)
        row = {"band": ll["band"], "null_q95": ll["null_q95"]}
        for doc in ("spec", "anth"):
            pairs = json.loads((DATA / f"st_c13_pairs_{doc}_v1.json").read_text())["pairs"]
            d = diffs(keys, Xl, f"c13|{doc}")
            row[doc] = frac(ll["Q"], np.mean([d[str(i)] for i, p in enumerate(pairs)
                                              if p["split"] == "train" and str(i) in d], axis=0))
        per_layer[int(h)] = row
    out["per_layer"] = per_layer
    return out


def positive_control(reads: dict) -> dict:
    a, b = reads["gpt_oss_20b"]["documents"]["spec"], reads["olmo3_think"]["documents"]["spec"]
    if not (reads["gpt_oss_20b"]["position_valid"] and reads["olmo3_think"]["position_valid"]):
        return {"status": "uninformative: a position is invalid (band below its null q95)"}
    if a["u"] is None or b["u"] is None:
        return {"status": "uninformative (a band floor at or below its null median)"}
    rng = np.random.default_rng(SEED)
    du = np.array(a["u_boot"])[rng.permutation(N_BOOT)] - np.array(b["u_boot"])
    lo = float(np.percentile(du, 2.5))
    return {"delta_u": a["u"] - b["u"], "ci95": [lo, float(np.percentile(du, 97.5))],
            "passes": bool(lo > 0),
            "consequence": "calibrated" if lo > 0 else
            "uncalibrated: the overlap read is not used on C_partner (section 13)"}


def invariance(raw: dict, std: dict) -> dict:
    """Amendment 42 item 3, on the clean model: standardization must keep its position validity and
    each document's ladder position."""
    same = {"position_valid": raw["position_valid"] == std["position_valid"],
            **{doc: raw["documents"][doc]["ladder_position"]
               == std["documents"][doc]["ladder_position"] for doc in raw["documents"]}}
    return {"holds": all(same.values()), "same": same,
            "raw": {"position_valid": raw["position_valid"],
                    **{d: raw["documents"][d]["ladder_position"] for d in raw["documents"]}},
            "std": {"position_valid": std["position_valid"],
                    **{d: std["documents"][d]["ladder_position"] for d in std["documents"]}}}


def branch42(pc: dict, inv: dict, gpt_valid: bool) -> str:
    """Amendment 42 item 5, written before data."""
    if not inv["holds"]:
        return ("invariance fails: standardization alters the clean instrument; the standardized "
                "positive control is reported, not of record; ST-A26 R_c stays open")
    if not gpt_valid:
        return ("GPT-OSS's position is invalid after standardization: the positive control is "
                "uninformative at this site; the document-contrast read is designed with that "
                "stated")
    if pc.get("passes"):
        return ("R_c: GPT-OSS's raw ladder was the outlier's; ST-A26 resolves by calibration and "
                "the standardized ladder is section 13's instrument for later overlap reads")
    return ("fails with GPT-OSS's position valid: R_c does not account for the failure; R_a and "
            "R_b go to the document-contrast read")


def amendment42(c13: Path) -> dict:
    """ST-A26's standardized read (Amendment 42); the raw read of record recomputed beside it, and
    the top-k projection-out variant beside, not of record. The family read is not touched."""
    reads = {mode: {m: model_read(c13, m, mode) for m in MODELS} for mode in ("raw", "std", "topk")}
    pc = {mode: positive_control(reads[mode]) for mode in reads}
    inv = invariance(reads["raw"]["olmo3_think"], reads["std"]["olmo3_think"])
    return {"rule": "ST_PLAN Amendment 42", "reads": reads, "positive_control": pc,
            "invariance_olmo3_think": inv,
            "branch": branch42(pc["std"], inv, reads["std"]["gpt_oss_20b"]["position_valid"]),
            "of_record": "std (Amendment 42); raw stays of record for Amendment 40; topk beside"}


def specific_dir(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Amendment 44 item 2: ``va``'s direction with ``vb``'s removed (both unit first), unit."""
    ua, ub = unit(va), unit(vb)
    return unit(ua - (ua @ ub) * ub)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a positive score exceeds a negative one), ties half."""
    d = np.asarray(pos)[:, None] - np.asarray(neg)[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def doc_contrast(c13: Path, model: str, mode: str = "std", spec2025: Path | None = None) -> dict:
    """Amendment 44 items 2-4 for one model: the document-specific directions on Amendment 42's
    ladder, the held-out specificity AUC, u per direction with a pair bootstrap. With ``spec2025``
    (the c13c outputs), C_spec's pairs are replaced by the 2025-04-11 revision's, extracted at the
    same site and placed in the saved pooled sample's coordinates (Amendment 47 item 3)."""
    z = np.load(c13 / model / "extract.npz")
    keys = [str(k) for k in z["keys"]]
    X0 = z["headline"].astype(np.float64)
    X = transform(X0, mode)
    rng = np.random.default_rng(SEED)
    src = {s: np.mean(list(diffs(keys, X, f"d1|{s}").values()), axis=0) for s in SOURCES}
    lad = ladder(X, src, rng, N_NULL)
    q50, b0 = lad["null_q50"], lad["band"][0]
    tr, te = {}, {}
    src_of = {"spec": ("spec", keys, X), "anth": ("anth", keys, X)}
    if spec2025 is not None:
        z2 = np.load(spec2025 / model / "extract_spec2025.npz")
        src_of["spec"] = ("spec2025", [str(k) for k in z2["keys"]],
                          transform_like(z2["headline"].astype(np.float64), X0, mode))
    for doc, (name, kk, XX) in src_of.items():
        pairs = json.loads((DATA / f"st_c13_pairs_{name}_v1.json").read_text())["pairs"]
        d = diffs(kk, XX, f"c13|{name}")
        tr[doc] = np.array([d[str(i)] for i, p in enumerate(pairs)
                            if p["split"] == "train" and str(i) in d])
        te[doc] = np.array([d[str(i)] for i, p in enumerate(pairs)
                            if p["split"] == "test" and str(i) in d])

    def stats(trs: dict, tes: dict) -> tuple[float, float, float, float, float]:
        vs, va = trs["spec"].mean(0), trs["anth"].mean(0)
        sp, ap = specific_dir(vs, va), specific_dir(va, vs)
        us = (frac(lad["Q"], sp) - q50) / (b0 - q50)
        ua = (frac(lad["Q"], ap) - q50) / (b0 - q50)
        uv = (frac(lad["Q"], vs) - q50) / (b0 - q50)  # Amendment 40's form on V_const(spec)
        return us, ua, auc(tes["spec"] @ sp, tes["anth"] @ sp), frac(lad["Q"], unit(
            unit(vs) - unit(va))), uv

    us, ua, a_, pfc, uv = stats(tr, te)
    boot = []
    for _ in range(N_BOOT):
        rs = {k: v[rng.integers(0, len(v), len(v))] for k, v in tr.items()}
        rt = {k: v[rng.integers(0, len(v), len(v))] for k, v in te.items()}
        boot.append(stats(rs, rt))
    B = np.array(boot)
    lo = float(np.percentile(B[:, 2], 2.5))
    return {"mode": mode, "position_valid": bool(b0 > lad["null_q95"]), "null_q50": q50,
            "null_q95": lad["null_q95"], "band": lad["band"],
            "u_spec_perp": float(us), "u_anth_perp": float(ua),
            "u_spec_perp_boot": B[:, 0].tolist(), "u_anth_perp_boot": B[:, 1].tolist(),
            "specificity_auc": a_, "specificity_auc_ci95": [lo, float(np.percentile(B[:, 2],
                                                                                    97.5))],
            "specificity_passes": bool(lo > 0.5), "pf_contrast": float(pfc),
            "u_vconst_spec": float(uv), "u_vconst_spec_boot": B[:, 4].tolist(),
            "spec_source": "C_spec2025" if spec2025 is not None else "C_spec",
            "n_train": {k: len(v) for k, v in tr.items()}, "n_test": {k: len(v)
                                                                       for k, v in te.items()}}


def branch44(check: dict, pc: dict) -> str:
    """Amendment 44 item 5, written before data."""
    if not pc.get("valid_positions", True):
        return "uninformative: a position is invalid (band at or below its null q95)"
    if not all(check.values()):
        return ("R_a, strong form: these pairs carry no document-specific content the instrument "
                "can read; the positive control is uninformative and section 13's overlap read "
                "needs document-specific stimuli (new pairs, GPU, own amendment)")
    if pc["passes"]:
        return ("R_a: the shared component masked the premise; the specific-direction read is "
                "section 13's calibrated overlap instrument")
    return ("R_b: training on a document does not draw its specific content toward the model's "
            "moral subspace at this site; the overlap read is not used on C_partner, and the "
            "partner deliverable keeps the family read")


def contrast_read(c13: Path, spec2025: Path | None, branch, rule: str) -> dict:
    """Amendment 44's form (and, with ``spec2025``, Amendment 47's): standardized of record, raw
    beside; Amendment 40's form on V_const(spec) beside."""
    out: dict = {"rule": rule}
    for mode in ("std", "raw"):
        r = {m: doc_contrast(c13, m, mode, spec2025) for m in MODELS}
        g, o = r["gpt_oss_20b"], r["olmo3_think"]
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(N_BOOT)
        b = {k: (np.array(g[k])[perm], np.array(o[k])) for k in
             ("u_spec_perp_boot", "u_anth_perp_boot", "u_vconst_spec_boot")}
        du = b["u_spec_perp_boot"][0] - b["u_spec_perp_boot"][1]
        did_b = (b["u_spec_perp_boot"][0] - b["u_anth_perp_boot"][0]) \
            - (b["u_spec_perp_boot"][1] - b["u_anth_perp_boot"][1])
        dv = b["u_vconst_spec_boot"][0] - b["u_vconst_spec_boot"][1]
        lo = float(np.percentile(du, 2.5))
        pc = {"delta_u": g["u_spec_perp"] - o["u_spec_perp"],
              "ci95": [lo, float(np.percentile(du, 97.5))], "passes": bool(lo > 0),
              "valid_positions": bool(g["position_valid"] and o["position_valid"])}
        did = {"did": (g["u_spec_perp"] - g["u_anth_perp"]) - (o["u_spec_perp"]
                                                               - o["u_anth_perp"]),
               "ci95": [float(np.percentile(did_b, 2.5)), float(np.percentile(did_b, 97.5))]}
        a40 = {"delta_u": g["u_vconst_spec"] - o["u_vconst_spec"],
               "ci95": [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))]}
        check = {m: r[m]["specificity_passes"] for m in MODELS}
        out[mode] = {"models": r, "specificity_check": check, "positive_control": pc,
                     "did_beside": did, "amendment40_form_beside": a40,
                     "branch": branch(check, pc)}
    out["of_record"] = "std; raw beside"
    return out


def amendment44(c13: Path) -> dict:
    """ST-A26's document-contrast read (Amendment 44), standardized of record, raw beside; its
    control verdict voided by Amendment 47 (version mismatch), the measurements descriptive."""
    return contrast_read(c13, None, branch44, "ST_PLAN Amendment 44 (control voided: Amendment 47)")


def branch47(check: dict, pc: dict) -> str:
    """Amendment 47 item 4, written before data."""
    if not pc.get("valid_positions", True):
        return "uninformative: a position is invalid (band at or below its null q95)"
    if not all(check.values()):
        return ("uninformative: the C_spec2025 pairs carry no document-specific content the "
                "instrument can read")
    if pc["passes"]:
        return ("calibrated: the overlap instrument passes its control in its document-specific "
                "form; the 2026 reads are interpreted against it")
    return ("the soft premise does not show at this site (not detected at the CI's width); the "
            "overlap read stays unused on C_partner")


def amendment47(c13: Path, c13c: Path) -> dict:
    """The rebuilt control (Amendment 47 item 3) on the 2025-04-11 Model Spec's pairs."""
    return contrast_read(c13, c13c, branch47, "ST_PLAN Amendment 47")


def family_scores(c13: Path, labels: dict[str, str]) -> dict:
    """Per model and document: P(right option) at dose 0 under ``none`` and the modal-right share,
    on the items ``labels`` covers."""
    out: dict = {}
    for m in MODELS:
        for doc in ("spec", "anth"):
            f = c13 / m / "family" / "none" / doc / "dl_chat_neutral.jsonl"
            if not f.exists():
                continue
            acc: dict[str, list] = {}
            modal: dict[str, list] = {}
            for r in _jsonl(f):
                sid = r["scenario_id"]
                if sid not in labels:
                    continue
                ls = sorted(r["option_logps"])
                p = np.exp([r["option_logps"][x] for x in ls])
                p = p / p.sum()
                by_opt = {r["order"][x]: q for x, q in zip(ls, p)}
                acc.setdefault(sid, []).append(by_opt.get(labels[sid], 0.0))
                modal.setdefault(sid, []).append(max(by_opt, key=by_opt.get) == labels[sid])
            pr = [float(np.mean(v)) for v in acc.values()]
            out[f"{m}/{doc}"] = {"n_items": len(pr), "P_right": T13.boot(pr),
                                 "modal_right_share": float(np.mean([np.mean(v)
                                                                     for v in modal.values()]))
                                 if modal else None}
    return out


def family_read(c13: Path, author: dict | None, key: dict | None, sub: dict | None) -> dict:
    """Amendment 40 item 5: the raters' reliability rule picks the labels of record; the
    construction label is read beside (and the author's 20 when the subagent's labels are of
    record)."""
    from deepsteer.kdg.schema import load_scenarios

    lab = {}
    for doc in ("spec", "anth"):
        for s in load_scenarios(DATA / f"st_c13_family_{doc}_v1.json")[0]:
            lab[s.id] = s.external_label["right_option_id"]
    out: dict = {"construction_label": True}
    labels, source = lab, "construction (raters pending)"
    a = None
    if author and key:
        a = {i: key["key"][i][x] for i, x in author.items() if x}
        if sub:
            agree = [a[i] == sub.get(i) for i in a]
            rel = float(np.mean(agree)) if agree else None
            out["reliability"] = {"n": len(agree), "agreement": rel, "bar": RELIABILITY_BAR,
                                  "disagreements": sorted(i for i in a if a[i] != sub.get(i))}
            labels, source = ((sub, "subagent on all held items (reliability met)")
                              if rel is not None and rel >= RELIABILITY_BAR else
                              (a, "author's 20 (reliability below the bar)"))
        else:
            labels, source = a, "author's 20 (no subagent ratings)"
    out["labels_of_record"] = source
    out.update(family_scores(c13, labels))
    if labels is not lab:
        out["construction_beside"] = family_scores(c13, lab)
        out["rater_vs_construction"] = {
            name: {"n": len(L), "agreement": float(np.mean([L[i] == lab.get(i) for i in L]))}
            for name, L in (("author", a or {}), ("subagent", sub or {})) if L}
    if a is not None and labels is not a:
        out["author20_beside"] = family_scores(c13, a)
    return out


DISPUTED = ("C13S-012", "C13S-013", "C13S-024", "C13A-026", "C13A-036")


def doc_marks(quotes_md: Path) -> dict[str, str]:
    """The author's ``doc:`` line per disputed item in the quote sheet (Amendment 45 item 2)."""
    import re

    out = {}
    for blk in re.split(r"(?m)^## (?=C13[SA]-\d{3})", quotes_md.read_text())[1:]:
        sid = blk.split("\n", 1)[0].strip()
        m = re.findall(r"(?m)^doc:[ \t]*(\w*)", blk)
        out[sid] = (m[0].lower() if m else "") or None
    return out


def amendment45(rater: dict, rater_key: dict, key: dict, author: dict, sub: dict,
                marks: dict | None) -> dict:
    """Amendment 45 item 4: the blind rater's agreement with the author's 20, the construction
    labels and the subagent; the five-item rule; the document-inconsistent construction labels."""
    from deepsteer.kdg.schema import load_scenarios

    lab, status = {}, {}
    for doc in ("spec", "anth"):
        for s in load_scenarios(DATA / f"st_c13_family_{doc}_v1.json")[0]:
            lab[s.id] = s.external_label["right_option_id"]
            status[s.id] = {o.option_id: o.norm_status for o in s.options}
    # the rater saw the subagent's items, so its letters decode through their key (``rater_key``,
    # all 38); the author's letters decode through the sheet's key
    g = {i: rater_key[i][r["letter"]] for i, r in rater.items() if r.get("letter")}
    a = {i: key["key"][i][x] for i, x in author.items() if x}

    def agree(x: dict, y: dict) -> dict:
        ids = sorted(set(x) & set(y))
        return {"n": len(ids), "agreement": float(np.mean([x[i] == y[i] for i in ids]))
                if ids else None}

    neutral = [i for i in DISPUTED if i in g and status[i][g[i]] == "neutral"]
    k = len(neutral)
    reading = ("not read (a disputed item is missing)" if any(i not in g for i in DISPUTED) else
               "sides with the author (ST-A28 R_a: the construction leans toward departure)"
               if k >= 4 else "sides with the construction (R_b: the author's check-first prior)"
               if k <= 1 else "split (R_c: the items are ambiguous)")
    out = {"rule": "ST_PLAN Amendment 45", "n_rated": len(g),
           "missing": sorted(set(rater) - set(g)),
           "vs_author_20": agree(g, a), "vs_construction_38": agree(g, lab),
           "vs_subagent_38": agree(g, sub), "disputed_neutral": neutral,
           "disputed_choice": {i: status[i].get(g.get(i)) for i in DISPUTED},
           "five_item_reading": reading}
    if marks is not None:
        out["author_doc_marks"] = marks
        out["construction_document_inconsistent"] = sorted(i for i, m in marks.items()
                                                           if m == "escalate")
    return out


def cell_a2(c13: Path) -> dict | None:
    import analyze_st2fx as FX

    d = c13 / "gpt_oss_20b"
    if not (d / "cellA2_selection.json").exists():
        return None
    sel = json.loads((d / "cellA2_selection.json").read_text())
    keep = set(sel["kept"])
    pn = {k: v for k, v in sel["p_violating_none"].items() if k in keep}
    out: dict = {"n_in_band": sel["n_in_band"], "n_read": len(keep), "shortfall": sel["shortfall"],
                 "P_violating_none": float(np.mean(list(pn.values()))) if pn else None}
    pa = {}
    for arm in ("beh2", "beh"):
        f = d / "cellA2" / arm / "dl_chat_neutral.jsonl"
        if f.exists():
            pa[arm] = FX.p_violating(_jsonl(f))
            s = sorted(set(pa[arm]) & set(pn))
            dd = T13.boot([pa[arm][i] - pn[i] for i in s])
            out[arm] = {"dP_violating": dd, "move": FX.move(dd["ci95"])}
    if "beh2" in out:
        out["reading"] = ("harm (ST-A23 R_a)" if out["beh2"]["move"] == "drop" else
                          "format (ST-A23 R_b)")
        if "beh" in pa:
            s = sorted(set(pa["beh2"]) & set(pa["beh"]))
            out["beh2_minus_beh"] = T13.boot([pa["beh2"][i] - pa["beh"][i] for i in s])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c13", type=Path, required=True)
    ap.add_argument("--author", type=Path, default=None)
    ap.add_argument("--key", type=Path, default=None)
    ap.add_argument("--subagent", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--amendment45", type=Path, default=None,
                    help="the blind rater's raw ratings (Amendment 45); needs --author, --key, "
                         "--subagent; reads outputs/c13_rating/a28_quotes.md for the doc marks")
    ap.add_argument("--amendment47", type=Path, default=None,
                    help="the c13c outputs: the rebuilt control on the 2025-04-11 Model Spec")
    ap.add_argument("--amendment44", action="store_true",
                    help="ST-A26's document-contrast read only (the family read is not touched)")
    ap.add_argument("--amendment42", action="store_true",
                    help="ST-A26's standardized read only (the family read is not touched)")
    a = ap.parse_args()
    rd = lambda p: json.loads(p.read_text()) if p else None  # noqa: E731
    if a.amendment45:
        qm = a.c13.parent / "c13_rating" / "a28_quotes.md"
        marks = doc_marks(qm) if qm.exists() else None
        rk = json.loads((a.c13.parent / "c13_rating" / "rater_key.json").read_text())
        res = amendment45(json.loads(a.amendment45.read_text())["ratings"], rk, rd(a.key),
                          rd(a.author), rd(a.subagent), marks)
        p = a.out or a.c13 / "c13_a45_read.json"
        p.write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    if a.amendment44 or a.amendment47:
        res = amendment47(a.c13, a.amendment47) if a.amendment47 else amendment44(a.c13)
        slim = json.loads(json.dumps(res, default=str))
        for mode in ("std", "raw"):
            for m in slim[mode]["models"].values():
                m.pop("u_spec_perp_boot"), m.pop("u_anth_perp_boot"), m.pop("u_vconst_spec_boot")
        p = a.out or a.c13 / ("c13_control2025_read.json" if a.amendment47
                              else "c13_doc_contrast_read.json")
        p.write_text(json.dumps(slim, indent=1))
        for mode in ("std", "raw"):
            x = res[mode]
            for m, r in x["models"].items():
                print(f"[{mode}] {m}: AUC {r['specificity_auc']:.3f} "
                      f"[{r['specificity_auc_ci95'][0]:.3f}, {r['specificity_auc_ci95'][1]:.3f}]; "
                      f"u(spec-perp) {r['u_spec_perp']:.3f}, u(anth-perp) {r['u_anth_perp']:.3f}; "
                      f"PF(contrast) {r['pf_contrast']:.3f}; valid {r['position_valid']}")
            print(f"[{mode}] positive control {x['positive_control']}; DiD {x['did_beside']}\n"
                  f"[{mode}] branch: {x['branch']}")
        print(f">> wrote {p}")
        return
    if a.amendment42:
        res = amendment42(a.c13)
        slim = json.loads(json.dumps(res, default=str))
        for mr in slim["reads"].values():
            for m in mr.values():
                for d in m["documents"].values():
                    d.pop("u_boot", None)
        p = a.out or a.c13 / "c13_std_read.json"
        p.write_text(json.dumps(slim, indent=1))
        for mode, mr in res["reads"].items():
            for m, r in mr.items():
                band = [round(x, 3) for x in r["heldout_band"]]
                print(f"[{mode}] {m}: PR {r['participation_ratio']:.1f} (raw "
                      f"{r['participation_ratio_raw']:.1f}), null {r['null_q50']:.3f} / "
                      f"{r['null_q95']:.3f}, band {band}, valid {r['position_valid']}, "
                      f"cos(spec, anth) {r['cos_documents']:.3f}")
                for doc, d in r["documents"].items():
                    print(f"     C_{doc}: PF {d['pf']:.3f} ({d['ladder_position']}); u {d['u']}")
            print(f"[{mode}] positive control: {res['positive_control'][mode]}")
        print(f">> invariance (OLMo-3-Think): {res['invariance_olmo3_think']['holds']}\n"
              f">> {res['branch']}\n>> wrote {p}")
        return
    reads = {m: model_read(a.c13, m) for m in MODELS if (a.c13 / m / "extract.npz").exists()}
    res: dict = {"models": reads}
    if len(reads) == 2:
        res["positive_control"] = positive_control(reads)
    res["family"] = family_read(a.c13, rd(a.author), rd(a.key), rd(a.subagent))
    res["cellA2"] = cell_a2(a.c13)
    p = a.out or a.c13 / "c13_read.json"
    slim = json.loads(json.dumps(res, default=str))
    for m in slim.get("models", {}).values():
        for d in m["documents"].values():
            d.pop("u_boot", None)
    p.write_text(json.dumps(slim, indent=1))
    for m, r in reads.items():
        band = [round(x, 3) for x in r["heldout_band"]]
        print(f"{m} (hidden {r['headline']}, PR {r['participation_ratio']:.1f}): null q50 "
              f"{r['null_q50']:.3f} q95 {r['null_q95']:.3f}, band {band}")
        for doc, d in r["documents"].items():
            print(f"   C_{doc}: PF {d['pf']:.3f} [{d['pf_ci95'][0]:.3f}, {d['pf_ci95'][1]:.3f}] "
                  f"({d['ladder_position']}); u {d['u']}; held-out sign share "
                  f"{d['heldout_sign_share']}")
    if "positive_control" in res:
        print(f">> positive control: {res['positive_control']}")
    print(f">> family: {res['family'].get('labels_of_record')}; cell A': "
          f"{(res['cellA2'] or {}).get('reading')}\n>> wrote {p}")


if __name__ == "__main__":
    main()
