#!/usr/bin/env python3
"""The D_rb rider's local read (ST_PLAN Amendments 34-35; zero GPU).

    python3 papers/st_live_reference/scripts/analyze_st2drb.py \
        --drb papers/st_live_reference/outputs/st2fx/gpt_oss_20b/drb [--t11 <T1.1 dir>]

At hidden 13, clause_end, per trace (arm - ``none``) on the same tokens:

- **Directions.** D_rb(spec) (set a), D_rb(pool) (set b, the one L-beh-3 trains on), D_h1 (set
  c): the mean shift, unnormalized; item bootstrap (items resampled with their traces) and
  split-half reliability (cos between random item halves; band q05 to q50).
- **Ladder.** Isotropic floor sqrt(2/(pi d)); covariance-matched null, |cos| of direction pairs
  drawn from N(0, Sigma) with Sigma from ``none``'s clause_end sample (q50, q95); the
  channel-chance line sqrt(2/(pi PR)).
- **Cosines** of D_rb(spec) with V_qwen, V_harm, care_harm, the anchor axis, D_h1, D_rb(pool).
- **Build gate** (35 item 4): the bail (D_rb(pool)'s split-half q05 at or below the null q95); the
  positive control's order on set (a) against D_rb(pool) (L-beh and epoch 3 tied, both above
  L-anchor, L-anchor above L-beh-2); agreement (signed cos(spec, pool), bootstrap 2.5% above the
  null q95).
- **The D_h1 rule** (35 item 5) and the training vector of record (its norm against 267.3, its
  sha256), written to ``--vec-out``.
- **Bank parity** (34 item 3): set (c)'s ``none`` prefill against T1.1's bank on the same rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]

OUTS = HERE.parent / "outputs"
SEED, N_BOOT, N_NULL = 0, 1000, 10000
ALPHA_SITE = 267.301171875  # 0.2 x the site's median norm (build_st2_inputs)
SPEC_OF_RECORD = {"beh": 0.1919, "beh_epoch3": 0.188, "anchor": 0.0236, "beh2": -0.093}


def load(drb: Path, arm: str) -> dict:
    z = np.load(drb / f"{arm}.npz")
    ce = list(z["classes"]).index("clause_end")
    return {"keys": [str(k) for k in z["keys"]], "sets": [str(s) for s in z["sets"]],
            "ce": z["mean_cls_h13"][:, ce].astype(np.float64), "z": z}


def item_of(key: str) -> str:
    return key.split("|")[1]


def diffs(base: dict, arm: dict, which: str) -> dict:
    """Per item: the (arm - none) clause_end means of its traces in set ``which``."""
    b = {k: v for k, s, v in zip(base["keys"], base["sets"], base["ce"]) if s == which}
    out: dict[str, list] = {}
    for k, s, v in zip(arm["keys"], arm["sets"], arm["ce"]):
        if s == which and k in b:
            out.setdefault(item_of(k), []).append(v - b[k])
    return {i: np.array(v) for i, v in out.items()}


def mean_dir(d: dict, items) -> np.ndarray:
    return np.concatenate([d[i] for i in items]).mean(0)


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-30)


def boot_dirs(d: dict, rng) -> np.ndarray:
    items = sorted(d)
    return np.array([mean_dir(d, rng.choice(items, len(items))) for _ in range(N_BOOT)])


def split_half(d: dict, rng) -> np.ndarray:
    items = np.array(sorted(d))
    out = []
    for _ in range(N_BOOT):
        p = rng.permutation(len(items))
        h = len(items) // 2
        out.append(float(unit(mean_dir(d, items[p[:h]])) @ unit(mean_dir(d, items[p[h:]]))))
    return np.array(out)


def participation_ratio(x: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False) ** 2
    return float(s.sum() ** 2 / max((s ** 2).sum(), 1e-30))


def null_pairs(sample: np.ndarray, rng) -> np.ndarray:
    """|cos| of direction pairs drawn from N(0, Sigma): v = Xc^T g, g ~ N(0, I_n)."""
    xc = sample - sample.mean(0, keepdims=True)
    out = []
    for _ in range(N_NULL // 500):
        v1 = rng.standard_normal((500, len(xc))) @ xc
        v2 = rng.standard_normal((500, len(xc))) @ xc
        out.append(np.abs((v1 * v2).sum(1) / np.linalg.norm(v1, axis=1)
                          / np.linalg.norm(v2, axis=1)))
    return np.concatenate(out)


def q(x, p) -> float:
    return float(np.quantile(x, p))


def references() -> dict:
    z = np.load(OUTS / "st2_inputs" / "gpt_oss_20b" / "st2_inputs.npz")
    c = np.load(OUTS / "st2c_inputs" / "gpt_oss_20b" / "st2c_inputs.npz")
    return {"V_qwen": c["V_qwen"], "V_harm": z["V_harm"], "care_harm": z["care_harm"],
            "anchor_axis": z["anchor_axis"]}


def bank_parity(none: dict, t11: Path) -> dict:
    """Set (c): the prefill's clause_end count and hidden-13 mean against T1.1's bank row (bank
    files are T1.1's chunks of ``CHUNK`` rows, opened only where set (c)'s rows live)."""
    import t11_driver as T11

    keys = {int(k.split("|")[2]): k for k, s in zip(none["keys"], none["sets"]) if s == "h1"}
    want = {r: none["ce"][none["keys"].index(k)] for r, k in keys.items()}
    cnt = {r: int(none["z"]["counts"][none["keys"].index(k), 0]) for r, k in keys.items()}
    got: dict[int, list] = {}
    for f in sorted({r // T11.CHUNK for r in want}):
        z = np.load(t11 / "bank" / f"bank_{f:04d}.npz")
        m = np.isin(z["row"], list(want)) & ((z["cls"] & 2) > 0)
        for r, v in zip(z["row"][m], z["resid"][m][:, 2]):  # hidden 7, 10, 13: index 2
            got.setdefault(int(r), []).append(v.astype(np.float64))
    rows = sorted(set(want) & set(got))
    dc = [len(got[r]) - cnt[r] for r in rows]
    dv = [float(np.abs(np.mean(got[r], 0) - want[r]).max() / max(np.abs(want[r]).max(), 1e-9))
          for r in rows if len(want[r]) == len(got[r][0])]
    return {"rows": len(rows), "of": len(want), "count_mismatches": int(sum(x != 0 for x in dc)),
            "max_rel_abs_diff_mean": max(dv) if dv else None}


def read(drb: Path, t11: Path | None) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(SEED)
    base = load(drb, "none")
    beh = load(drb, "beh")
    D = {s: diffs(base, beh, s) for s in ("spec", "pool", "h1")}
    vec = {s: mean_dir(D[s], sorted(D[s])) for s in D}
    boots = {s: boot_dirs(D[s], rng) for s in D}
    sh = {s: split_half(D[s], rng) for s in ("spec", "pool")}
    sample = base["z"]["ce_sample_h13"].astype(np.float64)
    pr = participation_ratio(sample)
    nul = null_pairs(sample, rng)
    d = sample.shape[1]
    ladder = {"floor_isotropic": float(np.sqrt(2 / (np.pi * d))), "null_q50": q(nul, .5),
              "null_q95": q(nul, .95), "participation_ratio": pr,
              "channel_chance": float(np.sqrt(2 / (np.pi * pr))), "n_sample": len(sample)}
    out: dict = {"ladder": ladder, "directions": {}}
    for s in D:
        nb = np.linalg.norm(boots[s], axis=1)
        out["directions"][s] = {"n_items": len(D[s]),
                                "n_traces": int(sum(len(v) for v in D[s].values())),
                                "norm": float(np.linalg.norm(vec[s])),
                                "norm_ci95": [q(nb, .025), q(nb, .975)]}
        if s in sh:
            out["directions"][s]["split_half"] = {"q05": q(sh[s], .05), "q50": q(sh[s], .5)}
    refs = {"D_h1": vec["h1"], "D_rb_pool": vec["pool"]}
    if d == 2880:  # the references of record (a dry or synthetic width never reads them)
        refs = {**references(), **refs}
    if any(len(v) != d for v in refs.values()):
        raise SystemExit("a reference direction is not at the sample's width")
    cos = {}
    for name, x in refs.items():
        bx = boots["h1"] if name == "D_h1" else boots["pool"] if name == "D_rb_pool" else None
        bc = [float(unit(a) @ unit(bx[i] if bx is not None else x))
              for i, a in enumerate(boots["spec"])]
        cos[name] = {"cos": float(unit(vec["spec"]) @ unit(x)), "ci95": [q(bc, .025), q(bc, .975)]}
    out["cosines_D_rb_spec"] = cos
    # the build gate (Amendment 35 item 4)
    bail = out["directions"]["pool"]["split_half"]["q05"] <= ladder["null_q95"]
    pool_u = unit(vec["pool"])
    proj = {}
    for arm in SPEC_OF_RECORD:
        a = beh if arm == "beh" else load(drb, arm)
        da = diffs(base, a, "spec")
        proj[arm] = {"on_pool": float(mean_dir(da, sorted(da)) @ pool_u),
                     "on_spec": float(mean_dir(da, sorted(da)) @ unit(vec["spec"])),
                     "specificity_of_record": SPEC_OF_RECORD[arm], "n_traces":
                         int(sum(len(v) for v in da.values()))}
    p = {k: v["on_pool"] for k, v in proj.items()}
    order_ok = bool(min(p["beh"], p["beh_epoch3"]) > p["anchor"] > p["beh2"])
    agree_b = [float(unit(a) @ unit(b)) for a, b in zip(boots["spec"], boots["pool"])]
    agree = {"cos": cos["D_rb_pool"]["cos"], "ci95": [q(agree_b, .025), q(agree_b, .975)]}
    agree["passes"] = bool(agree["ci95"][0] > ladder["null_q95"])
    gate = {"bail": bool(bail), "order_ok": order_ok, "agreement": agree,
            "build": bool(not bail and order_ok and agree["passes"])}
    gate["verdict"] = ("bail: D_rb(pool) is not distinguishable from noise" if bail else
                       "build L-beh-3" if gate["build"] else
                       "report to the author with the predicted outcome first")
    out["positive_control"] = proj
    out["build_gate"] = gate
    # the D_h1 rule (item 5)
    h1u = unit(vec["h1"])
    c_ph = float(pool_u @ h1u)
    bh = [float(unit(a) @ unit(b)) for a, b in zip(boots["pool"], boots["h1"])]
    ortho = bool(abs(c_ph) > ladder["null_q95"]
                 and abs(c_ph) >= out["directions"]["pool"]["split_half"]["q05"])
    v = vec["pool"] - (vec["pool"] @ h1u) * h1u if ortho else vec["pool"]
    v = v.astype(np.float32)
    out["d_h1_rule"] = {"cos_pool_h1": c_ph, "ci95": [q(bh, .025), q(bh, .975)],
                        "orthogonalize": ortho}
    out["training_vector"] = {"kind": "D_rb(pool) orthogonal to D_h1" if ortho else "D_rb(pool)",
                              "norm": float(np.linalg.norm(v)),
                              "norm_over_site_alpha": float(np.linalg.norm(v) / ALPHA_SITE),
                              "sha256_float32": hashlib.sha256(v.tobytes()).hexdigest()}
    if t11 is not None:
        out["bank_parity"] = bank_parity(base, t11)
    out["bias_directions"] = [  # estimator-traps: which side of each rule a known bias favors
        {"bias": "bootstrap and split-half cosines are attenuated (resampled or half-sample "
                 "directions are noisier)", "rule": "agreement (bootstrap lower bound)",
         "favors": "not building (against agreement)"},
        {"bias": "the D_h1 rule compares a full-sample cosine with a half-sample band",
         "rule": "D_h1 rule", "favors": "orthogonalizing (the protection-preserving side)"},
        {"bias": "the same attenuation lowers D_rb(pool)'s split-half q05",
         "rule": "bail", "favors": "bailing (against building)"}]
    return out, v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drb", type=Path, required=True)
    ap.add_argument("--t11", type=Path, default=OUTS / "t11" / "gpt_oss_20b")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--vec-out", type=Path, default=None)
    a = ap.parse_args()
    res, v = read(a.drb, a.t11 if (a.t11 / "bank").exists() else None)
    p = a.out or a.drb / "st2drb_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    if a.vec_out:
        a.vec_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.vec_out, d_train=v)
    lad, g = res["ladder"], res["build_gate"]
    print(f"ladder: floor {lad['floor_isotropic']:.3f}, null q50 {lad['null_q50']:.3f} q95 "
          f"{lad['null_q95']:.3f}, channel {lad['channel_chance']:.3f} "
          f"(PR {lad['participation_ratio']:.1f})")
    for s, d in res["directions"].items():
        print(f"D[{s}] |.| {d['norm']:.2f} items {d['n_items']} split-half {d.get('split_half')}")
    for n, c in res["cosines_D_rb_spec"].items():
        print(f"cos(D_rb(spec), {n}) {c['cos']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]")
    pc = {k: round(x["on_pool"], 3) for k, x in res["positive_control"].items()}
    print(f"positive control on D_rb(pool) {json.dumps(pc)}")
    print(f">> gate: {g['verdict']}; D_h1 rule {res['d_h1_rule']}; vector {res['training_vector']}")
    print(f">> wrote {p}")


if __name__ == "__main__":
    main()
