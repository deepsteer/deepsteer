"""Amendment 15 Tier-B units: the OLMo request-twin lift (15.1) and the Qwen C1 read cell (15.2).

Both reuse ``papers/d3_decision_anatomy/scripts/c1_session.py`` as a subprocess (the harness of
record), driven by environment flags: ``REQUEST_TWINS_SET=union`` (Amendment 15.1 stimulus set),
``SWEEP=1``, ``RT_CAP=5`` for the pilot gate, ``STANDARDIZE=1`` for Qwen (A1), ``BOUNDARY=1`` on the
15.2 fallback. The pooled / alone / replication analysis is pure numpy on the saved per-twin arrays
(``rt_following`` tags each delta row with its twin, so the set split is exact).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import data
from .common import Ctx, REPO

C1 = REPO / "papers" / "d3_decision_anatomy" / "scripts" / "c1_session.py"
KS = [1, 3, 8, 16]


def _synth_c1_outputs(out: Path, key: str, rng, n_twins: int = 12, hidden: int = 64) -> tuple[Path, Path]:
    """Dry-run stand-in for a c1_session run: random per-twin arrays with the real file layout."""
    from deepsteer.datasets import get_request_twins_union
    foll = [a for _f, a, _b in get_request_twins_union()]
    idx = rng.choice(len(foll), n_twins, replace=False)
    full = -np.abs(rng.standard_normal(n_twins)) * 0.1
    save = dict(refusal=rng.standard_normal(hidden), harm=rng.standard_normal(hidden),
                Vbasis=np.linalg.qr(rng.standard_normal((hidden, 3)))[0], channel_act=rng.standard_normal((20, hidden)),
                layer=16, cell_full_deltas=full, cell_restricted_deltas=full * 0.3,
                cell_random_deltas=full * 0.01, full_judgment_deltas=np.abs(rng.standard_normal(30)) * 0.05,
                transport_control_deltas=np.abs(rng.standard_normal(30)) * 0.025,
                rt_following=np.array([foll[i] for i in idx], dtype=object), sweep_ks=np.array(KS),
                sweep_refusal=np.stack([full * f for f in (0.02, 0.31, 0.29, 0.28)]),
                sweep_judgment=np.stack([np.abs(rng.standard_normal(30)) * 0.05 * f for f in (0.05, 0.46, 0.59, 0.66)]),
                sweep_random=np.stack([full * 0.01 for _ in KS]), cell_harm_deltas=full * 0.3)
    npz = out / f"c1_inputs_{key}.npz"
    np.savez(npz, **save)
    js = out / f"c1_session_{key}.json"
    js.write_text(json.dumps({"key": key, "screen_counts": {"request_twins": n_twins, "compositional_twins": 30},
                              "cells": {"n_request_twins": n_twins, "sweep": {"shape_verdict": {"verdict": "harm_saturating"}}}}))
    return npz, js


def run_c1(ctx: Ctx, key: str, env: dict[str, str], rt_cap: int | None = None) -> tuple[Path, Path]:
    """Run c1_session for this model into ctx.out (or synthesize in dry-run). Returns (npz, json)."""
    if ctx.dry:
        return _synth_c1_outputs(ctx.out, key, ctx.rng, n_twins=(rt_cap or 12))
    e = {**os.environ, **env}
    e.pop("VALIDATE", None)
    if rt_cap:
        e["RT_CAP"] = str(rt_cap)
    cmd = [sys.executable, str(C1), "--model", ctx.spec.repo, "--key", key, "--layer", str(ctx.layer),
           "--out", str(ctx.out)]
    subprocess.run(cmd, env=e, check=True)
    return ctx.out / f"c1_inputs_{key}.npz", ctx.out / f"c1_session_{key}.json"


def pilot_gate(npz: Path, n_required: int = 4) -> dict:
    """Amendment 15.1 pilot gate: deltas finite; full-cell sign coherent on >= n_required of the twins."""
    z = np.load(npz, allow_pickle=True)
    full = np.asarray(z["cell_full_deltas"], float)
    finite = bool(np.isfinite(full).all() and np.isfinite(z["cell_restricted_deltas"]).all())
    sign = int(np.sum(np.sign(full) == np.sign(np.median(full)))) if full.size else 0
    return {"n": int(full.size), "finite": finite, "sign_coherent": sign,
            "passed": bool(finite and full.size >= 1 and sign >= min(n_required, full.size))}


def pooled_sweep_analysis(npz: Path, set_of: dict[str, str], rng, n_boot: int = 2000) -> dict:
    """Replication (original) / alone (w4) / pooled analyses of the rank sweep from per-twin arrays.

    R_refusal(k) = mean(sweep_refusal[k, idx]) / mean(full[idx]); R_judgment(k) from the (shared)
    judgment twins; shape verdict via the frozen Amendment-4 rule; bootstrap CIs over twins; the
    alone-vs-pooled sign rule on R_judgment(16) - R_refusal(16).
    """
    import sweep as sw
    z = np.load(npz, allow_pickle=True)
    full = np.asarray(z["cell_full_deltas"], float)
    harm = np.asarray(z["cell_harm_deltas"], float) if "cell_harm_deltas" in z.files else None
    tags = np.array([set_of.get(str(t), "unknown") for t in z["rt_following"]])
    ks = [int(k) for k in z["sweep_ks"]]
    SR, SJ = np.asarray(z["sweep_refusal"], float), np.asarray(z["sweep_judgment"], float)
    FJ = np.asarray(z["full_judgment_deltas"], float)
    Rj = {k: float(SJ[i].mean() / FJ.mean()) for i, k in enumerate(ks)}

    def analyse(idx: np.ndarray, label: str) -> dict:
        if idx.size < 3:
            return {"label": label, "n": int(idx.size), "status": "insufficient"}
        Rr = {k: float(SR[i, idx].mean() / full[idx].mean()) for i, k in enumerate(ks)}
        hr = float(harm[idx].mean() / full[idx].mean()) if harm is not None else float("nan")
        boots = {k: [] for k in ks}
        for _ in range(n_boot):
            j = rng.choice(idx, idx.size)
            for i, k in enumerate(ks):
                boots[k].append(SR[i, j].mean() / full[j].mean())
        ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
        verdict = sw.shape_verdict(Rr, Rj, hr if np.isfinite(hr) else Rr[min(ks)], ks)
        kmax = max(ks)
        return {"label": label, "n": int(idx.size), "R_refusal_k": Rr, "R_refusal_ci95": ci,
                "R_judgment_k": Rj, "harm_rank1_R": hr, "shape_verdict": verdict,
                "gap_sign_at_kmax": float(np.sign(Rj[kmax] - Rr[kmax]))}

    out = {"npz": str(npz), "n_total": int(full.size),
           "replication_original": analyse(np.where(tags == "original")[0], "original"),
           "alone_w4": analyse(np.where(tags == "w4")[0], "w4"),
           "pooled": analyse(np.arange(full.size), "pooled")}
    a, p = out["alone_w4"], out["pooled"]
    out["pooled_is_primary"] = bool("gap_sign_at_kmax" in a and "gap_sign_at_kmax" in p
                                    and a["gap_sign_at_kmax"] == p["gap_sign_at_kmax"])
    out["rule"] = ("pooled result is primary only if the alone (w4) result agrees in sign of "
                   "R_judgment(16) - R_refusal(16); else both reported, neither pooled")
    return out


def unit_15_1(ctx: Ctx) -> dict:
    """OLMo-3-Instruct: pilot gate (5 twins) -> full union run (folded-primary) -> pooled analysis."""
    env = {"REQUEST_TWINS_SET": "union", "SWEEP": "1"}
    env.pop("STANDARDIZE", None)                       # folded-primary (NI-4), the run-of-record env
    pnpz, _ = run_c1(ctx, "olmo3_w4_pilot", env, rt_cap=5)
    gate = pilot_gate(pnpz)
    ctx.manifest.gate("15.1_pilot_olmo3", gate["passed"], gate)
    ctx.manifest.add(pnpz, "15.1", ctx.key)
    if not gate["passed"]:
        ctx.manifest.status("15.1", "stopped_at_pilot", json.dumps(gate))
        return {"unit": "15.1", "pilot": gate, "status": "stopped_at_pilot"}
    npz, js = run_c1(ctx, "olmo3_w4", env)
    ctx.manifest.add(npz, "15.1", ctx.key); ctx.manifest.add(js, "15.1", ctx.key)
    res = pooled_sweep_analysis(npz, data.request_twin_sets(), ctx.rng, n_boot=ctx.n(2000, 100))
    res.update({"unit": "15.1", "pilot": gate, "env": env})
    ctx.save_json("pooled_sweep_olmo3_w4", "15.1", res)
    return res


def unit_15_2(ctx: Ctx) -> dict:
    """Qwen2.5-7B-Instruct: standardized C1 read cell with the screen gate + BOUNDARY fallback."""
    env = {"REQUEST_TWINS_SET": "union", "SWEEP": "1", "STANDARDIZE": "1", "ROBUSTIFY": "zscore"}
    pnpz, pjs = run_c1(ctx, "qwen25_w4_pilot", env, rt_cap=5)
    screened = json.loads(Path(pjs).read_text()).get("screen_counts", {}).get("request_twins", 0)
    gate = pilot_gate(pnpz)
    gate["screened_request_twins"] = int(screened)
    ctx.manifest.gate("15.2_pilot_qwen25", gate["passed"], gate)
    ctx.manifest.add(pnpz, "15.2", ctx.key)
    mode = "request"
    if screened < 12:
        env["BOUNDARY"] = "1"; mode = "boundary"
    npz, js = run_c1(ctx, "qwen25_w4", env)
    ctx.manifest.add(npz, "15.2", ctx.key); ctx.manifest.add(js, "15.2", ctx.key)
    rec = json.loads(Path(js).read_text())
    n = int(rec.get("cells", {}).get("n_request_twins", 0))
    res = {"unit": "15.2", "pilot": gate, "stimulus_mode": mode, "n_request_twins": n, "env": env,
           "shape_verdict": (rec.get("cells", {}).get("sweep") or {}).get("shape_verdict"),
           "status": "indeterminate_operating_point" if n < 12 else "ran"}
    ctx.save_json("qwen25_read_cell", "15.2", res)
    return res


UNITS = {"15.1": unit_15_1, "15.2": unit_15_2}
