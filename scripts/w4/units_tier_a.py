"""Amendment 14 Tier-A units. Each takes a :class:`Ctx`, saves its per-unit arrays through
``ctx.save``/``ctx.save_json`` (the mandatory outputs), and returns a small summary dict. Verdict
rules are NOT applied here (Session W4-3); the units compute and save the pre-registered
quantities and their nulls/positive controls so the verdicts are zero-GPU afterwards.
"""

from __future__ import annotations

import numpy as np

from deepsteer.geometry.participation import pr_profile
from deepsteer.geometry.reliability import (
    mean_diff,
    permutation_self_cosine_null,
    split_half_self_cosine,
)

from . import data
from .common import W4_OUT, Ctx, REPO


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _frac(Q: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(Q.T @ _unit(v)))


def _ortho(vs) -> np.ndarray:
    return np.linalg.qr(np.stack([_unit(v) for v in vs], 1))[0]


def _cov_null_frac(Q: np.ndarray, X: np.ndarray, rng, k: int = 500) -> float:
    """q95 of frac(Q, covariance-matched random direction) from a centered sample X (n, d)."""
    Xc = X - X.mean(0)
    n = Xc.shape[0]
    return float(np.percentile([_frac(Q, Xc.T @ rng.standard_normal(n)) for _ in range(k)], 95))


def split_half_paired(diffs: np.ndarray, n_splits: int, rng) -> dict:
    """Split-half self-cosine of a mean of per-pair difference vectors (the moral positive control)."""
    D = np.asarray(diffs, np.float64)
    r = np.empty(n_splits)
    for s in range(n_splits):
        i = rng.permutation(D.shape[0])
        h = D.shape[0] // 2
        r[s] = float(_unit(D[i[:h]].mean(0)) @ _unit(D[i[h:]].mean(0)))
    return {"median": float(np.median(r)), "ci95": [float(x) for x in np.percentile(r, [2.5, 97.5])]}


# ------------------------------------------------------------------ 14.1 -------------------------

def _split_half_unit(ctx: Ctx, fmt: str, tag: str) -> dict:
    hp = data.heretic_prompts(ctx.dry, n=ctx.n(400, 12))
    L = ctx.layer
    H = ctx.x.last_token_acts(hp["harmful"], fmt, L)
    S = ctx.x.last_token_acts(hp["harmless"], fmt, L)
    ctx.save(f"{tag}_samples", "14.1", harmful=H, harmless=S, layer=L, format=fmt,
             harmful_prompts=np.array(hp["harmful"], dtype=object),
             harmless_prompts=np.array(hp["harmless"], dtype=object))
    n_splits = ctx.n(200, 20)
    sh = split_half_self_cosine(H, S, n_splits=n_splits, rng=ctx.rng)
    null = permutation_self_cosine_null(H, S, n_perm=ctx.n(200, 20), rng=ctx.rng)
    # every resampled half-direction (Amendment 14.1 save list)
    halves = []
    for _ in range(n_splits):
        ip, ineg = ctx.rng.permutation(H.shape[0]), ctx.rng.permutation(S.shape[0])
        h1, h2 = H.shape[0] // 2, S.shape[0] // 2
        halves.append(np.stack([mean_diff(H[ip[:h1]], S[ineg[:h2]]), mean_diff(H[ip[h1:]], S[ineg[h2:]])]))
    ctx.save(f"{tag}_split_half_directions", "14.1", halves=np.stack(halves),
             per_split=sh["per_split"], per_perm=null["per_perm"])
    # positive control: moral-stories per-pair diffs (base/instruct saved by D1 phase 2), split-half
    pc = None
    key = "base" if ctx.spec.kind == "base" else "instruct"
    z = data.local_npz(data.D1_OUT / "phase2" / key / "diffs_moral_stories.npz")
    if z is not None and f"layer{L}" in z.files:
        pc = split_half_paired(z[f"layer{L}"], n_splits, ctx.rng)
    elif ctx.dry:
        pc = split_half_paired(ctx.rng.standard_normal((20, H.shape[1])) + 3.0, n_splits, ctx.rng)
    summ = {"unit": "14.1", "side": tag, "format": fmt, "layer": L, "n_harmful": int(H.shape[0]),
            "n_harmless": int(S.shape[0]), "direction": mean_diff(H, S),
            "split_half": {k: v for k, v in sh.items() if k != "per_split"},
            "permutation_null": {k: v for k, v in null.items() if k != "per_perm"},
            "positive_control_moral_stories_split_half": pc,
            "positive_control_available": pc is not None}
    ctx.save_json(f"{tag}_split_half", "14.1", summ)
    return summ


def unit_14_1_proto(ctx: Ctx) -> dict:
    """OLMo-3 base: per-sample proto-refusal activations + split-half reliability (raw format)."""
    return _split_half_unit(ctx, "raw", "proto_refusal")


def unit_14_1_gate(ctx: Ctx) -> dict:
    """OLMo-3-Instruct rider: split-half reliability of the instruct gate (chat format)."""
    return _split_half_unit(ctx, "chat", "gate")


# ------------------------------------------------------------------ 14.2 -------------------------

def unit_14_2(ctx: Ctx) -> dict:
    """Llama L12: severity/boundary harm bases (rank 1⊂2⊂4) vs the engage-driving moral PCs."""
    import sweep as sw
    L = ctx.layer
    sev = data.severity_twins(ctx.dry)
    bnd = data.boundary_twins(ctx.dry)
    sev_pairs = [(v, f) for (_fnd, _lvl, f, v) in sev]          # violating - following
    bnd_pairs = [(v, f) for (_fnd, _sub, f, v) in bnd]
    C_sev = ctx.x.pair_contrasts(sev_pairs, L)
    C_bnd = ctx.x.pair_contrasts(bnd_pairs, L)
    moral = data.moral_pairs(ctx.dry, n_cap=ctx.n(None, 6))
    all_pairs = [p for v in moral.values() for p in v]
    C_mor = ctx.x.pair_contrasts(all_pairs, L)
    ctrl = data.control_pairs(ctx.dry, n_cap=ctx.n(None, 6))
    C_ctl = {c: ctx.x.pair_contrasts(v, L) for c, v in ctrl.items()}

    # standardized frame with sigma from the SAVED C1 channel_act (bit-reproducible); dry: unit sigma
    z = data.local_npz(REPO / "papers/d3_decision_anatomy/outputs/c1_inputs_llama31_L12.npz")
    if z is not None and not ctx.dry:
        sig = np.sqrt(z["channel_act"].var(0)); sig = np.where(sig > 1e-8, sig, 1.0)
        saved_cos_harm_pc = None
        import json
        js = REPO / "papers/d3_decision_anatomy/outputs/c1_session_llama31_L12.json"
        rec = json.loads(js.read_text()) if js.exists() else {}
        saved_cos_harm_pc = (rec.get("cells", {}).get("sweep") or {}).get("cos_harm_pc")
        eng = (rec.get("cells", {}).get("engage_sweep") or {})
        harm = z["harm"]
    else:
        sig = np.ones(C_mor.shape[1]); saved_cos_harm_pc = None
        eng = {"ks": [1, 3, 8, 16], "R_engage_refusal_k": {"1": 0.02, "3": 0.48, "8": 0.63, "16": 0.53}}
        harm = _unit(C_sev.mean(0))
    std = lambda M: np.asarray(M, np.float64) / sig
    moral_pcs = sw.nested_pca_basis(std(C_mor), [16])[16]
    cos_harm_pc = sw.cos_harm_components(moral_pcs, std(harm), n=8)
    parity = None
    if saved_cos_harm_pc:
        parity = {"max_abs_diff": float(np.max(np.abs(np.array(cos_harm_pc[:len(saved_cos_harm_pc)])
                                                        - np.array(saved_cos_harm_pc)))),
                  "tolerance": 0.05}
    w = sw.engage_marginal_weights(eng["R_engage_refusal_k"], eng["ks"])
    ks = [1, 2, 4]
    H_sev = sw.nested_pca_basis(std(C_sev), ks)
    H_bnd = sw.nested_pca_basis(std(C_bnd), ks)
    cap_sev = sw.harm_capture_curve(H_sev, moral_pcs, w)
    cap_bnd = sw.harm_capture_curve(H_bnd, moral_pcs, w)
    # channel-matched control nulls: the same nested procedure on non-moral contrasts + random bases
    cap_ctl = {c: sw.harm_capture_curve(sw.nested_pca_basis(std(M), ks), moral_pcs, w)
               for c, M in C_ctl.items()}
    d = moral_pcs.shape[0]
    rand = {j: [] for j in ks}
    for _ in range(ctx.n(200, 10)):
        R = np.linalg.qr(ctx.rng.standard_normal((d, 4)))[0]
        cc = sw.harm_capture_curve({j: R[:, :j] for j in ks}, moral_pcs, w)
        for j in ks:
            rand[j].append(cc[j]["engage_weighted_capture"])
    rand_q95 = {j: float(np.percentile(v, 95)) for j, v in rand.items()}
    # positive control: the moral PCs' own split-half rank-4 basis
    i = ctx.rng.permutation(C_mor.shape[0]); h = C_mor.shape[0] // 2
    pcsA = sw.nested_pca_basis(std(C_mor[i[:h]]), [4])[4]
    self_cap = sw.harm_capture_curve({4: pcsA}, moral_pcs, w)[4]["engage_weighted_capture"]
    ctx.save("severity_contrasts_L12", "14.2", severity=C_sev, boundary=C_bnd, sigma=sig, layer=L,
             severity_levels=np.array([lvl for (_f, lvl, _a, _b) in sev]),
             boundary_sublevels=np.array([s for (_f, s, _a, _b) in bnd], dtype=object))
    ctx.save("moral_contrasts_L12", "14.2", contrasts=C_mor, moral_pcs=moral_pcs,
             sources=np.array([s for s, v in moral.items() for _ in v], dtype=object))
    ctx.save("harm_bases_L12", "14.2", **{f"severity_rank{j}": H_sev[j] for j in ks},
             **{f"boundary_rank{j}": H_bnd[j] for j in ks},
             **{f"control_{c}_rank{j}": sw.nested_pca_basis(std(M), ks)[j] for c, M in C_ctl.items() for j in ks})
    summ = {"unit": "14.2", "layer": L, "n_severity": len(sev_pairs), "n_boundary": len(bnd_pairs),
            "n_moral_pairs": len(all_pairs), "cos_harm_pc": cos_harm_pc, "parity_vs_saved": parity,
            "engage_marginal_weights": w, "capture_severity": cap_sev, "capture_boundary": cap_bnd,
            "capture_controls": cap_ctl, "random_basis_q95": rand_q95,
            "positive_control_self_capture_rank4": self_cap}
    ctx.save_json("harm_capture_L12", "14.2", summ)
    return summ


# ------------------------------------------------------------------ 14.3 -------------------------

def unit_14_3(ctx: Ctx) -> dict:
    """GPT-OSS: refusal direction + decision-token sample, P_prefill vs P_dec graded reads, A5 band."""
    import reasoning_cells as rc
    L = ctx.layer
    hp = data.heretic_prompts(ctx.dry, n=ctx.n(64, 8))
    Xh = ctx.x.end_of_prompt_acts(hp["harmful"], L)
    Xs = ctx.x.end_of_prompt_acts(hp["harmless"], L)
    refusal = mean_diff(Xh, Xs)
    X = np.concatenate([Xh, Xs], 0)
    prof = pr_profile(X, n_boot=ctx.n(2000, 50), n_null=ctx.n(200, 10), rng=ctx.rng)
    ctx.save("decision_token_sample", "14.3", channel_act=X, refusal=refusal, layer=L)

    # A5 band-below-null half: V_moral sources at the harmony decision token, held-one-out vs cov null
    moral = data.moral_pairs(ctx.dry, n_cap=ctx.n(60, 6))
    src = {}
    for s, pairs in moral.items():
        A = ctx.x.end_of_prompt_acts([a for a, _ in pairs], L)
        B = ctx.x.end_of_prompt_acts([b for _, b in pairs], L)
        src[s] = mean_diff(A, B)
    names = list(src)
    band = {}
    for s in names:
        Q = _ortho([src[o] for o in names if o != s])
        band[s] = {"frac": _frac(Q, src[s]), "null_q95": _cov_null_frac(Q, X, ctx.rng, k=ctx.n(500, 50))}
    band_min = min(v["frac"] for v in band.values())
    null_max = max(v["null_q95"] for v in band.values())
    ctx.save("vmoral_sources_decision_token", "14.3", **{s: v for s, v in src.items()})

    # graded disengage re-read: the same violating items (boundary twins, ceiling-refusing at baseline)
    bnd = data.boundary_twins(ctx.dry)
    viol = [v for (_f, _s, _foll, v) in bnd]
    foll = [f for (_f, _s, f, _v) in bnd]
    series = rc.GRADED_EXCULPATING
    mnt = ctx.n(512, 16)
    items, rows = [], []
    for v in viol:
        base = ctx.x.harmony_prefill_reads(v, "", L, mnt)
        if base["final_refused"] is True or (ctx.dry and len(items) < 3):
            items.append(v)
            rows.append({"item": v, "baseline": base,
                         "graded": [ctx.x.harmony_prefill_reads(v, p, L, mnt) for p in series]})
        if len(items) >= ctx.n(10, 3):
            break
    eng_rows = []
    for f in foll[:ctx.n(7, 2)]:
        eng_rows.append({"item": f, "baseline": ctx.x.harmony_prefill_reads(f, "", L, mnt),
                         "engage": ctx.x.harmony_prefill_reads(f, rc.INCULPATING, L, mnt)})

    def proj(a):
        return None if a is None else float(np.asarray(a) @ refusal)

    def stat_for(site: str) -> dict:
        base_p, grad_p, flips = [], [], []
        for r in rows:
            b = proj(r["baseline"][site])
            g = [proj(x[site]) for x in r["graded"]]
            if b is None or any(x is None for x in g):
                continue
            base_p.append(b); grad_p.append(g)
            flips.append([0 if x["final_refused"] is None else int(not x["final_refused"]) for x in r["graded"]])
        if len(base_p) < 2:
            return {"n_items": len(base_p), "verdict": "insufficient"}
        st = rc.graded_disengage_stat(base_p, grad_p, flips)
        moves = [g[-1] - b for b, g in zip(base_p, grad_p)]
        bt = [np.mean(ctx.rng.choice(moves, len(moves))) for _ in range(ctx.n(2000, 100))]
        st["mean_move_at_max_ci95"] = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
        st["n_items_read"] = len(base_p)
        return st

    stats = {"P_prefill": stat_for("p_prefill"), "P_dec": stat_for("p_dec")}
    # random-direction null at P_dec: chance monotone fraction / mean move
    rand_mono, rand_move = [], []
    Xc = X - X.mean(0)
    for _ in range(ctx.n(200, 10)):
        rd = _unit(Xc.T @ ctx.rng.standard_normal(Xc.shape[0]))
        mono = 0; tot = 0; mv = []
        for r in rows:
            b = r["baseline"]["p_dec"]; g = [x["p_dec"] for x in r["graded"]]
            if b is None or any(x is None for x in g):
                continue
            pb = float(np.asarray(b) @ rd); pg = [float(np.asarray(x) @ rd) for x in g]
            tot += 1; mono += int(all(np.diff([pb] + pg) <= 0)); mv.append(pg[-1] - pb)
        if tot:
            rand_mono.append(mono / tot); rand_move.append(float(np.mean(mv)))
    null = {"monotone_frac_q95": float(np.percentile(rand_mono, 95)) if rand_mono else None,
            "mean_move_q05": float(np.percentile(rand_move, 5)) if rand_move else None,
            "n_random_dirs": len(rand_mono)}
    engage_stat = {"n": len(eng_rows),
                   "P_dec_mean_move_toward_refuse": float(np.mean([proj(r["engage"]["p_dec"]) - proj(r["baseline"]["p_dec"])
                                                                   for r in eng_rows if r["engage"]["p_dec"] is not None
                                                                   and r["baseline"]["p_dec"] is not None]))
                   if eng_rows else None,
                   "flips_to_refuse": int(sum(1 for r in eng_rows if r["engage"]["final_refused"] is True
                                              and r["baseline"]["final_refused"] is False))}
    # per-rollout saves
    def pack(site):
        return np.array([[np.nan] * X.shape[1] if x[site] is None else np.asarray(x[site])
                         for r in rows for x in [r["baseline"], *r["graded"]]])
    ctx.save("graded_reads", "14.3", p_prefill=pack("p_prefill"), p_dec=pack("p_dec"),
             items=np.array(items, dtype=object), n_strengths=len(series),
             final_refused=np.array([[np.nan if x["final_refused"] is None else float(x["final_refused"])
                                      for x in [r["baseline"], *r["graded"]]] for r in rows]),
             token_ids=np.array([np.asarray(x["ids"]) for r in rows for x in [r["baseline"], *r["graded"]]],
                                dtype=object))
    summ = {"unit": "14.3", "layer": L, "pr_profile_decision_token": prof, "band_held_one_out": band,
            "band_min": band_min, "null_q95_max": null_max, "band_below_null": bool(band_min < null_max),
            "graded_stats": stats, "random_direction_null_P_dec": null, "engage_arm_P_dec": engage_stat,
            "n_items": len(items), "replication_flip_at_max_P_behavior": stats["P_prefill"].get("frac_flip_at_max")}
    ctx.save_json("decision_token_reread", "14.3", summ)
    return summ


# ------------------------------------------------------------------ 14.4 -------------------------

def unit_14_4(ctx: Ctx) -> dict:
    """Think / GPT-OSS: per-rollout P0–P3 activations, PR per position, band-below-null, saves."""
    from deepsteer.reasoning.think_io import CoTFormat
    L = ctx.layer
    fmt = CoTFormat.HARMONY_ANALYSIS if ctx.spec.kind == "reasoning_moe" else CoTFormat.THINK_TAGS
    window = 16 if ctx.spec.kind == "reasoning_moe" else 256
    mnt = (1024 if ctx.spec.kind == "reasoning_moe" else window + 64) if not ctx.dry else 24
    hp = data.heretic_prompts(ctx.dry, n=ctx.n(400, 8))
    n_gen = ctx.n(32, 4)
    pos = ("p0", "p1", "p2", "p2_full", "p3")
    acts = {p: {"harmful": [], "harmless": []} for p in pos}
    flags = {"harmful": [], "harmless": []}
    ids = {"harmful": [], "harmless": []}
    for side in ("harmful", "harmless"):
        for prompt in hp[side][:n_gen]:
            r = ctx.x.rollout_positions(prompt, L, mnt, window, fmt)
            for p in pos:
                acts[p][side].append(r[p])
            flags[side].append([r["closed"], r["win_ok"], r["p3"] is not None])
            ids[side].append(np.asarray(r["ids"]))
    # prompt-side positions for the remaining prompts ride the same rollout path with tiny budget
    def stack(lst):
        keep = [a for a in lst if a is not None]
        return np.stack(keep) if keep else np.zeros((0, ctx.x.d))
    pr = {}; refusal_dirs = {}
    for p in pos:
        Hh, Hs = stack(acts[p]["harmful"]), stack(acts[p]["harmless"])
        if min(Hh.shape[0], Hs.shape[0]) >= 3:
            Xp = np.concatenate([Hh, Hs], 0)
            pr[p] = pr_profile(Xp, n_boot=ctx.n(2000, 30), n_null=ctx.n(200, 10), rng=ctx.rng)
            refusal_dirs[p] = mean_diff(Hh, Hs)
            pr[p]["n_harmful"], pr[p]["n_harmless"] = int(Hh.shape[0]), int(Hs.shape[0])
        else:
            pr[p] = {"status": "unmeasured", "n_harmful": int(Hh.shape[0]), "n_harmless": int(Hs.shape[0])}
    # band-below-null per position: moral sources through the same rollout pipeline (P0/P1 prompt-side)
    moral = data.moral_pairs(ctx.dry, n_cap=ctx.n(32, 3))
    band = {}
    src_acts = {p: {} for p in pos}
    for s, pairs in moral.items():
        for p in pos:
            src_acts[p][s] = ([], [])
        for a, b in pairs:
            ra = ctx.x.rollout_positions(a, L, mnt, window, fmt)
            rb = ctx.x.rollout_positions(b, L, mnt, window, fmt)
            for p in pos:
                if ra[p] is not None and rb[p] is not None:
                    src_acts[p][s][0].append(ra[p]); src_acts[p][s][1].append(rb[p])
    for p in pos:
        dirs = {s: mean_diff(np.stack(A), np.stack(B)) for s, (A, B) in src_acts[p].items()
                if len(A) >= 2}
        if len(dirs) < 3 or "status" in pr[p]:
            band[p] = {"status": "unmeasured"}
            continue
        Xp = np.concatenate([stack(acts[p]["harmful"]), stack(acts[p]["harmless"])], 0)
        rows = {}
        for s in dirs:
            Q = _ortho([dirs[o] for o in dirs if o != s])
            rows[s] = {"frac": _frac(Q, dirs[s]), "null_q95": _cov_null_frac(Q, Xp, ctx.rng, k=ctx.n(500, 30))}
        band[p] = {"held_one_out": rows, "band_min": min(v["frac"] for v in rows.values()),
                   "null_q95_max": max(v["null_q95"] for v in rows.values())}
        band[p]["band_below_null"] = bool(band[p]["band_min"] < band[p]["null_q95_max"])
    ctx.save("p0p3_rollouts", "14.4", layer=L, window_n=window, max_new_tokens=mnt,
             **{f"{p}_{side}": stack(acts[p][side]) for p in pos for side in ("harmful", "harmless")},
             flags_harmful=np.array(flags["harmful"]), flags_harmless=np.array(flags["harmless"]),
             ids_harmful=np.array(ids["harmful"], dtype=object), ids_harmless=np.array(ids["harmless"], dtype=object))
    for p, d in refusal_dirs.items():
        ctx.save(f"refusal_{p.upper()}", "14.4", refusal=d, layer=L)   # closes MISSING_ARTIFACTS A3 (Think)
    if ctx.spec.kind != "reasoning_moe" or ctx.dry:
        mft = data.mft_pairs(ctx.dry, n_cap=ctx.n(None, 4))
        dirs = {f: _unit(ctx.x.raw_pair_diffs(v, L).mean(0)) for f, v in mft.items()}
        ctx.save("mft_directions", "14.4", **{f"{f}_layer{L}": v for f, v in dirs.items()})  # closes A1
    else:
        mft = data.mft_pairs(ctx.dry, n_cap=None)
        dirs = {f: _unit(ctx.x.raw_pair_diffs(v, L).mean(0)) for f, v in mft.items()}
        ctx.save("mft_directions", "14.4", **{f"{f}_layer{L}": v for f, v in dirs.items()})
    summ = {"unit": "14.4", "layer": L, "window_n": window, "max_new_tokens": mnt, "n_gen_per_side": n_gen,
            "pr_by_position": pr, "band_by_position": band,
            "closed_rate_harmful": float(np.mean([f[0] for f in flags["harmful"]])) if flags["harmful"] else None,
            "closed_rate_harmless": float(np.mean([f[0] for f in flags["harmless"]])) if flags["harmless"] else None}
    ctx.save_json("pr_audit", "14.4", summ)
    return summ


# ------------------------------------------------------------------ 14.5 -------------------------

def unit_14_5(ctx: Ctx) -> dict:
    """OLMo-3-Instruct: reconciled R3(iii) cross-ablation with paired difference-CIs and a bail rule."""
    from b1_judgment_direction import build_prompt, is_refusal, parse_verdict
    L = ctx.layer
    # Pre-registered sizes: refusal direction from the Heretic 400/400 train set, outcomes on the
    # 100 held-out harmful_eval requests (the first pod run used 64/64 + 64 via a shared cap).
    hp = data.heretic_prompts(ctx.dry, n=None if not ctx.dry else 8)
    harmful = hp["harmful_eval"][:ctx.n(100, 6)]
    items = data.ethics_items(ctx.dry, n_cap=ctx.n(120, 6))
    flips = [bool(b) for b in ctx.rng.integers(0, 2, len(items))]
    # Harness parity with Paper 6's 0.575 baseline: chat-template generation for both outcomes.
    # The first pod run generated from the bare prompt and floored refusal at 0.03 (bail invalid).
    fmt = "chat" if ctx.spec.kind != "base" else "raw"
    refusal = mean_diff(ctx.x.last_token_acts(hp["harmful"], fmt, L),
                        ctx.x.last_token_acts(hp["harmless"], fmt, L))
    z = data.local_npz(REPO / "papers/d2_decision_coupling/outputs/olmo3/b1_judgment_dir_olmo3.npz")
    judgment = _unit(z["judgment_dir"]) if (z is not None and not ctx.dry) else _unit(ctx.rng.standard_normal(refusal.shape[0]))
    persona = _unit(ctx.x.raw_pair_diffs(data.persona_pairs(ctx.dry, n_cap=ctx.n(None, 6)), L).mean(0))
    randoms = [_unit(ctx.rng.standard_normal(refusal.shape[0])) for _ in range(ctx.n(5, 2))]

    def refusal_outcomes():
        return np.array([int(is_refusal(ctx.x.generate(p, 64, fmt=fmt))) for p in harmful])

    def judgment_outcomes():
        out = []
        for (scn, gt), f in zip(items, flips):
            v = parse_verdict(ctx.x.generate(build_prompt(scn, f)[0], 6, fmt=fmt))
            out.append(int(v == gt))
        return np.array(out)

    base_r, base_j = refusal_outcomes(), judgment_outcomes()
    base_rate = float(base_r.mean())
    bail = (not ctx.dry) and base_rate < 0.40
    conds = {"judgment_decision": judgment, "refusal": refusal, "persona": persona}
    for i, r in enumerate(randoms):
        conds[f"random_{i}"] = r
    outcomes = {"baseline": {"refusal": base_r, "judgment": base_j}}
    if not bail:
        for name, dvec in conds.items():
            with ctx.x.ablate(L, dvec):
                outcomes[name] = {"refusal": refusal_outcomes(), "judgment": judgment_outcomes()}

    def paired_diff_ci(a, b, c, d):
        """CI of mean(a-b) - mean(c-d) over prompts (paired per prompt)."""
        x = (a - b) - (c - d)
        bt = [np.mean(ctx.rng.choice(x, len(x))) for _ in range(ctx.n(2000, 100))]
        return {"point": float(x.mean()), "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}

    res = {"unit": "14.5", "layer": L, "n_harmful": len(harmful), "n_judgment": len(items),
           "prompt_format": fmt, "n_refusal_direction": [len(hp["harmful"]), len(hp["harmless"])],
           "outcome_harness": "compliance_gap.greenblatt._classify_response (opening-refusal rule)",
           "baseline_refusal_rate": base_rate, "baseline_judgment_acc": float(base_j.mean()),
           "bail_floor_limited": bool(bail), "bail_rule": "baseline refusal >= 0.40 required",
           "mde_rate_diff_n_harmful": float(1.96 * np.sqrt(2 * 0.25 / max(1, len(harmful)))),
           "mde_rate_diff_n_judgment": float(1.96 * np.sqrt(2 * 0.25 / max(1, len(items))))}
    if not bail:
        rand_keys = [k for k in conds if k.startswith("random_")]
        rand_r = np.mean([outcomes[k]["refusal"] for k in rand_keys], 0)
        rand_j = np.mean([outcomes[k]["judgment"] for k in rand_keys], 0)
        res["rates"] = {k: {"refusal_rate": float(v["refusal"].mean()), "judgment_acc": float(v["judgment"].mean())}
                        for k, v in outcomes.items()}
        res["arrow_judgment_to_refusal"] = paired_diff_ci(outcomes["judgment_decision"]["refusal"], base_r, rand_r, base_r)
        res["arrow_refusal_to_judgment"] = paired_diff_ci(outcomes["refusal"]["judgment"], base_j, rand_j, base_j)
        res["persona_control"] = {"refusal_delta": float(outcomes["persona"]["refusal"].mean() - base_rate),
                                  "judgment_delta": float(outcomes["persona"]["judgment"].mean() - base_j.mean())}
    ctx.save("cross_ablation_outcomes", "14.5", layer=L,
             **{f"{k}_refusal": v["refusal"] for k, v in outcomes.items()},
             **{f"{k}_judgment": v["judgment"] for k, v in outcomes.items()},
             refusal_dir=refusal, judgment_dir=judgment, persona_dir=persona, random_dirs=np.stack(randoms),
             harmful_prompts=np.array(harmful, dtype=object))
    ctx.save_json("cross_ablation", "14.5", res)
    return res


def unit_14_5_gen(ctx: Ctx) -> dict:
    """A8 discriminator (rider 2026-09-12): re-generate under the SAVED 14.5 directions and keep the
    texts. Baseline / refusal / judgment_decision / random_0 on the same 100 prompts, greedy; the
    classifier outcome per text is compared with the saved outcomes (determinism check, logged)."""
    from b1_judgment_direction import is_refusal
    src = ctx.out / "cross_ablation_outcomes.npz"
    if not src.exists():                       # partial rerun: read the run of record's save
        src = W4_OUT / ctx.key / "cross_ablation_outcomes.npz"
    z = np.load(src, allow_pickle=True)
    L = int(z["layer"]); prompts = [str(p) for p in z["harmful_prompts"]]
    fmt = "chat" if ctx.spec.kind != "base" else "raw"
    conds = {"refusal": z["refusal_dir"], "judgment_decision": z["judgment_dir"], "random_0": z["random_dirs"][0]}
    texts, outcomes, agree = {}, {}, {}
    texts["baseline"] = [ctx.x.generate(p, 64, fmt=fmt) for p in prompts]
    for name, d in conds.items():
        with ctx.x.ablate(L, np.asarray(d, float)):
            texts[name] = [ctx.x.generate(p, 64, fmt=fmt) for p in prompts]
    for name, ts in texts.items():
        outcomes[name] = np.array([int(is_refusal(t)) for t in ts])
        saved = z[f"{name}_refusal"]
        agree[name] = float((outcomes[name] == saved).mean()) if saved.shape == outcomes[name].shape else None
    ctx.save("cross_ablation_generations", "14.5_gen", layer=L, source_npz=str(src.name),
             harmful_prompts=np.array(prompts, dtype=object),
             **{f"{k}_text": np.array(v, dtype=object) for k, v in texts.items()},
             **{f"{k}_refusal": v for k, v in outcomes.items()})
    res = {"unit": "14.5_gen", "layer": L, "n": len(prompts), "prompt_format": fmt, "source": str(src.name),
           "rates": {k: float(v.mean()) for k, v in outcomes.items()},
           "agreement_with_saved_outcomes": agree,
           "note": "ANOMALIES A8 discriminator; texts saved for the coherence tally. No verdict here."}
    ctx.save_json("cross_ablation_generations", "14.5_gen", res)
    return res


# ------------------------------------------------------------------ 14.6 -------------------------

def unit_14_6(ctx: Ctx) -> dict:
    """Per-position activation samples + PR profiles (CI, PR/d, nulls); OLMo-Instruct extra saves."""
    L = ctx.layer
    if ctx.spec.kind == "reasoning_moe":
        # the decision-token sample is produced by 14.3; here add a content position (t_inst analog)
        hp = data.heretic_prompts(ctx.dry, n=ctx.n(64, 8))
        X = ctx.x.end_of_prompt_acts(hp["harmful"] + hp["harmless"], L)
        prof = {"harmony_decision_token": pr_profile(X, n_boot=ctx.n(2000, 30), n_null=ctx.n(200, 10), rng=ctx.rng)}
        ctx.save("position_samples", "14.6", harmony_decision_token=X, layer=L)
        summ = {"unit": "14.6", "layer": L, "pr_profiles_raw": prof}
        ctx.save_json("pr_profiles", "14.6", summ)
        return summ
    moral = data.moral_pairs(ctx.dry, n_cap=ctx.n(40, 6))
    texts = [t for v in moral.values() for pair in v for t in pair]
    acts = ctx.x.position_acts(texts, L)
    prof_raw, prof_std = {}, {}
    for p, X in acts.items():
        prof_raw[p] = pr_profile(X, n_boot=ctx.n(2000, 30), n_null=ctx.n(200, 10), rng=ctx.rng)
        sig = X.std(0); sig = np.where(sig > 1e-8, sig, 1.0)
        prof_std[p] = pr_profile(X / sig, n_boot=ctx.n(2000, 30), n_null=ctx.n(200, 10), rng=ctx.rng)
    ctx.save("position_samples", "14.6", layer=L, **acts, texts=np.array(texts, dtype=object))
    summ = {"unit": "14.6", "layer": L, "n_texts": len(texts), "pr_profiles_raw": prof_raw,
            "pr_profiles_standardized": prof_std}
    if ctx.key == "olmo3_instruct":
        # MISSING_ARTIFACTS A4: instruct per-pair diff arrays for fables/ethics (raw, L16)
        for s in ("fables", "ethics"):
            ctx.save(f"axis_diffs_{s}", "14.6", **{f"layer{L}": ctx.x.raw_pair_diffs(moral[s], L)})
        # MISSING_ARTIFACTS Amendment 2: mean_content slices for the refusal + judgment prompt sets
        from b1_judgment_direction import build_prompt
        hp = data.heretic_prompts(ctx.dry, n=ctx.n(64, 8))
        items = data.ethics_items(ctx.dry, n_cap=ctx.n(120, 6))
        jprompts = [build_prompt(s, False)[0] for s, _ in items]
        mc = {k: ctx.x.position_acts(v, L)["mean_content"]
              for k, v in (("harmful", hp["harmful"]), ("harmless", hp["harmless"]), ("judgment", jprompts))}
        ctx.save("mean_content_slices", "14.6", layer=L, **mc)
        summ["extra_saves"] = ["axis_diffs_fables", "axis_diffs_ethics", "mean_content_slices"]
    ctx.save_json("pr_profiles", "14.6", summ)
    return summ


def unit_14_6b(ctx: Ctx) -> dict:
    """Llama: reply-inversion specificity null (harm direction vs 20 random directions, matched norm)."""
    L = ctx.layer
    hp = data.heretic_prompts(ctx.dry, n=ctx.n(100, 8))
    H = ctx.x.last_token_acts(hp["harmful"], "chat", L)   # t_inst analog read via the same extractor
    S = ctx.x.last_token_acts(hp["harmless"], "chat", L)
    harm = mean_diff(H, S)
    norm = ctx.x.residual_norm(hp["harmless_eval"][:ctx.n(20, 4)], L)
    evalset = hp["harmless_eval"][:ctx.n(100, 6)]
    clean = ctx.x.steer_margins(evalset, np.zeros_like(harm), L)
    out = {"unit": "14.6b", "layer": L, "n_eval": len(evalset), "residual_norm": norm, "alphas": {}}
    n_rand = ctx.n(20, 3)
    rand_dirs = [_unit(ctx.rng.standard_normal(harm.shape[0])) for _ in range(n_rand)]
    margins = {"clean": clean}
    for alpha in (0.5, 1.0):
        hm = ctx.x.steer_margins(evalset, alpha * norm * harm, L)
        rm = np.array([ctx.x.steer_margins(evalset, alpha * norm * r, L) for r in rand_dirs])
        margins[f"harm_a{alpha}"] = hm; margins[f"random_a{alpha}"] = rm
        flip = lambda m: float(np.nanmean((m > 0) & (clean <= 0)))
        shift = lambda m: float(np.nanmean(m - clean))
        out["alphas"][str(alpha)] = {"harm_flip_frac": flip(hm), "harm_margin_shift": shift(hm),
                                     "random_flip_q95": float(np.percentile([flip(r) for r in rm], 95)),
                                     "random_shift_q95": float(np.percentile([shift(r) for r in rm], 95)),
                                     "n_random": n_rand}
    ctx.save("reply_inversion_margins", "14.6b", layer=L, harm_dir=harm, random_dirs=np.stack(rand_dirs),
             **{k: np.asarray(v) for k, v in margins.items()})
    ctx.save_json("reply_inversion_null", "14.6b", out)
    return out


UNITS = {"14.1_proto": unit_14_1_proto, "14.1_gate": unit_14_1_gate, "14.2": unit_14_2, "14.3": unit_14_3,
         "14.5_gen": unit_14_5_gen,
         "14.4": unit_14_4, "14.5": unit_14_5, "14.6": unit_14_6, "14.6b": unit_14_6b}
