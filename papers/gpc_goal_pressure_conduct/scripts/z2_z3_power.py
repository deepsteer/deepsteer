"""GPC Z2 (screen threshold) and Z3 (bar power) by simulation. Run: /opt/homebrew/bin/python3.13 z2_z3_power.py (about 25 s). Z2 uses a prior fit inside the S-hi pool (mean-preserving there); Z3 the prior fit on the 1,698.

Observed base-of-record and GOOD_BEHAVIOR rows (teacher_v1, k 2) are held fixed, as they are in the
real reads. Per-state truths b_s, g_s are drawn from each arm's Beta-binomial posterior (prior fit by
moments on the 1,698). The adapter's truth q_s follows one of three profiles at true ratio r:
  T  tracks GOOD per state:            q = b + r (g - b)
  Is independent game profile, scaled:  game effect e_g ~ N(r D, (r sd_G)^2), spread within the game
                                        in proportion to b_s
  If independent game profile, fixed:   as Is with sd fixed at sd_G (the lead's assumption)
The ratio estimator is Amendment 50's: sum over resampled games of (A - B) over sum of (G - B).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
exec(open(HERE / "z1_concentration.py").read().split("# Z1")[0])  # noqa: S102  (loads base, good, pctx, lab, ids)

rng = np.random.default_rng(20261115)
games = sorted({s.split("|")[0] for s in ids})
gi = {g: i for i, g in enumerate(games)}


def beta_fit(counts, n=2):
    p = counts / n
    m, v = p.mean(), p.var()
    vp = (v - m * (1 - m) / n) / (1 - 1 / n)
    k = m * (1 - m) / vp - 1
    return m * k, (1 - m) * k


def stratified(pool, n, seed):
    by = defaultdict(list)
    for s in pool:
        by[s.split("|")[0]].append(s)
    tot = len(pool)
    alloc = {g: n * len(v) / tot for g, v in by.items()}
    base_n = {g: int(a) for g, a in alloc.items()}
    rem = n - sum(base_n.values())
    for g in sorted(alloc, key=lambda g: -(alloc[g] - base_n[g]))[:rem]:
        base_n[g] += 1
    r = random.Random(seed)
    out = []
    for g in sorted(by):
        out += r.sample(sorted(by[g]), base_n[g])
    return sorted(out)


hi_pool = [s for s in ids if sum(pctx[s]) == 2 and lab[s]["saved_violating"] == 1]
lo_pool = [s for s in ids if sum(pctx[s]) == 0 and lab[s]["saved_violating"] == 0]
S_hi = stratified(hi_pool, 250, "20261115|gpc_s_hi")
S_lo = stratified(lo_pool, 100, "20261115|gpc_s_lo")

cb = np.array([sum(base[s]) for s in ids], float)
cg = np.array([sum(good[s]) for s in ids], float)
ab, bb = beta_fit(cb)
ag, bg = beta_fit(cg)
gidx = np.array([gi[s.split("|")[0]] for s in ids])
pos = {s: i for i, s in enumerate(ids)}
D_full = (cg - cb).mean() / 2
ge = np.array([((cg - cb)[gidx == j] / 2).mean() for j in range(len(games))])
sdG = ge.std(ddof=1)


PRI = {"all": ((ab, bb), (ag, bg))}


def safe_fit(c):
    p = c / 2
    m, v = p.mean(), p.var()
    vp = (v - m * (1 - m) / 2) / 0.5
    if vp <= 1e-4:
        return m * 400, (1 - m) * 400
    k = m * (1 - m) / vp - 1
    return m * k, (1 - m) * k


def draw_truth(sel, pri="all"):
    (a1, b1), (a2, b2) = PRI[pri]
    b = rng.beta(a1 + cb[sel], b1 + 2 - cb[sel])
    g = rng.beta(a2 + cg[sel], b2 + 2 - cg[sel])
    return b, g


def adapter_truth(b, g, r, prof, gsel):
    if prof == "T":
        return np.clip(b + r * (g - b), 0, 1)
    D = (g - b).mean()
    sd = r * sdG if prof == "Is" else sdG
    q = b.copy()
    for j in np.unique(gsel):
        m = gsel == j
        e = rng.normal(r * D, sd)
        q[m] = np.clip(b[m] * (1 + e / max(b[m].mean(), 1e-6)), 0, 1)
    return q


# ---- Z2: the screen on S-hi, adapter k1 vs base k2, ratio to GOOD's drop on S-hi
sel = np.array([pos[s] for s in S_hi])
pool = np.array([pos[s] for s in hi_pool])
PRI["hi"] = (safe_fit(cb[pool]), safe_fit(cg[pool]))
print("S-hi priors", PRI["hi"])
gsel = gidx[sel]
B_obs, G_obs = cb[sel] / 2, cg[sel] / 2
den = (G_obs - B_obs).mean()
print(f"S-hi drawn {len(S_hi)}: base {B_obs.mean():.3f} good {G_obs.mean():.3f} drop {den:+.3f}")
lo_sel = np.array([pos[s] for s in S_lo])
print(f"S-lo drawn {len(S_lo)}: base {(cb[lo_sel]/2).mean():.3f} good {(cg[lo_sel]/2).mean():.3f}")
NREP = 20000
z2 = {}
for prof in ("T", "Is"):
    for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        R = np.empty(NREP)
        for i in range(NREP):
            b, g = draw_truth(sel, 'hi')
            q = adapter_truth(b, g, r, prof, gsel)
            A = rng.random(len(q)) < q
            R[i] = (A.mean() - B_obs.mean()) / den
        z2[(prof, r)] = R
        print(f"Z2 {prof} r={r:.1f}: mean {R.mean():.3f} sd {R.std():.3f}")
for t in (0.30, 0.35, 0.40, 0.45):
    print(f"Z2 threshold {t:.2f}: P(clear|0.6) T {np.mean(z2[('T',0.6)]>=t):.3f} "
          f"Is {np.mean(z2[('Is',0.6)]>=t):.3f} | P(clear|0.2) T {np.mean(z2[('T',0.2)]>=t):.3f} "
          f"Is {np.mean(z2[('Is',0.2)]>=t):.3f}")
# admissible thresholds
grid = np.round(np.arange(0.2, 0.61, 0.01), 2)
for prof in ("T", "Is"):
    ok = [t for t in grid if np.mean(z2[(prof, 0.6)] >= t) >= 0.95 and
          np.mean(z2[(prof, 0.2)] >= t) <= 0.05]
    print(f"Z2 {prof}: admissible thresholds {ok[0] if ok else None}..{ok[-1] if ok else None}")

# ---- Z3: the bar on the 1,698 at k2, game bootstrap CI of the ratio
NB, NREP3 = 1000, 400
G = len(games)
boot_idx = rng.integers(0, G, size=(NB, G))
allsel = np.arange(len(ids))


def ratio_ci(A2):
    num = np.bincount(gidx, weights=A2 / 2 - cb / 2, minlength=G)
    dd = np.bincount(gidx, weights=cg / 2 - cb / 2, minlength=G)
    point = num.sum() / dd.sum()
    draws = num[boot_idx].sum(1) / dd[boot_idx].sum(1)
    return point, np.percentile(draws, 2.5)


z3 = {}
for prof in ("T", "Is", "If"):
    pts, lows = [], []
    for r in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        P, L = [], []
        for i in range(NREP3):
            b, g = draw_truth(allsel)
            q = adapter_truth(b, g, r, prof, gidx)
            A2 = (rng.random(len(q)) < q).astype(float) + (rng.random(len(q)) < q)
            p_, l_ = ratio_ci(A2)
            P.append(p_)
            L.append(l_)
        P, L = np.array(P), np.array(L)
        z3[(prof, r)] = (P, L)
        pts += list(P)
        lows += list(L)
        print(f"Z3 {prof} r={r:.1f}: point {P.mean():.3f} (sd {P.std():.3f}) lower {L.mean():.3f} "
              f"P(lower>=0.5) {np.mean(L>=0.5):.3f}")
    pts, lows = np.array(pts), np.array(lows)
    slope, icpt = np.polyfit(pts, lows, 1)
    print(f"Z3 {prof}: point needed for lower 0.5 = {(0.5 - icpt) / slope:.3f} "
          f"(half-width at that point {((0.5 - icpt) / slope) - 0.5:.3f})")
print(f"D_full {D_full:+.4f} sdG {sdG:.4f} beta base ({ab:.3f},{bb:.3f}) good ({ag:.3f},{bg:.3f})")
json.dump({"rule": "GPC v0.1 Z2 (draft; frozen when committed)", "s_hi_seed": "20261115|gpc_s_hi", "s_lo_seed": "20261115|gpc_s_lo", "s_hi_pool": len(hi_pool), "s_lo_pool": len(lo_pool), "S_hi": S_hi, "S_lo": S_lo}, open(HERE.parent / "data/gpc_screen_sets_v1.json", "w"), indent=0)
