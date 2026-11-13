#!/usr/bin/env python3
"""MACHIAVELLI's throughput on one A100: a fluid model from a run's own rate curve (zero GPU).

    python3 mach_throughput_model.py SESSION_LOG EPISODES_JSONL [LONGEST_GAME_DECISIONS]

``D(A)``, the engine's decisions per hour at ``A`` active episodes, is binned from the runner's
60-second ``mach decisions N active A`` lines; each active episode progresses at ``D(n)/n`` (vLLM
serves its running batch evenly, and requests beyond the KV cache wait in order). Episodes carry
the run's own decision counts, longest first. The finalist pod's MACHIAVELLI run of 2026-11-13 is
the source of `MACH_THROUGHPUT_NOTE.md`; the model reproduces that run's active-episode curve.
"""
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

L = Path(sys.argv[1]).read_text().splitlines()
pts = [(float(m.group(1)), int(m.group(2)), int(m.group(3)))
       for x in L if (m := re.search(r"\s(\d+)s mach decisions (\d+) active (\d+)", x))]

# D(A): aggregate decisions per hour at A active episodes (the run's 60 s samples, binned)
bins = collections.defaultdict(list)
for (t0, d0, a0), (t1, d1, a1) in zip(pts, pts[1:]):
    if t1 > t0 and a0 == a1:
        bins[a0].append((d1 - d0) / (t1 - t0) * 3600)
xs, ys = [], []
for lo in [4, 5, 6, 7, 8, 9] + list(range(10, 121, 10)):
    hi = lo + 1 if lo < 10 else lo + 10
    v = [r for a, rs in bins.items() if lo <= a < hi for r in rs]
    if len(v) >= 3:
        xs.append(st.mean(a for a, rs in bins.items() if lo <= a < hi for _ in rs))
        ys.append(st.mean(v))
D_MAX = st.mean(ys[-6:])  # the plateau


def D(a: float) -> float:
    if a <= 0:
        return 0.0
    if a <= xs[0]:
        return ys[0] * a / xs[0]
    if a >= xs[-1]:
        return D_MAX  # beyond the run: KV-limited, extra episodes only wait
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= a <= x1:
            return y0 + (y1 - y0) * (a - x0) / (x1 - x0)


# episodes: this run's finished decision counts; luminous-underground's four from the log's tail
eps = [json.loads(x) for x in open(sys.argv[2])]
LUM = float(sys.argv[3]) if len(sys.argv) > 3 else 590.0
base = [(e["arm"], e["game"], e["episode"], e["decisions"]) for e in eps if e["status"] == "done"]
games = {g for _, g, _, _ in base} | {"luminous-underground"}
for arm in ("base", "cand"):
    for k in (0, 1):
        if not any(a == arm and g == "luminous-underground" and e == k for a, g, e, _ in base):
            base.append((arm, "luminous-underground", k, LUM))


def episodes(n_cand: int, subset=None):
    out = []
    for a, g, e, d in base:
        if subset is not None and g not in subset:
            continue
        if a == "base":
            out.append(("base", g, e, d))
        else:
            for c in range(n_cand):
                out.append((f"cand{c}", g, e, d))
    return sorted(out, key=lambda r: -r[3])  # longest first (MCH's cell_jobs order)


def simulate(ep, cap=10**9, prio_games=(), prio_batch=60, tail_swap=None, dt=0.005):
    """Fluid model: active episodes share D(n); priority episodes skip the queue and run at the
    per-episode rate of an uncontended batch of ``min(n, prio_batch)``; tail_swap = (n_at, speedup,
    restart_h). Returns the finish time (h) and the active count over time."""
    queue = [list(r) for r in ep]
    active, t, swapped, trace = [], 0.0, False, []
    while queue or active:
        while queue and len(active) < cap:
            active.append(queue.pop(0))
        n = len(active)
        if tail_swap and not swapped and n <= tail_swap[0] and not queue:
            t += tail_swap[2]
            swapped = True
        sp = tail_swap[1] if swapped else 1.0
        pr = [r for r in active if r[1] in prio_games]
        rest = [r for r in active if r[1] not in prio_games]
        tot = D(n) * sp
        rp = D(min(n, prio_batch)) / min(n, prio_batch) * sp if pr else 0.0
        rp = min(rp, tot / n * n if rest == [] else rp)
        rr = (tot - rp * len(pr)) / len(rest) if rest else 0.0
        for r in active:
            r[3] -= (rp if r in pr else rr) * dt
        active = [r for r in active if r[3] > 0]
        t += dt
        trace.append((t, len(active) + len(queue)))
    return t, trace


if __name__ == "__main__":
    print("curve", [(round(a), round(d)) for a, d in zip(xs, ys)], "plateau", round(D_MAX))
    print("episodes", len(episodes(1)), "decisions", sum(r[3] for r in episodes(1)))
