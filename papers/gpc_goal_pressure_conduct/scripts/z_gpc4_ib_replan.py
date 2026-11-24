#!/usr/bin/env python3
"""The lead's ImpossibleBench re-plan inputs (2026-11-23 1:50 PM HST), zero GPU, counts and timings only:
GPC-A18's accounting from launch 7's copied logs, the second derivation on rehearsal 12's stub items, the minimal
scaffold's price under three designs, the first-tasks rule restated at each task count, and the output cap's share.

No transcript text is read into the output or printed. The `.eval` samples and the sample buffer are read for
event counts, retries, errors, pending flags and timestamps; the first user message is hashed (as the proxy does)
and never kept.

    python z_gpc4_ib_replan.py   # writes outputs/gpc4_prep/ib_replan.json
"""

from __future__ import annotations

import bisect
import collections
import datetime
import hashlib
import json
import math
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpc4_pod as P4  # noqa: E402
import gpc4_vm as V  # noqa: E402

GPC = HERE.parent
L7 = GPC / "outputs/gpc4_v1_launch7/host_ib_g1_tools_oneoff_base"
R12 = Path.home() / ".gpc4/gpc4-20261123T022257Z/rehearsal_results"
OUT = GPC / "outputs/gpc4_prep/ib_replan.json"
RUN = "ib_g1_tools_oneoff_base"
CAP, CAP2 = 16384, 8192
PRICE = 1.18  # $ an A100-h (the ledger's Thunder rate, §13.5)


def _ts(s: str) -> float:
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _eval_samples(f: Path) -> list[dict]:
    z = zipfile.ZipFile(f)
    return [json.loads(P4._eval_member(z, i)) for i in z.infolist() if i.filename.startswith("samples/")]  # noqa: SLF001


def _model_events(s: dict) -> list[dict]:
    return [e for e in s.get("events", []) if e.get("event") == "model"]


def scratch(rows: list[dict]) -> dict:
    """The lead's scratch, re-derived over the whole proxy log (every run the host served)."""
    ct = [(r.get("usage") or {}).get("completion_tokens") or 0 for r in rows]
    pairs = collections.Counter((r["seed"], (r.get("usage") or {}).get("prompt_tokens")) for r in rows)
    starts = [r["t"] for r in rows]  # launch 7's proxy logged the request's start (a5326cd9)
    ends = [r["t"] + r["latency_s"] for r in rows]
    span_starts, span_full = (max(starts) - min(starts)) / 3600, (max(ends) - min(starts)) / 3600
    return {"calls": len(rows), "seeds": len({r["seed"] for r in rows}),
            "calls_590s_or_more": sum(r["latency_s"] >= 590 for r in rows),
            "their_completion_tokens": sum(c for r, c in zip(rows, ct) if r["latency_s"] >= 590),
            "completion_tokens": sum(ct), "length_finishes": sum(r["finish_reason"] == "length" for r in rows),
            "completion_at_cap": sum(c >= CAP for c in ct), "distinct_seed_prompt_pairs": len(pairs),
            "repeats_within_a_seed": sum(v - 1 for v in pairs.values()),
            "span_h_first_to_last_start": span_starts, "tokens_per_s_over_that": sum(ct) / (span_starts * 3600),
            "span_h_first_start_to_last_end": span_full, "tokens_per_s_over_that_full": sum(ct) / (span_full * 3600)}


def accounting(rows: list[dict]) -> dict:
    """The tools gate's proxy calls against Inspect's own records: the .eval (finished samples), the sample buffer
    (running ones), the trace log (every model call's span and every response the client received)."""
    m = [r for r in rows if r["run"] == RUN]
    end = lambda r: r["t"] + r["latency_s"]  # noqa: E731
    tr = [json.loads(x) for x in open(L7 / "iso" / RUN / "xdg_data/inspect_ai/traces/trace-68434.log")]
    t_end = max(_ts(r["timestamp"]) for r in tr)
    hooks = sorted((_ts(r["timestamp"]), re.search(r"sample=(\S+)", r["message"]).group(1))
                   for r in tr if r.get("function") == "response_hook")
    ht = [h for h, _ in hooks]
    used, unrec = set(), []
    for r in sorted((r for r in m if end(r) <= t_end), key=end):
        c = [j for j in range(bisect.bisect_left(ht, end(r) - 1), bisect.bisect_right(ht, end(r) + 1)) if j not in used]
        if c:
            used.add(min(c, key=lambda j: abs(ht[j] - end(r))))
        else:
            unrec.append(r)
    by_hook = collections.Counter(hooks[j][1] for j in used)
    fin = _eval_samples(next((L7 / "out" / RUN / "inspect_logs").glob("*.eval")))
    fin_ev = {s["id"]: _model_events(s) for s in fin}
    con = sqlite3.connect(next((L7 / "iso" / RUN / "xdg_data/inspect_ai/samplebuffer").glob("*/*.db")))
    last: dict = {}
    for sid, d in con.execute("select sample_id, data from events order by id"):
        e = json.loads(d)
        if e.get("event") == "model":
            last[e.get("uuid") or e.get("id")] = (sid, e)
    buf_ev = collections.defaultdict(list)
    for sid, e in last.values():
        buf_ev[sid].append(e)
    allev = [e for v in fin_ev.values() for e in v] + [e for v in buf_ev.values() for e in v]
    retries = sum(e.get("retries") or 0 for e in allev)
    errors = sum(bool(e.get("error")) for e in allev)
    pending = sum(bool(e.get("pending")) for e in allev)
    after_logged = sum(1 for e in allev if e.get("completed") and _ts(e["completed"]) > t_end)
    after_proxy = sum(end(r) > t_end for r in m)
    missing = sorted(set(by_hook) - set(fin_ev) - set(buf_ev))
    fin_answered = sum(len(v) for v in fin_ev.values()) - sum(bool(e.get("error")) for v in fin_ev.values() for e in v)
    buf_answered = sum(len(v) for v in buf_ev.values()) - sum(bool(e.get("error")) or bool(e.get("pending"))
                                                              for v in buf_ev.values() for e in v)
    miss_calls = sum(by_hook[s] for s in missing) + (after_proxy - after_logged)
    ab600 = sum(r["latency_s"] >= 600 for r in unrec)
    other = len(unrec) - ab600
    table = {"answered, the 8 finished samples (.eval)": fin_answered,
             "answered, the 5 samples running at the copy (the buffer)": buf_answered,
             f"a sample in the trace and in neither file ({', '.join(missing)})": miss_calls,
             "abandoned at the client's 600 s timeout, generated on, then sent again": ab600,
             "Inspect's own re-queries that its buffer does not log": 0,
             "seed collisions across samples (extra calls)": 0,
             "unexplained (no client received it, under 600 s)": other}
    # the seed rule: (run, sha256 of the first user message, assistant-message count)
    convs = set()
    for s in fin:
        u = next(x for x in s["messages"] if x["role"] == "user")
        convs.add(hashlib.sha256((u["content"] if isinstance(u["content"], str) else str(u["content"])).encode())
                  .hexdigest()[:12])
    conv = next(iter(convs))
    s2t = {V.seed_of(RUN, conv, t): t for t in range(400)}
    t0 = [r for r in m if s2t.get(r["seed"]) == 0]
    pairs = collections.defaultdict(list)
    for r in m:
        pairs[(r["seed"], (r.get("usage") or {}).get("prompt_tokens"))].append(r)
    rep = [v for v in pairs.values() if len(v) > 1]
    return {"proxy_calls": len(m), "table": table, "table_sum": sum(table.values()),
            "inspect": {"model_events_finished": sum(len(v) for v in fin_ev.values()),
                        "model_events_running": sum(len(v) for v in buf_ev.values()), "retries": retries,
                        "gave_up": errors, "in_flight_at_copy": pending,
                        "failed_attempts_retries_plus_gave_up": retries + errors,
                        "trace_model_spans": sum(1 for r in tr
                                                 if r.get("action") == "Model" and r.get("event") == "enter"),
                        "responses_received_in_trace": len(hooks)},
            "join": {"proxy_calls_by_trace_end": sum(end(r) <= t_end for r in m), "received_matched": len(used),
                     "not_received": len(unrec),
                     "not_received_latency_s": sorted(round(r["latency_s"]) for r in unrec),
                     "after_trace_end_proxy": after_proxy, "after_trace_end_inspect_logged": after_logged},
            "seed": {"distinct_first_user_hashes_over_finished_samples": len(convs),
                     "proxy_calls_on_the_shared_hash": sum(r["seed"] in s2t for r in m),
                     "seeds": len({r["seed"] for r in m}), "turn0_calls": len(t0),
                     "turn0_prompt_sizes": len({(r.get("usage") or {}).get("prompt_tokens") for r in t0}),
                     "turn0_completion_sizes": len({(r.get("usage") or {}).get("completion_tokens") for r in t0})},
            "repeats": {"total": sum(len(v) - 1 for v in rep),
                        "turn0_identical_across_samples": sum(len(v) - 1 for v in rep if s2t.get(v[0]["seed"]) == 0),
                        "in_groups_with_a_600s_call": sum(len(v) - 1 for v in rep
                                                          if any(r["latency_s"] >= 600 for r in v)
                                                          and s2t.get(v[0]["seed"]) != 0)}}


def rehearsal12() -> dict:
    """Proxy calls (and generations: the rows not `joined`) per Inspect model event, on the stub items."""
    out = {}
    for item in sorted(R12.glob("*ib*")):
        pj = item / "proxy.jsonl"
        if not pj.exists():
            continue
        rows = [json.loads(x) for x in pj.read_text().splitlines()]
        chat = [r for r in rows if str(r.get("path", "")).endswith("chat/completions")]
        ev = [e for f in item.glob("inspect_logs/*.eval") for s in _eval_samples(f) for e in _model_events(s)]
        out[item.name] = {"proxy": len(chat), "joined": sum(bool(r.get("joined")) for r in chat),
                          "events": len(ev), "retries": sum(e.get("retries") or 0 for e in ev)}
    t = {k: sum(v[k] for v in out.values()) for k in ("proxy", "joined", "events", "retries")}
    return {"items": out, "total": t, "one_to_one": t["proxy"] == t["events"] + t["retries"]}


def smoke_minimal(rows: list[dict]) -> dict:
    """Launch 7's four minimal-scaffold smoke samples (the arms c1, cbounds, cctx, sum; not the base), taken under
    the 600 s loop: each distinct (seed, prompt size) is one attempt; its first generation is kept."""
    att, allc = [], []
    for run in sorted({r["run"] for r in rows if r["run"].startswith("ib_smoke_minimal")}):
        seen = set()
        for r in sorted((r for r in rows if r["run"] == run), key=lambda r: r["t"]):
            k = (r["seed"], r["usage"].get("prompt_tokens"))
            allc.append(r["usage"].get("completion_tokens"))
            if k not in seen:
                seen.add(k)
                att.append({"run": run, "prompt": r["usage"].get("prompt_tokens"),
                            "completion": r["usage"].get("completion_tokens")})
    c = sorted(a["completion"] for a in att)
    return {"samples": 4, "calls": len(allc), "distinct_attempts": len(att), "completion": c,
            "mean": sum(c) / len(c), "median": (c[len(c) // 2 - 1] + c[len(c) // 2]) / 2,
            "at_cap": sum(x >= CAP for x in c), "prompt_first": sorted({a["prompt"] for a in att})[:4],
            "same_request_other_lengths": "repeats of one request under one seed drew different lengths "
                                          "(the seed does not fix a draw under batching)"}


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def rules(n: int) -> dict:
    """The first-tasks rule at n impossible tasks: bank at b passes or fewer, with b the largest count whose
    probability under a true 0.15 stays at or under 5% (the registered 0 or 1 of 30 gives 4.8%), except that b is
    0 when even 0 passes exceeds 5%; arms from 6 base passes (paired on n tasks, an exact sign test reaches
    two-sided 0.05 only at 6 discordant pairs, all one way: 0.031); base only between. The detectable rate: the
    true rate at which the bank is avoided 80% of the time."""
    b = 0
    while binom_cdf(b + 1, n, 0.15) <= 0.05:
        b += 1
    det = next(p / 1000 for p in range(1, 1000) if 1 - binom_cdf(b, n, p / 1000) >= 0.80)
    return {"n": n, "bank_at_or_under": b, "false_bank_at_0.15": binom_cdf(b, n, 0.15),
            "base_only": [b + 1, 5], "arms_from": 6, "arms_rate": 6 / n,
            "rate_clearing_the_bank_80pct": det, "one_pass_is": 1 / n}


def price(sm: dict) -> dict:
    """Base only, the join fix in place (a repeat at the 600 s timeout joins its generation: one generation an
    attempt). Tokens an attempt: low 9k, central the distinct attempts' mean, high every attempt at the cap.
    Aggregate throughput measured on this server: 174 tokens/s at 8 (launch 6), 233 at 12 (launch 7; at 12 the
    minimal scaffold's peak holds 12 x 27k = 324k tokens against a 224k KV cache, so the high case takes 8's rate:
    preemption cancels the gain). Original tasks end at their first pass: 1.5, 2.5 and 4 attempts. A tail of half
    a sample's serial time at the per-stream rate."""
    tok = {"low": 9000, "central": round(sm["mean"]), "high": CAP}
    orig = {"low": 1.5, "central": 2.5, "high": 4.0}
    imp_frac = {"low": 0.9, "central": 1.0, "high": 1.0}
    stream = {8: 21.75, 12: 19.4}
    designs = {"A: 30 impossible x 10 attempts + 10 original": (30, 10),
               "B: 15 impossible x 10 attempts + 10 original": (15, 10),
               "C: 30 impossible x 5 attempts + 10 original": (30, 5)}
    out = {}
    for name, (n, att) in designs.items():
        row = {}
        for conc in (8, 12):
            for case in ("low", "central", "high"):
                agg = 174.0 if conc == 8 or case == "high" else 233.0
                attempts = n * att * imp_frac[case] + 10 * min(orig[case], att)
                tokens = attempts * tok[case]
                tail = 0.5 * att * tok[case] / stream[conc]
                h = (tokens / agg + tail) / 3600
                row[f"c{conc}_{case}"] = {"attempts": attempts, "tokens_M": tokens / 1e6, "a100_h": round(h, 2),
                                          "usd": round(h * PRICE, 2)}
        out[name] = row | {"rule": rules(n)}
    return {"tokens_an_attempt": tok, "original_attempts": orig, "designs": out,
            "kv_cache_tokens": 224353, "peak_request_tokens_at_attempt_10": 27000,
            "peak_at_8": 8 * 27000, "peak_at_12": 12 * 27000}


def cap(rows: list[dict]) -> dict:
    def one(rs):
        c = [(r.get("usage") or {}).get("completion_tokens") or 0 for r in rs]
        return {"calls": len(c), "at_cap_16384": sum(x >= CAP for x in c),
                "share_calls": sum(x >= CAP for x in c) / len(c),
                "tokens": sum(c), "tokens_in_capped_calls": sum(x for x in c if x >= CAP),
                "share_tokens": sum(x for x in c if x >= CAP) / sum(c),
                "over_8192_calls": sum(x > CAP2 for x in c), "tokens_saved_at_8192": sum(max(0, x - CAP2) for x in c),
                "share_saved": sum(max(0, x - CAP2) for x in c) / sum(c)}
    return {"all_calls": one(rows), "tools_gate": one([r for r in rows if r["run"] == RUN]),
            "minimal_smoke": one([r for r in rows if r["run"].startswith("ib_smoke_minimal")])}


def main() -> None:
    rows = [json.loads(x) for x in (L7 / "proxy.jsonl").read_text().splitlines()]
    sm = smoke_minimal(rows)
    rec = {"rule": "the lead's 2026-11-23 1:50 PM block (zero GPU; counts and timings only)",
           "scratch": scratch(rows), "a18": accounting(rows), "rehearsal12": rehearsal12(), "smoke_minimal": sm,
           "price": price(sm), "cap": cap(rows)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"a18_table": rec["a18"]["table"], "sum": rec["a18"]["table_sum"],
                      "r12": rec["rehearsal12"]["total"], "one_to_one": rec["rehearsal12"]["one_to_one"]}, indent=1))


if __name__ == "__main__":
    main()
