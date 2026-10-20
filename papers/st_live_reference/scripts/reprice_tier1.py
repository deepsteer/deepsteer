#!/usr/bin/env python3
"""The Tier 1 re-price from the T1.0 relaunch's measured rates (G1 input; Amendments 7 and later).

    python3 papers/st_live_reference/scripts/reprice_tier1.py --screen-dir OUT \\
        [--write OUT/tier1_reprice.md]

Rates, per model, from its own records (vLLM engine, Amendment 7):
- generation (every trace cell except the steering gate): rollouts x mean tokens to the answer at
  the cap of record / the full-batch aggregate decoded tokens per second of the setting of record
  (the screen's, or GPT-OSS-20B's parity probe); continuous batching makes total tokens over
  throughput the right model;
- the steering gate (T1.3) needs residual-stream hooks during generation, which vLLM does not
  offer: HF generation at the largest batch the HF memory probe fits, its measured seconds per
  step times the expected batch-maximum length;
- readouts (HF teacher-forced): single-row where the model's unpadded replicate floor exceeds the
  KDG bar (0.05 nats): GPT-OSS-20B by default (ST-A3; §31's one-sequence-at-a-time precedent,
  0.131 s per row at ~740 tokens, scaled linearly in length), the second family by its own
  measured floor, at its own rate (measured single-row when its padded check failed and the probe
  read unbatched; else bounded by one batch forward per row, r x b); batched otherwise, at the
  measured batched rate. Every readout rate scales linearly in length.
Parts the T1.0 records cannot measure keep a "plan" price (a ST_PLAN line item) or an
"estimate" (a split of a §7 row the plan prices only as a whole). Every row prints the cumulative
A100-h with it included, starting from the A100-h already spent (``SPENT``, plus this session).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_lib as L  # noqa: E402

ENVELOPE = 35.0
BRAKE_HELD, BRAKE_SECOND = 65, 50
PANEL_ROWS = 586 * 8 * 4  # four E cells at dose 0, eight permutations
KDG_BAR = 0.05
S31_SINGLE_ROW_S, S31_TOKENS = 0.131, 740  # KDG_RESULTS §31: 625 rows in 81.8 s, GPT-OSS-20B, A100
# T1.1 pod (w4bn754dc3u8qs), GPT-OSS-20B on torch 2.13 with eager experts, single-row: the dose-0
# cell's 2,112 rows took 840 s between the bank index and dose0_h1 file mtimes (the judgment cell
# 132 rows in 53 s); the record holds no per-cell timing before 2026-10-19 (now `dose0_read_s`).
# At 0.40 s for ~370 prompt tokens vs 0.475 s for ~870 prompt + trace tokens, the eager forward is
# mostly fixed cost, so dose-0 rows are priced at their own rate, not length-scaled.
T11_DOSE0_S = {"gpt_oss_20b": 840.0 / 2112}
T15_SUBSET = 2  # author, 2026-10-19: post-reasoning forward on a seeded 2 of 8 per scenario
# A100-h already spent on the ST line before this session (author, 2026-10-19)
# (the relaunch's own row is an estimate: its pod logged no start time; from the rider pod on,
# --session-h defaults to the pod's session_start/session_end timestamps)
SPENT = {"T1.0 first pod (OLMo-3-Think, 7y7jn6pwphbzb5)": 0.8,
         "VALIDATE pods x4 (146trytky4d1z3, 40yy8fg8wtnucu, vgjndx7dmqyfa8, nw2ot7pye4ad1a)": 0.85,
         "T1.0 relaunch (esmxvsxj8wzkaq; estimate)": 1.8,
         "rider pod (t10b; timestamps)": 1.81,
         "rider pod 2 (t10c; timestamps)": 0.65,
         "rider pod 2b (gyyo2g10b95uqs; download lost; estimate)": 0.75,
         "rider pod 2c (t10d; timestamps)": 1.0}
PROVISION_H = 0.1  # pod creation to script start, and the download after the end timestamp
# HF-side measurements a later session (the rider pod) replaces in an earlier session's record
HF_KEYS = ("padded_check", "fast_path_rung", "hf_memory_probe", "read_seconds_per_row",
           "probe_mean_len", "load_seconds")
FIXED = {"T1.2 AtP": (0.5, "plan"), "T1.4 training (3 adapters)": (3.0, "plan"),
         "T1.5 capability + specificity + Heretic": (2.0, "estimate"),
         "T1.6 training (2 QLoRA adapters)": (1.5, "estimate"),
         "T1.6 direction riders": (0.2, "estimate")}


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def hf_step(hf: dict) -> float | None:
    """Seconds per decode step at the HF probe's largest fitting batch. Probes recorded before the
    warm-up fix (no ``warmed``) timed the short decode cold, so their slope is biased low; their
    warm second timing bounds the step from above (t_2n = prefill + 2n steps, prefill >= 0)."""
    big = hf.get("largest_fitting_batch")
    p = next((x for x in hf.get("probes") or [] if x.get("batch") == big and x.get("fits")), None)
    if p is None:
        return None
    if p.get("warmed"):
        return p["seconds_per_step"]
    return p["t_decode_2n"] / (2 * p["decode_steps"])


def rung_alone_rate(rung: dict) -> float | None:
    """Seconds per row of the rung's first pass on the kept fast path, each row read alone
    (batch 1): a measured single-row rate, first-call cost included (so biased high)."""
    kept = rung.get("kept", "eager")
    if kept == "eager":
        return None
    comp = kept.endswith("compile")
    a = next((x for x in rung.get("attempts") or [] if x.get("ok") and x.get("compiled") == comp),
             None)
    if a is None or not a.get("first_fast_pass_seconds") or not rung.get("n"):
        return None
    return a["first_fast_pass_seconds"] / rung["n"]


def _record_file(d: Path) -> Path:
    """A session's per-model record: the screen's, or T1.1's."""
    return d / "t11_record.json" if (d / "t11_record.json").exists() else d / "screen_record.json"


def model_rates(d: Path, overlays: tuple = ()) -> dict | None:
    """Measured rates for one model directory (None if it has no generation timing); HF-side
    keys from ``overlays`` (later sessions' records of the same model) replace its own."""
    rec = json.loads(_record_file(d).read_text())
    for o in overlays:
        rec.update({k: o[k] for k in HF_KEYS if o.get(k) is not None})
    pv = rec.get("phase_v") or {}
    timing = pv.get("screen_timing") or pv.get("parity_timing") or pv.get("t11_timing")
    if not timing:
        return None
    stage = ("screen" if pv.get("screen_timing") else "parity" if pv.get("parity_timing")
             else "t11")
    rel = {"screen": "screen/h1_screen.jsonl", "parity": "parity/parity_vllm.jsonl",
           "t11": "gen/h1_rollouts.jsonl"}[stage]
    rows = _rows(d / rel)
    lens = [r["tokens_to_answer"] for r in rows if r.get("status") not in (None, "not_run")]
    pad = rec.get("padded_check") or {}
    floor = pad.get("noise_floor_replicate_unpadded")
    hf = (rec.get("hf_memory_probe") or {})
    return {
        "model": d.name, "dir": d, "rec": rec, "stage": stage,
        "tps": timing["tokens_per_second"], "choice": timing.get("choice"),
        "mean_tokens": float(np.mean(np.minimum(lens, L.CAP_OF_RECORD))) if lens else 0.0,
        "lens": lens,
        # Amendment 4 item 1: the development model's screen rollouts count toward T1.1 (same
        # effort, cap, sampler, harness); parity rows use §40's prompts and do not
        "screen_rollouts": len(lens) if stage == "screen" else 0,
        "prompt_len": rec.get("probe_mean_len") or 350.0,
        "batched_read_s": rec.get("read_seconds_per_row"),
        "alone_read_s": rung_alone_rate(rec.get("fast_path_rung") or {}),
        "replicate_floor": floor,
        "single_row": d.name == "gpt_oss_20b" or (floor is not None and floor > KDG_BAR),
        "hf_step_s": hf_step(hf),
        "hf_batch": hf.get("largest_fitting_batch"),
        "load_h": (sum(x or 0 for x in timing.get("load_seconds", [])) / max(1, len(
            timing.get("load_seconds", [])))) / 3600 + (rec.get("load_seconds") or 0) / 3600,
        "elapsed_h": rec.get("elapsed_h") or 0.0,
    }


def gen_h(m: dict, n: int) -> float:
    return n * m["mean_tokens"] / max(m["tps"], 1e-9) / 3600


def read_rate(m: dict) -> tuple[float, str]:
    """(seconds per readout row at the probe's mean length, basis)."""
    r = m["batched_read_s"]
    if r is None:
        px = m.get("read_proxy")
        if px is None:
            raise SystemExit(f"{m['model']}: no measured read rate in its record and no proxy")
        rate, basis = read_rate(px)
        return rate, f"proxy: {px['model']}'s rate ({basis}); this model's HF phase did not run"
    if not m["single_row"]:
        return r, "measured (batched)"
    if m["rec"].get("single_row_read_s"):  # T1.1's own single-row forward, measured
        return m["rec"]["single_row_read_s"], "measured single-row (T1.1, prompt + trace)"
    if m["model"] == "gpt_oss_20b":
        return (S31_SINGLE_ROW_S * m["prompt_len"] / S31_TOKENS,
                "§31 single-row precedent, length-scaled")
    pad = m["rec"].get("padded_check") or {}
    if not pad.get("ok"):  # the probe read unbatched (sc.unbatched): r is the single-row rate
        return r, "measured single-row"
    if m.get("alone_read_s"):
        return m["alone_read_s"], ("measured single-row (the rung's alone pass on the kept "
                                   "path, first-call cost included)")
    return r * min(pad.get("batch") or 1, 16), "single-row bound (one batch forward per row)"


def read_h(m: dict, n: int, gen_tokens: float = 0.0) -> float:
    """n readouts of prompt (+ gen_tokens of trace) at ``read_rate``, scaled by length."""
    return n * read_rate(m)[0] * (m["prompt_len"] + gen_tokens) / m["prompt_len"] / 3600


def hf_gen_h(m: dict, n: int) -> float | None:
    """HF generation (hooks needed), at the HF probe's largest fitting batch."""
    if not m["hf_step_s"] or not m["hf_batch"]:
        return None
    b = m["hf_batch"]
    steps = L.expected_batch_max(m["lens"], b, L.CAP_OF_RECORD)
    return math.ceil(n / b) * steps * m["hf_step_s"] / 3600


def gate_price(gm: dict) -> tuple[float, str]:
    """T1.3 (960 rollouts, HF generate with hooks) from the HF probe, unless the probe ran on
    uncompiled flex attention: that is a reference-speed kernel no arm would run (the relaunch's
    GPT-OSS rung kept it under the candidate-floor bar, since replaced), so T1.3 keeps its plan
    price (author, 2026-10-19)."""
    rung = gm["rec"].get("fast_path_rung") or {}
    h = hf_gen_h(gm, 20 * 5 * 8 + 5 * 4 * 8)
    if h is None:
        return 1.5, "plan (no HF memory probe for this model)"
    path = ((gm["rec"].get("hf_memory_probe") or {}).get("path")) or ""
    if not path.startswith("eager") and rung.get("kept") == "fast" and rung.get(
            "fast_impl") == "flex_attention":
        return 1.5, "plan (probe ran on uncompiled flex attention, not a path of record)"
    probes = (gm["rec"].get("hf_memory_probe") or {}).get("probes") or [{}]
    return h, ("measured" if probes[0].get("warmed")
               else "measured bound (cold-probe fix: warm 2n timing / 2n)")


RR_PROMPTS, RR_SELECT_PROMPTS, RR_ANSWER, RR_ARM_CAP = 360, 180, 64, 0.5


def rr_h(m: dict, prompts: int, rollouts: int, arms: int) -> tuple[float, str]:
    """Amendment 10 (author, 2026-10-19): refusal-retention generation at the model's measured
    vLLM rate, the answer budget on top of the mean tokens to the answer, one engine load per
    arm; an arm over 0.5 A100-h drops to 2 rollouts. Upper bound: every twin kept."""
    def arm(r: int) -> float:
        return prompts * r * (m["mean_tokens"] + RR_ANSWER) / max(m["tps"], 1e-9) / 3600 \
            + m["load_h"]
    r = rollouts if arm(rollouts) <= RR_ARM_CAP else 2
    return arms * arm(r), f"measured rate; {r} rollouts x {arms} arm(s); upper bound (60 kept)"


def dose0_rate(m: dict) -> tuple[float, str] | None:
    """Measured single-row dose-0 rate from a T1.1 record (or the T1.1 pod's timestamps)."""
    if m["rec"].get("dose0_read_s"):
        return m["rec"]["dose0_read_s"], "measured single-row dose-0 (T1.1)"
    if m["stage"] == "t11" and m["model"] in T11_DOSE0_S:
        return T11_DOSE0_S[m["model"]], "measured single-row dose-0 (T1.1 pod file timestamps)"
    return None


def rr_uncalibrated(m: dict) -> bool:
    """Amendment 10's pre-registered branch: a selection read as uncalibrated selects nothing."""
    d = m.get("dir")
    if d is None:
        return False
    f = d.parent / "_rr" / d.name / "rr" / "rr_selection_v3.json"
    return f.exists() and json.loads(f.read_text()).get("status", "").startswith("uncalibrated")


def price(m: dict) -> dict:
    """Both roles; the steering gate from the model's own HF probe (else its plan price)."""
    gate = gate_price(m)
    t = m["mean_tokens"]
    rb = read_rate(m)[1]
    dev = {
        f"T1.1 generation 264 x 8 less the screen's {m['screen_rollouts']} (vLLM)":
            (gen_h(m, 264 * 8 - m["screen_rollouts"]), "measured"),
        "T1.1 readout pass (post-reasoning 2,112 + dose-0 2,112 + judgment 132)":
            (read_h(m, 264 * 8, t) + read_h(m, 264 * 8) + read_h(m, 132), rb),
        "T1.1 loads (vLLM + HF)": (m["load_h"], "measured"),
        "T1.1 refusal-retention selection (Amendment 10: 60 x {plain, care, operator} x 4)":
            rr_h(m, RR_SELECT_PROMPTS, 4, 1),
        "T1.2 AtP": FIXED["T1.2 AtP"],
        "T1.3 steering gate 960 rollouts (HF generate with hooks)": gate,
        "T1.4 training (3 adapters)": FIXED["T1.4 training (3 adapters)"],
        "T1.5 generation 132 x 8 x 3 + none re-run (vLLM)":
            (gen_h(m, 132 * 8 * 3 + BRAKE_HELD * 8), "measured"),
        "T1.5 readouts (post-reasoning x 3 + panel dose-0 x 3)":
            (read_h(m, 132 * 8 * 3, t) + read_h(m, PANEL_ROWS * 3), rb),
        "T1.5 capability + specificity + Heretic": FIXED["T1.5 capability + specificity + Heretic"],
        "T1.5 refusal-retention (Amendment 10: none^re + 3 adapters x 360 x 4)":
            rr_h(m, RR_PROMPTS, 4, 4),
    }
    second = {
        "T1.6 generation (60 x 2 top-up, 100 x 6, 100 x 6 x 2, none re-run) (vLLM)":
            (gen_h(m, 60 * 2 + 100 * 6 + 100 * 6 * 2 + BRAKE_SECOND * 6), "measured"),
        "T1.6 readouts (post-reasoning, panel dose-0 x 2)":
            (read_h(m, 160 * 6 + 100 * 6 * 2, t) + read_h(m, PANEL_ROWS * 2),
             rb),
        "T1.6 loads": (2 * m["load_h"], "measured"),
        "T1.6 training (2 QLoRA adapters)": FIXED["T1.6 training (2 QLoRA adapters)"],
        "T1.6 direction riders": FIXED["T1.6 direction riders"],
        "T1.6 refusal-retention (Amendment 10: none^re + best + L-beh x 360 x 2)":
            rr_h(m, RR_PROMPTS, 2, 3),
    }
    second["T1.6 refusal-retention selection (Amendment 10 item 3 on the unadapted model; "
           "60 x {plain, care, operator} x 4)"] = rr_h(m, RR_SELECT_PROMPTS, 4, 1)
    variants: dict = {}
    if m["stage"] == "t11":  # T1.1 has run: its rows are spent; T1.5's none re-run was v2's
        for k in [k for k in dev if k.startswith("T1.1")]:
            dev[k] = (0.0, "done (T1.1 pod; in the spend)")
        k5 = next(k for k in dev if k.startswith("T1.5 generation"))
        dev[k5] = (gen_h(m, 132 * 8 * 3), "measured (none re-run done in T1.1 v2)")
        d0 = dose0_rate(m)
        if d0 is not None:  # T1.1's rate is per rollout including the trace: no length rescale
            r_post = m["rec"]["single_row_read_s"]
            panel = PANEL_ROWS * 3 * d0[0] / 3600
            kr = next(k for k in dev if k.startswith("T1.5 readouts"))
            dev[kr] = (132 * 8 * 3 * r_post / 3600 + panel,
                       f"post-reasoning {r_post:.3f} s/rollout + panel dose-0 {d0[0]:.3f} s/row "
                       f"({d0[1]})")
            sub = 132 * T15_SUBSET * 3 * r_post / 3600
            full = 132 * (8 - T15_SUBSET) * r_post / 3600  # one adapter falling back
            variants["T1.5 readouts, variant: post-reasoning forward on a seeded "
                     f"{T15_SUBSET} of 8 per scenario per adapter (banking + the §31 >= 0.99 "
                     "top-letter check), panel dose-0 full single-row"] = (
                sub + panel, dev[kr][0],
                f"each adapter failing the check adds {full:.2f} (its letters from full forwards)")
        if rr_uncalibrated(m):
            kq = next(k for k in dev if k.startswith("T1.5 refusal-retention"))
            dev[kq] = (0.0, "uncalibrated on this model (Amendment 10 pre-registration): no arm")
    return {"development": dev, "second_family": second, "variants": variants,
            "development_total": sum(v[0] for v in dev.values()),
            "second_family_total": sum(v[0] for v in second.values())}


def session_hours(d: Path) -> float | None:
    """Wall hours between the pod's session_start.json and session_end.json, plus provisioning."""
    from datetime import datetime

    try:
        s = json.loads((d / "session_start.json").read_text())["session_start_utc"]
        e = json.loads((d / "session_end.json").read_text())["session_end_utc"]
    except FileNotFoundError:
        return None
    f = "%Y-%m-%dT%H:%M:%SZ"
    return (datetime.strptime(e, f) - datetime.strptime(s, f)).total_seconds() / 3600 + PROVISION_H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-dir", type=Path, nargs="+", required=True,
                    help="session output dirs, oldest first: a model's generation rates come from "
                         "the latest dir with its generation timing, HF-side measurements from "
                         "any later dir (the rider pod's GPT-OSS rung and probe)")
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--session-h", type=float, default=None,
                    help="A100-h of the latest session (default: its session_start/end "
                         "timestamps + 0.1 provisioning, else the models' elapsed + 0.25)")
    a = ap.parse_args()
    found: dict[str, tuple] = {}
    for sd in a.screen_dir:
        for d in sorted(p for p in sd.iterdir() if _record_file(p).exists()):
            r = json.loads(_record_file(d).read_text())
            pv = r.get("phase_v") or {}
            if pv.get("screen_timing") or pv.get("parity_timing") or pv.get("t11_timing"):
                found[d.name] = (d, [])
            elif d.name in found:
                found[d.name][1].append(r)
    models = {k: m for k, (d, ov) in found.items() if (m := model_rates(d, tuple(ov)))}
    if not models:
        raise SystemExit(f"no records with generation timing under {a.screen_dir}")
    session_h, how = a.session_h, "given"
    if session_h is None:
        session_h, how = session_hours(a.screen_dir[-1]), "pod timestamps + 0.1 provisioning"
    if session_h is None:
        session_h, how = (sum(m["elapsed_h"] for m in models.values()) + 0.25,
                          "models' elapsed + 0.25 setup")
    spent = sum(SPENT.values()) + session_h
    # a model whose HF phase did not run (the relaunch's Qwen3.8 revision-gate stop) borrows the
    # readout rate of a measured dense model of its size class, labelled as a proxy
    dense = [m for k, m in models.items() if k != "gpt_oss_20b" and m["batched_read_s"]]
    for m in models.values():
        if m["batched_read_s"] is None and dense:
            m["read_proxy"] = dense[0]
    prices = {k: price(m) for k, m in models.items()}
    lines = ["# Tier 1 re-priced from the T1.0 relaunch (G1 input)", "",
             "Spent so far: " + "; ".join(f"{k} {v:.2f}" for k, v in SPENT.items())
             + f"; this session {session_h:.2f} ({how})"
             + f" → **{spent:.2f} A100-h**.", "",
             "Readouts are single-row where the model's unpadded replicate floor exceeds 0.05 "
             "nats (GPT-OSS-20B by ST-A3 and §31's precedent); rows marked plan keep ST_PLAN "
             "prices, estimate splits a §7 row priced only as a whole. The pair rider (≤ 0.3) is "
             "not included.", ""]
    for k, m in models.items():
        p = prices[k]
        lines += [f"## {k}", "",
                  f"stage {m['stage']}; setting of record `{m['choice']}`; full-batch "
                  f"{m['tps']:.0f} tok/s; mean tokens to answer {m['mean_tokens']:.0f}; "
                  f"replicate floor {m['replicate_floor']}; readouts "
                  f"{read_rate(m)[1]}, {read_rate(m)[0]:.3f} s/row at "
                  f"{m['prompt_len']:.0f} tokens", ""]
        for role in ("development", "second_family"):
            lines += [f"**{role}**", "",
                      "| row | A100-h | cumulative (spent + rows so far) | basis |",
                      "|---|---|---|---|"]
            cum = spent
            for row, (h, basis) in p[role].items():
                cum += h
                lines.append(f"| {row} | {h:.2f} | {cum:.2f} | {basis} |")
            lines += [f"| **total** | **{p[role + '_total']:.2f}** | **{cum:.2f}** | |", ""]
            if role == "development" and p.get("variants"):
                lines += ["Priced variants (not in the total):", "",
                          "| variant | A100-h | row of record | delta | note |",
                          "|---|---|---|---|---|"]
                for row, (h, ref, note) in p["variants"].items():
                    lines.append(f"| {row} | {h:.2f} | {ref:.2f} | {h - ref:+.2f} | {note} |")
                lines.append("")
    lines += ["## Envelope by role assignment (spent + development + second family)", "",
              "| development | second family | total A100-h | within 35 |", "|---|---|---|---|"]
    for dev in models:
        for sec in models:
            if sec != dev:
                tot = spent + prices[dev]["development_total"] + prices[sec]["second_family_total"]
                ok = "yes" if tot <= ENVELOPE else "NO"
                lines.append(f"| {dev} | {sec} | {tot:.2f} | {ok} |")
    if len(models) < 2:
        lines += ["", f"Only {', '.join(models)} has generation timing under these dirs, so no "
                  "role pair is priced and the second-family block above uses this model's own "
                  "rates; re-run locally with every banked session dir (oldest first) for the "
                  "envelope."]
    text = "\n".join(lines) + "\n"
    print(text)
    out = a.write or (a.screen_dir[-1] / "tier1_reprice.md")
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(
        {"spent": {**SPENT, "this_session": session_h, "total": spent},
         "prices": {k: {r: {n: list(v) for n, v in p[r].items()} for r in
                        ("development", "second_family", "variants")} for k, p in prices.items()}},
        indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
