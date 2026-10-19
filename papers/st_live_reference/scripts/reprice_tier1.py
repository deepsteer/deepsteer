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
# A100-h already spent on the ST line before this session (author, 2026-10-19)
SPENT = {"T1.0 first pod (OLMo-3-Think, 7y7jn6pwphbzb5)": 0.8,
         "VALIDATE pods x4 (146trytky4d1z3, 40yy8fg8wtnucu, vgjndx7dmqyfa8, nw2ot7pye4ad1a)": 0.85}
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


def model_rates(d: Path) -> dict | None:
    """Measured rates for one model directory of the relaunch (None if nothing usable)."""
    rec = json.loads((d / "screen_record.json").read_text())
    pv = rec.get("phase_v") or {}
    timing = pv.get("screen_timing") or pv.get("parity_timing")
    if not timing:
        return None
    stage = "screen" if pv.get("screen_timing") else "parity"
    rel = "screen/h1_screen.jsonl" if stage == "screen" else "parity/parity_vllm.jsonl"
    rows = _rows(d / rel)
    lens = [r["tokens_to_answer"] for r in rows if r.get("status") not in (None, "not_run")]
    pad = rec.get("padded_check") or {}
    floor = pad.get("noise_floor_replicate_unpadded")
    hf = (rec.get("hf_memory_probe") or {})
    return {
        "model": d.name, "rec": rec, "stage": stage,
        "tps": timing["tokens_per_second"], "choice": timing.get("choice"),
        "mean_tokens": float(np.mean(np.minimum(lens, L.CAP_OF_RECORD))) if lens else 0.0,
        "lens": lens,
        # Amendment 4 item 1: the development model's screen rollouts count toward T1.1 (same
        # effort, cap, sampler, harness); parity rows use §40's prompts and do not
        "screen_rollouts": len(lens) if stage == "screen" else 0,
        "prompt_len": rec.get("probe_mean_len") or 350.0,
        "batched_read_s": rec.get("read_seconds_per_row"),
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
        raise SystemExit(f"{m['model']}: no measured read rate in its record")
    if not m["single_row"]:
        return r, "measured (batched)"
    if m["model"] == "gpt_oss_20b":
        return (S31_SINGLE_ROW_S * m["prompt_len"] / S31_TOKENS,
                "§31 single-row precedent, length-scaled")
    pad = m["rec"].get("padded_check") or {}
    if not pad.get("ok"):  # the probe read unbatched (sc.unbatched): r is the single-row rate
        return r, "measured single-row"
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
        return 1.5, "plan"
    if rung.get("kept") == "fast" and rung.get("fast_impl") == "flex_attention":
        return 1.5, "plan (probe ran on uncompiled flex attention, not a path of record)"
    probes = (gm["rec"].get("hf_memory_probe") or {}).get("probes") or [{}]
    return h, ("measured" if probes[0].get("warmed")
               else "measured bound (cold-probe fix: warm 2n timing / 2n)")


def price(m: dict, gate_m: dict | None = None) -> dict:
    """Both roles. ``gate_m``: the model whose HF probe prices the steering gate (GPT-OSS)."""
    gate = gate_price(gate_m or m)
    t = m["mean_tokens"]
    rb = read_rate(m)[1]
    dev = {
        f"T1.1 generation 264 x 8 less the screen's {m['screen_rollouts']} (vLLM)":
            (gen_h(m, 264 * 8 - m["screen_rollouts"]), "measured"),
        "T1.1 readout pass (post-reasoning 2,112 + dose-0 2,112 + judgment 132)":
            (read_h(m, 264 * 8, t) + read_h(m, 264 * 8) + read_h(m, 132), rb),
        "T1.1 loads (vLLM + HF)": (m["load_h"], "measured"),
        "T1.2 AtP": FIXED["T1.2 AtP"],
        "T1.3 steering gate 960 rollouts (HF generate with hooks)": gate,
        "T1.4 training (3 adapters)": FIXED["T1.4 training (3 adapters)"],
        "T1.5 generation 132 x 8 x 3 + none re-run (vLLM)":
            (gen_h(m, 132 * 8 * 3 + BRAKE_HELD * 8), "measured"),
        "T1.5 readouts (post-reasoning x 3 + panel dose-0 x 3)":
            (read_h(m, 132 * 8 * 3, t) + read_h(m, PANEL_ROWS * 3), rb),
        "T1.5 capability + specificity + Heretic": FIXED["T1.5 capability + specificity + Heretic"],
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
    }
    return {"development": dev, "second_family": second,
            "development_total": sum(v[0] for v in dev.values()),
            "second_family_total": sum(v[0] for v in second.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-dir", type=Path, required=True)
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--session-h", type=float, default=None,
                    help="A100-h of this session incl. setup (default: the models' elapsed + 0.25)")
    a = ap.parse_args()
    models = {}
    for d in sorted(p for p in a.screen_dir.iterdir() if (p / "screen_record.json").exists()):
        m = model_rates(d)
        if m is not None:
            models[d.name] = m
    if not models:
        raise SystemExit(f"no relaunch records under {a.screen_dir}")
    session_h = a.session_h if a.session_h is not None else (
        sum(m["elapsed_h"] for m in models.values()) + 0.25)
    spent = sum(SPENT.values()) + session_h
    gate_m = models.get("gpt_oss_20b")
    prices = {k: price(m, gate_m) for k, m in models.items()}
    lines = ["# Tier 1 re-priced from the T1.0 relaunch (G1 input)", "",
             "Spent so far: " + "; ".join(f"{k} {v:.2f}" for k, v in SPENT.items())
             + f"; this session {session_h:.2f} (models' elapsed + 0.25 setup unless given)"
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
    lines += ["## Envelope by role assignment (spent + development + second family)", "",
              "| development | second family | total A100-h | within 35 |", "|---|---|---|---|"]
    for dev in models:
        for sec in models:
            if sec != dev:
                tot = spent + prices[dev]["development_total"] + prices[sec]["second_family_total"]
                ok = "yes" if tot <= ENVELOPE else "NO"
                lines.append(f"| {dev} | {sec} | {tot:.2f} | {ok} |")
    text = "\n".join(lines) + "\n"
    print(text)
    out = a.write or (a.screen_dir / "tier1_reprice.md")
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(
        {"spent": {**SPENT, "this_session": session_h, "total": spent},
         "prices": {k: {r: {n: list(v) for n, v in p[r].items()} for r in
                        ("development", "second_family")} for k, p in prices.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
