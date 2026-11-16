#!/usr/bin/env python3
"""Rate windows for a running session (CST Amendment 52 item 4), provider-neutral.

    rate_watch.py --log SESSION_LOG --stage-file FILE --forecast JSON --out JSONL [--every 300]

Every ``--every`` seconds: the last vLLM progress fragment in the session log (``Processed prompts:
... i/n [mm:ss<..., est. speed input: ... toks/s, output: X toks/s]``; tqdm's speed is the mean
since the bar began, so tokens so far are about X times the elapsed seconds), and the generated
tokens per second over the window since the previous sample of the same bar. A window counts only
when one bar ran through it (the same total, its elapsed time advanced by at least 0.8 of the
window); otherwise the line says no window (a load, a parity read, a stage boundary). The stage is
read from ``--stage-file`` (the runner writes it); a counted window under 0.70 of that stage's
forecast is flagged ``RATE FLAG``. One JSON line per sample to ``--out`` and a ``>>`` line to
stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

BAR = re.compile(r"Processed prompts:.*?(\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<[^\]]*?"
                 r"output: ([\d.]+) toks/s")
FLAG = 0.70


def last_bar(text: str) -> dict | None:
    """The last progress fragment's (done, total, elapsed seconds, output tokens per second)."""
    frag = None
    for piece in re.split(r"[\r\n]", text):
        m = BAR.search(piece)
        if m:
            frag = m
    if frag is None:
        return None
    done, total, h, mm, ss, out = frag.groups()
    el = int(h or 0) * 3600 + int(mm) * 60 + int(ss)
    return {"done": int(done), "total": int(total), "elapsed_s": el, "out_tps": float(out),
            "tokens": float(out) * el}


def window(prev: dict | None, cur: dict | None, every: float) -> float | None:
    """Generated tokens per second over the window, or None when one bar did not span it."""
    if not prev or not cur or prev["total"] != cur["total"]:
        return None
    de = cur["elapsed_s"] - prev["elapsed_s"]
    if de < 0.8 * every:
        return None
    return (cur["tokens"] - prev["tokens"]) / de


def sample(log: Path, stage_file: Path, forecast: dict, prev: dict | None, every: float):
    text = log.read_text(errors="replace")[-400_000:] if log.exists() else ""
    cur = last_bar(text)
    stage = stage_file.read_text().strip() if stage_file.exists() else ""
    tps = window(prev, cur, every)
    fc = forecast.get(stage)
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
           "window_tps": tps, "forecast_tps": fc,
           "ratio": (tps / fc) if tps is not None and fc else None}
    rec["flag"] = bool(rec["ratio"] is not None and rec["ratio"] < FLAG)
    return rec, cur


def line(rec: dict) -> str:
    """The log line for one sample; a stage the forecast map does not name prints its rate with no
    ratio (GPC-1, 2026-11-15: a training stage's window crashed the formatter on a None ratio)."""
    if rec["window_tps"] is None:
        return f">> rate window ({rec['stage'] or 'no stage'}): no window"
    tag = "RATE FLAG" if rec["flag"] else "rate window"
    ratio = f"{rec['ratio']:.2f}" if rec["ratio"] is not None else "no forecast"
    return (f">> {tag} ({rec['stage']}): {rec['window_tps']:.0f} tokens/s against "
            f"{rec['forecast_tps']} forecast ({ratio})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--stage-file", type=Path, required=True)
    ap.add_argument("--forecast", required=True, help="JSON: stage -> forecast tokens/s")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=float, default=300.0)
    a = ap.parse_args()
    forecast, prev = json.loads(a.forecast), None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        time.sleep(a.every)
        rec, prev = sample(a.log, a.stage_file, forecast, prev, a.every)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(line(rec), flush=True)


if __name__ == "__main__":
    main()
