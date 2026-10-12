"""GPT-OSS-20B reasoning-dose curve on the H1 set (KDG_GPTOSS_SPEC G-A28 item 3; pod p2o, after the
judgment readout).

On the stratified sample of 64 (``data/harm_divergence/h1_dose_sample.json``), the action prompt
(``dl_chat_neutral``, permutation 0, sampler v2) at three reasoning levels: low (cap 1,536), a
fresh medium trace (cap 1,536, seed SEED + 90,000), and high, first as a pilot of 16 at cap
8,192, then the other 48 only if the pilot completes within the cap on >= 90% and its projected
time fits the envelope. Each arm is banked per batch; rows are written after every arm; no
forward readout (the post-reasoning letter is one-hot, §31), the action is the sampled letter.
The caller has already passed the VALIDATE re-check in this process (the judgment readout runs
it first).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import kdg_dose_stated as kds

CAP_LOW_MED, CAP_HIGH = 1536, 8192
SEEDS = {"low": 91_000, "medium": 90_000, "high_pilot": 92_000, "high_rest": 92_001}
PILOT_COMPLETION, REST_TIME_FACTOR = 0.90, 1.5  # rest batch of 48 projected at 1.5 x the pilot's


def run_dose_curve(model, cfg, scen: dict, sample: list[str], pilot: list[str], out: Path,
                   t_start: float, dry: bool, envelope_h: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    lid = {x: model.token_id(x) for x in "ABCDE"}
    letter_ids = set(lid.values())
    partial = out / "dc_partial.jsonl"
    partial.unlink(missing_ok=True)
    rest = [i for i in sample if i not in set(pilot)]
    rec: dict = {"spec": "KDG_GPTOSS_SPEC G-A28 item 3", "n": len(sample), "pilot": len(pilot),
                 "seeds": SEEDS, "envelope_h": envelope_h, "arms": {}}
    rows: list[dict] = []

    def elapsed_h() -> float:
        return kds.SETUP_H + (time.time() - t_start) / 3600

    def deadline() -> float:
        return t_start + (envelope_h - kds.SETUP_H) * 3600

    def save() -> None:
        with open(out / "dc.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (out / "dc_record.json").write_text(json.dumps(rec, indent=1, default=float))

    def arm(name: str, level: str, ids: list[str], cap: int, est: float | None) -> list[dict]:
        model.reasoning_level = level
        jobs = [(scen[i], "dl_chat_neutral", 0) for i in ids]
        got, _, secs = kds._generate(model, cfg, jobs, 64 if dry else cap,
                                     kds.SEED + SEEDS[name], letter_ids, partial, name,
                                     deadline(), est_batch_seconds=0.0 if dry else est,
                                     batch_size=len(jobs))
        for r in got:
            r["arm"] = name
            r["reasoning_level"] = level
        run = [r for r in got if r["status"] != "not_run"]
        rec["arms"][name] = {"level": level, "n": len(jobs), "run": len(run),
                             "identity": sum(r["token_identity"] for r in run),
                             "truncated": sum(r["status"] == "truncated" for r in run),
                             "seconds": float(sum(secs)), "elapsed_h": elapsed_h()}
        rows.extend(got)
        save()
        print(f">> dose curve {name}: {len(run)}/{len(jobs)} run, "
              f"{rec['arms'][name]['identity']} identity, {sum(secs):.0f}s", flush=True)
        return run

    try:
        arm("low", "low", sample, CAP_LOW_MED, 900.0)
        arm("medium", "medium", sample, CAP_LOW_MED, 2200.0)
        pr = arm("high_pilot", "high", pilot, CAP_HIGH, 3600.0)
        done = sum(r["status"] != "truncated" for r in pr) / max(1, len(pilot))
        est_rest = REST_TIME_FACTOR * rec["arms"]["high_pilot"]["seconds"]
        fits = elapsed_h() + est_rest / 3600 <= envelope_h
        ok = len(pr) == len(pilot) and done >= PILOT_COMPLETION and fits
        rec["high_gate"] = {"completion_within_cap": done, "projected_rest_seconds": est_rest,
                            "fits_envelope": fits, "pass": ok or dry}
        save()
        print(f">> high pilot gate: completion {done:.2f}, projected rest {est_rest:.0f}s, "
              f"fits {fits} -> {'run the other 48' if ok or dry else 'stop: back to the author'}",
              flush=True)
        if ok or dry:
            arm("high_rest", "high", rest, CAP_HIGH, est_rest)
        rec["outcome"] = ("dc_complete" if ok or dry else "high_gate_stop")
    finally:
        model.reasoning_level = cfg.reasoning_level
        rec.setdefault("outcome", "error")
        save()
    return rec
