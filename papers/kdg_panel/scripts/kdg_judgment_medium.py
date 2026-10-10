"""GPT-OSS-20B judgment readout at the default reasoning level (KDG_GPTOSS_SPEC G-A23 item 3; p2k).

The panel's letter-only judgment prompt (``jl_chat_neutral``) at ``Reasoning: medium``, permutation
0, one trace per scenario on the 175 divergent scenarios, independent of the action trace (its own
prompt and seeds). Order: VALIDATE re-check under identical batching (bail on fail, nothing
generated), the first batch of 88 as the timing step (stop and bank if it projects past the
envelope), the second batch under a deadline, rows written, then the forward readout one sequence
at a time (G-A20) with letter-step distributions and post-reasoning residuals. Record checkpointed
after every step.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import kdg_dose_stated as kds
import numpy as np

JM_BATCH, JM_ENVELOPE_H, JM_SEED_OFFSET, JM_ORDER_OFFSET = 88, 1.7, 70_000, 4
P2K_BATCH_SECONDS = 2643.0  # p2k's measured first batch (G-A24 item 2: the continuation's guard)
CELL = "jl_chat_neutral"


def run_judgment_medium(model, cfg, scen: dict, ids: list[str], pilot_rows: list[dict], out: Path,
                        t_start: float, dry: bool, envelope_h: float = JM_ENVELOPE_H,
                        done_ids: set[str] | None = None) -> dict:
    """``done_ids`` (G-A24 item 2, p2l): the scenarios p2k already ran. They must be exactly the
    first batch of p2k's permutation; the rest run as one batch with p2k's second-batch seed."""
    out.mkdir(parents=True, exist_ok=True)
    lid = {x: model.token_id(x) for x in "ABCDE"}
    letter_ids = set(lid.values())
    model.reasoning_level = cfg.reasoning_level
    cap = 64 if dry else kds.MAIN_CAP
    batch = 8 if dry else JM_BATCH  # the dry run still exercises the second-batch path
    partial = out / "jm_partial.jsonl"
    partial.unlink(missing_ok=True)
    rec: dict = {"spec": "KDG_GPTOSS_SPEC G-A23 item 3", "cell": CELL, "cap": cap,
                 "batch": batch, "envelope_h": envelope_h, "n_scenarios": len(ids),
                 "readout_batch": kds.READOUT_BATCH}

    def elapsed_h():
        return kds.SETUP_H + (time.time() - t_start) / 3600

    def checkpoint():
        (out / "jm_record.json").write_text(json.dumps(rec, indent=1, default=float))

    def write_rows(rows):
        with open(out / "jm.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def finish(outcome, rows, pids):
        write_rows(rows)  # rows before the readout: a readout crash keeps them
        rec["outcome_before_readout"] = outcome
        checkpoint()
        kds.free_cache(model)
        ok, logp, resid, fsecs = kds._readout_batched(model, rows, pids, lid)
        if len(ok) != sum(r["token_identity"] for r in rows):
            raise RuntimeError("an identity row got no readout row")
        rec["forward_seconds"] = fsecs
        rec["readout_rows"] = len(ok)
        rec["outcome"] = outcome
        rec["elapsed_h_at_end"] = elapsed_h()
        write_rows(rows)
        np.savez_compressed(out / "jm.npz", logp_letter_step=logp.astype(np.float16),
                            resid_post_reasoning=resid, readout_rows=np.array(ok))
        checkpoint()
        return rec

    # (a) VALIDATE re-check under identical batching (gates everything)
    seqs, pos = kds.rebuild_validate_seqs(model, cfg, pilot_rows, scen)
    rec["validate"] = kds.validate_matched(model, seqs, pos, list(lid.values()), dry)
    checkpoint()
    print(f">> VALIDATE re-check: batch16 {rec['validate']['matched_batch16_max']:.4f}, alone "
          f"{rec['validate']['matched_alone_max']:.4f} -> pass {rec['validate']['pass']}",
          flush=True)
    if not rec["validate"]["pass"]:
        rec["outcome"] = "bail_validate"
        checkpoint()
        return rec
    order = np.random.default_rng(kds.ORDER_SEED + JM_ORDER_OFFSET).permutation(len(ids))
    jobs = [(scen[ids[i]], CELL, 0) for i in order]
    if done_ids is not None:  # p2l: p2k's second batch, as p2k would have run it
        head = {s.id for s, _, _ in jobs[:batch]}
        if head != set(done_ids):
            raise RuntimeError(f"p2k's rows are not its first batch ({len(head ^ set(done_ids))} "
                               "ids differ); refusing to continue on a different split")
        rest = jobs[batch:]
        rec["continuation_of"] = "p2k"
        rec["n_done_before"] = len(done_ids)
        deadline = t_start + (envelope_h - kds.SETUP_H) * 3600 - kds.FWD_S_PER_ROW * len(rest)
        rows, pids, secs = kds._generate(model, cfg, rest, cap, kds.SEED + JM_SEED_OFFSET + 1,
                                         letter_ids, partial, "jm", deadline,
                                         est_batch_seconds=0.0 if dry else P2K_BATCH_SECONDS,
                                         batch_size=len(rest))
        rec["generation"] = {"n": len(rows), "run": sum(r["status"] != "not_run" for r in rows),
                             "identity": sum(r["token_identity"] for r in rows),
                             "truncated": sum(r["status"] == "truncated" for r in rows),
                             "gen_seconds": float(sum(secs))}
        checkpoint()
        return finish("jm_complete" if rec["generation"]["run"] == len(rest) else "stop_report",
                      rows, pids)
    # (b) timing step: the first batch, banked
    first, rest = jobs[:batch], jobs[batch:]
    rows, pids, secs = kds._generate(model, cfg, first, cap, kds.SEED + JM_SEED_OFFSET,
                                     letter_ids, partial, "jm", batch_size=batch)
    n_rest = math.ceil(len(rest) / batch)
    proj = elapsed_h() + (secs[0] * n_rest + kds.FWD_S_PER_ROW * len(jobs)) / 3600
    rec["timing"] = {"first_batch_seconds": secs[0], "remaining_batches": n_rest,
                     "projected_hours_at_end": proj, "elapsed_h": elapsed_h()}
    checkpoint()
    print(f">> timing: first batch {secs[0]:.0f}s, projected {proj:.2f} A100-h "
          f"(envelope {envelope_h})", flush=True)
    if proj > envelope_h:
        return finish("stop_report", rows, pids)
    # (c) the rest under a deadline at the envelope
    if rest:
        deadline = t_start + (envelope_h - kds.SETUP_H) * 3600 - kds.FWD_S_PER_ROW * len(jobs)
        r2, p2, s2 = kds._generate(model, cfg, rest, cap, kds.SEED + JM_SEED_OFFSET + 1,
                                   letter_ids, partial, "jm", deadline, est_batch_seconds=secs[0],
                                   batch_size=batch)
        rows += r2
        pids += p2
        secs = [*secs, *s2]
    rec["generation"] = {"n": len(rows), "run": sum(r["status"] != "not_run" for r in rows),
                         "identity": sum(r["token_identity"] for r in rows),
                         "truncated": sum(r["status"] == "truncated" for r in rows),
                         "gen_seconds": float(sum(secs))}
    checkpoint()
    return finish("jm_complete", rows, pids)
