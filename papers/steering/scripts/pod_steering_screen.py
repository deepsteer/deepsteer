#!/usr/bin/env python3
"""T1.0 screen, one model per process (STEERING_PLAN v0.2.5 §2, Amendments 2-4; T0.11).

    python3 papers/steering/scripts/pod_steering_screen.py --model olmo3_think --out OUT [--dry-run]

Order, back to back on one load (Amendment 3 item 3):
  1. GATES (exit 2, nothing else runs): resolved revision and template sha equal the registry;
     transformers pinned; option letters single tokens; split and H1 sets intact; no H1_held id in
     the pilot or screen order (Amendment 4 item 1). The DeltaNet kernel path is recorded.
  2. Padded-vs-alone check (0.05 nats); a failure switches every later cell to unbatched reads and
     generation and is flagged for ANOMALIES as the KDG-A20 pattern (Amendment 4 item 4).
  3. Trace-length pilot (8 x 4, cap 4,096) at the shipped default effort, stepping down while the
     p95 tokens-to-answer exceeds 1,536 (Amendment 2 item 3); projection of the rest of this
     model's run from the measured step time; exit 3 after banking if it exceeds the envelope.
  4. Muse only: the card-sampler 8 x 4 descriptive row if it projects under 0.2 A100-h.
  5. Dose-0 candidates in order (letter mass, forward vs generation, C0 at 0.9 kappa*, known-gap
     band on the C0 sample); the first passing all is of record; none -> no E regime.
  6. Panel dose-0 letter cells (586 x 8, five cells) when a candidate passed.
  7. H1 screen (first N of the seeded H1_train order x 4, cap 1,536), banked per batch, under the
     envelope deadline; post-reasoning letter readout; dose-0 readout on the same scenarios x 8;
     brake, truncation rate and the truncation-limited mark.
Writes ``timing.json`` (inputs to ``reprice_tier1.py``) and ``screen_record.json`` per model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import screen_lib as L  # noqa: E402  (puts the KDG scripts on sys.path)
from kdg_pod_lib import Manifest, sha256_text  # noqa: E402
from reasoning_formats import build_format  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

KDG = L.REPO / "papers" / "kdg_panel"
SPLIT = L.REPO / "papers" / "steering" / "split_v1.json"
H1_SET = KDG / "data" / "harm_divergence" / "hdiv_set.json"
H1_KEPT = KDG / "data" / "harm_divergence" / "h1_kept_ids.json"
TRANSFORMERS_PIN = "5.17.0"
SCREEN_N = {"olmo3_think": 100, "qwen38_27b": 60, "muse_glimmer_30b": 60}
# (pilot batch, screen generation batch, readout batch): OLMo-3-7B has 32 KV heads, so a batch of
# 32 at the 4,096 pilot cap needs ~80 GB of cache; Qwen3.8 (16 full-attention layers) and Muse
# (2 KV heads) fit at 32
BATCHES = {"olmo3_think": (8, 24, 16), "qwen38_27b": (32, 32, 16),
           "muse_glimmer_30b": (32, 32, 8)}
ENVELOPE_H = {"olmo3_think": 1.5, "qwen38_27b": 2.5, "muse_glimmer_30b": 2.5}
CARD_ROW_MAX_H = 0.2


def registry() -> dict:
    import yaml

    return yaml.safe_load((KDG / "models.yaml").read_text())["steering"]


def gates(sc: L.Screen, spec: dict, split: dict, order: list[str], H: dict, dry: bool) -> dict:
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, detail) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    held = set(split["H1_held"])
    # most probable failure of the whole screen: a held-out scenario enters selection
    check("no_h1_held_in_order", not (set(order) & held), sorted(set(order) & held)[:5])
    check("order_is_h1_train", set(order) == set(split["H1_train"]), len(order))
    check("h1_kept_all_present", all(s in H for s in split["H1_train"] + split["H1_held"]),
          len(H))
    tsha = sha256_text(sc.tok.chat_template)
    check("template_sha256", tsha == spec["template_sha256"], tsha)
    letters = L.rf.letter_token_ids(sc.tok)
    check("letters_single_tokens", sorted(letters.values()) == list("ABCDE"), letters)
    if not dry:
        import torch
        import transformers

        commit = getattr(sc.model.model.config, "_commit_hash", None)
        check("revision", commit == spec["revision"], commit)
        check("transformers_pin", transformers.__version__ == TRANSFORMERS_PIN,
              transformers.__version__)
        dt = next(sc.model.model.parameters()).dtype
        check("dtype_bf16", dt == torch.bfloat16, str(dt))
        check("n_layers", sc.model.wb.info.n_layers == spec["n_layers"],
              sc.model.wb.info.n_layers)
    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def kernel_path() -> dict:
    return {m: importlib.util.find_spec(m) is not None
            for m in ("kernels", "fla", "causal_conv1d", "flash_attn")}


def brake(screen_rows: list[dict], dose0_rows: list[dict]) -> dict:
    """B = mean_s P_reasoned(s) - mean_s P_dose0(s), paired by scenario (scenario bootstrap).
    P_reasoned: share of parsed sampled actions that protect; P_dose0: forced readout's
    renormalised protect mass, mean over 8 permutations."""
    import analyze_phase1_session_a as A

    pr: dict[str, list] = {}
    for r in screen_rows:
        if r.get("norm_status") is not None:
            pr.setdefault(r["scenario_id"], []).append(r["norm_status"] == "violating")
    pd: dict[str, list] = {}
    for r in dose0_rows:
        Ls = sorted(r["option_logps"])
        m = np.exp([r["option_logps"][x] for x in Ls])
        m = m / m.sum()
        pd.setdefault(r["scenario_id"], []).append(
            sum(q for x, q in zip(Ls, m) if r["norm_status_by_letter"][x] == "violating"))
    ids = sorted(s for s in pr if s in pd)
    d = np.array([np.mean(pr[s]) - np.mean(pd[s]) for s in ids])
    return {"B": A.boot(d), "P_reasoned": A.boot(np.array([np.mean(pr[s]) for s in ids])),
            "P_dose0": A.boot(np.array([np.mean(pd[s]) for s in ids])), "n_scenarios": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(SCREEN_N))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--envelope-hours", type=float, default=None)
    ap.add_argument("--date-pin", default=None, help="YYYY-MM-DD; default: today (UTC)")
    ap.add_argument("--stub-length", type=json.loads, default=None,
                    help="dry run only: mean trace length per effort level (JSON)")
    ap.add_argument("--stub-padding-noise", type=float, default=0.0)
    ap.add_argument("--stub-low-mass", default="", help="dry run only: comma list of candidates")
    ap.add_argument("--tiny-local", type=Path, default=None,
                    help="local test only: a tiny random checkpoint saved with the model's real "
                         "tokenizer; runs the real wrapper on CPU at reduced caps and subsets, "
                         "revision/pin/dtype/layer gates recorded as skipped")
    a = ap.parse_args()
    if a.tiny_local is not None:  # reduced caps so a CPU run takes seconds; never on a pod
        L.PILOT_CAP, L.CAP_OF_RECORD, L.C0_ANSWER_TOKENS = 24, 16, 2

    t_start = time.time()
    key, dry = a.model, a.dry_run
    spec = registry()[key]
    envelope = a.envelope_hours or ENVELOPE_H[key]
    deadline = t_start + envelope * 3600
    date_pin = a.date_pin or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    split = json.loads(SPLIT.read_text())
    order = L.h1_train_order(split)
    n_screen = SCREEN_N[key]
    pilot_ids, screen_ids = order[: L.PILOT_N], order[:n_screen]
    Hs, h1_meta = load_scenario_dir([H1_SET])
    kept = set(json.loads(H1_KEPT.read_text())["ids"])
    H = {s.id: s for s in Hs if s.id in kept}
    panel_all, metas = load_scenario_dir(sorted((KDG / "data").glob("*_scenarios_*.json")))
    panel = L.panel_586(panel_all)
    if not dry and len(panel) != 586:
        raise SystemExit(f"panel of record has {len(panel)} rows, expected 586")
    C0 = L.c0_sample(panel)
    if dry or a.tiny_local is not None:  # small enough to run in seconds, every unit exercised
        panel, C0 = C0[:6] + [s for s in panel if s.role != "primary"][:2], C0[:6]
        pilot_ids, screen_ids = pilot_ids[:4], screen_ids[:6]

    out = a.out / key
    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out, dry, metas + h1_meta)  # one manifest per model directory
    rec: dict = {"model": key, "spec": spec, "plan": "STEERING_PLAN v0.2.5 (Amendments 2-4)",
                 "date_pin": date_pin, "envelope_h": envelope, "screen_n": len(screen_ids),
                 "pilot_ids": pilot_ids, "screen_ids": screen_ids, "order_seed": L.ORDER_SEED,
                 "split_sha256": sha256_text(SPLIT.read_text()), "kernel_path": kernel_path()}
    manifest.data["steering_screen"] = rec

    def bank(status: str, code: int) -> int:
        rec["status"] = status
        rec["elapsed_h"] = (time.time() - t_start) / 3600
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix in (".jsonl", ".json", ".npz"):
                manifest.add(p, p.parent.name, key)
        L.Screen.save_json(sc, "screen_record.json", rec)
        manifest.add(out / "screen_record.json", "record", key)
        print(f"manifest: {manifest.write()}  status: {status}", flush=True)
        return code

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format(key, tok)
    t_load = time.time()
    if dry:
        from screen_stub import ScreenStub

        model = ScreenStub(key, tok, fmt, spec["n_layers"], length_by_effort=a.stub_length,
                           padding_noise=a.stub_padding_noise,
                           low_mass_candidates=tuple(x for x in a.stub_low_mass.split(",") if x))
    elif a.tiny_local is not None:
        import torch
        from kdg_pod_lib import ModelWrapper

        model = ModelWrapper(str(a.tiny_local), None, torch_dtype=torch.float32)
    else:
        import torch
        from kdg_pod_lib import ModelWrapper

        model = ModelWrapper(spec["repo"], spec["revision"], torch_dtype=torch.bfloat16)
    rec["load_seconds"] = time.time() - t_load
    pb, gb, rb = BATCHES[key]
    sc = L.Screen(key, spec, model, fmt, out, date_pin, dry=dry, gen_batch=pb, read_batch=rb)
    sc.effort = fmt.effort_ladder[0]

    g = gates(sc, spec, split, order, H, dry or a.tiny_local is not None)
    if a.tiny_local is not None:
        g["skipped"] = "revision, transformers pin, dtype, n_layers (tiny local checkpoint)"
        rec["tiny_local"] = str(a.tiny_local)
    rec["gates"] = g
    if not g["ok"]:
        print(json.dumps(g, indent=1, default=str), flush=True)
        return bank("gate_failed", 2)

    # 2. padded vs alone, on the pilot prompts at the default effort (mixed lengths)
    probe = [j["ids"] for j in L.h1_jobs(sc, H, order[:16], 1)]
    t0 = time.time()
    rec["padded_check"] = L.padded_check(sc, probe)
    rec["padded_check"]["seconds"] = time.time() - t0
    if not rec["padded_check"]["ok"]:
        sc.unbatched = True
        rec["anomaly_flag"] = ("KDG-A20 pattern: padded-vs-alone exceeds 0.05 nats; every cell "
                               "runs unbatched (Amendment 4 item 4); enter in ANOMALIES")
    rec["unbatched"] = sc.unbatched
    t0 = time.time()
    _, _ = L.read_ids(sc, probe, sc.bs(sc.read_batch))
    rec["read_seconds_per_row"] = (time.time() - t0) / len(probe)
    rec["probe_mean_len"] = float(np.mean([len(x) for x in probe]))

    # 3. trace-length pilot and step-down
    sd = L.step_down(sc, H, pilot_ids)
    rec["pilot"] = sd
    chosen = sd["tried"][-1]
    sc.gen_batch = gb
    plen = [r["tokens_to_answer"] for r in
            map(json.loads, (out / "pilot" / f"pilot_{sc.effort}.jsonl").read_text().splitlines())]
    rec["timing_pilot"] = chosen["timing"]
    n_reads = (len(C0) * 8 * 3 * len(fmt.dose0) + len(panel) * 8 * 5 + len(screen_ids) * 8)
    post_scale = (rec["probe_mean_len"] + float(np.mean(plen))) / rec["probe_mean_len"]
    proj = {
        "screen_generation_h": L.gen_hours(chosen["timing"], plen, len(screen_ids) * 4,
                                           chosen["timing"]["batch"]),
        "c0_generation_h": (len(C0) * 4 * len(fmt.dose0) / gb * L.C0_ANSWER_TOKENS
                            * (chosen["timing"]["seconds_per_step"] or 0.0)
                            + len(C0) * 4 * len(fmt.dose0) * rec["read_seconds_per_row"]) / 3600,
        "readout_h": n_reads * rec["read_seconds_per_row"] / 3600,
        "post_reasoning_readout_h": len(screen_ids) * 4 * rec["read_seconds_per_row"]
        * post_scale / 3600,
        "elapsed_h": (time.time() - t_start) / 3600,
        "note": "generation projected at the pilot batch's step time (conservative where the "
                "screen batch is larger)",
    }
    proj["total_h"] = sum(v for k, v in proj.items() if k.endswith("_h"))
    rec["projection"] = proj
    print(f">> {key} projection: {proj['total_h']:.2f} h against the {envelope} h envelope",
          flush=True)
    if proj["total_h"] > envelope:
        return bank("envelope_stop_after_pilot", 3)

    # 4. Muse's card sampler, descriptive only
    if key == "muse_glimmer_30b":
        cost = L.gen_hours(chosen["timing"], plen, L.PILOT_N * L.PILOT_R, chosen["timing"]["batch"],
                           cap=L.PILOT_CAP)
        rec["card_sampler_row"] = {"projected_h": cost, "run": cost <= CARD_ROW_MAX_H}
        if cost <= CARD_ROW_MAX_H:
            rows, timing = L.run_generation(sc, L.h1_jobs(sc, H, pilot_ids, L.PILOT_R),
                                            cap=L.CAP_OF_RECORD, stage="card_sampler",
                                            sampler=spec["card_sampler"])
            sc.save_jsonl("card_sampler/card_sampler.jsonl", rows)
            rec["card_sampler_row"].update(summary=L.screen_summary(rows), timing=timing,
                                           sampler=spec["card_sampler"],
                                           label="descriptive sensitivity row, never a statistic")

    # 5. dose-0 candidates
    rec["dose0"] = L.choose_dose0(sc, C0)

    # 6. panel letter cells
    if sc.dose0 is not None:
        rec["panel"] = [L.run_letter_cell(sc, panel, cell, "panel") for cell in L.PANEL_CELLS]
    else:
        rec["panel"] = "skipped: no dose-0 candidate passed (no E regime; Amendment 2 item 4)"

    # 7. H1 screen
    jobs = L.h1_jobs(sc, H, screen_ids, 4)
    rows, timing = L.run_generation(sc, jobs, cap=L.CAP_OF_RECORD, stage="screen",
                                    deadline=deadline)
    rec["timing_screen"] = timing
    pids = {(j["s"].id, j["k"]): j["ids"] for j in jobs}
    ok, lp, resid = L.post_reasoning_readout(sc, rows, pids)
    sc.save_jsonl("screen/h1_screen.jsonl", rows)
    np.savez_compressed(out / "screen" / "h1_screen_readout.npz", letter_logp=lp.astype(np.float16),
                        resid_decision=resid, rows=np.array(ok), layers=np.array(sc.layers()))
    summ = L.screen_summary(rows)
    last_level = sc.effort == fmt.effort_ladder[-1]
    summ["truncation_limited"] = bool(not summ["truncation_ok"] and last_level)
    rec["screen"] = summ
    cand = sc.dose0 or fmt.dose0[0]
    d0, d0lp, d0res, _ = L.letter_rows(sc, [H[s] for s in screen_ids], "dl_chat_neutral", cand)
    sc.save_jsonl("screen/h1_dose0.jsonl", d0)
    np.savez_compressed(out / "screen" / "h1_dose0.npz", letter_logp=d0lp.astype(np.float16),
                        resid_decision=d0res, layers=np.array(sc.layers()))
    rec["brake"] = brake(rows, d0)
    rec["brake"]["dose0_candidate"] = cand.name
    rec["brake"]["dose0_of_record"] = sc.dose0 is not None
    rec["effort_of_record"] = sc.effort
    L.Screen.save_json(sc, "timing.json", {
        "model": key, "effort_of_record": sc.effort, "unbatched": sc.unbatched,
        "load_seconds": rec["load_seconds"], "read_seconds_per_row": rec["read_seconds_per_row"],
        "probe_mean_len": rec["probe_mean_len"], "pilot": rec["timing_pilot"],
        "pilot_tokens_to_answer": plen, "screen": timing,
        "screen_tokens_to_answer": [r.get("tokens_to_answer") for r in rows
                                    if r["status"] != "not_run"],
        "batches": {"pilot": pb, "generation": gb, "read": rb}, "dry_run": dry})
    status = "ok" if timing["n_run"] == timing["n_jobs"] else "envelope_stop_in_screen"
    return bank(status, 0 if status == "ok" else 3)


if __name__ == "__main__":
    sys.exit(main())
