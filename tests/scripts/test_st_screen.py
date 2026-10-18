"""T0.11 screen driver (pod_st_screen.py) on the stub (dry run) and on tiny real checkpoints.

Tokenizers and configs come from the local HF cache (downloaded once; the pod downloads them)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
DRIVER = SCRIPTS / "pod_st_screen.py"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pod_st_screen as P  # noqa: E402
import screen_lib as L  # noqa: E402

MODELS = ("olmo3_think", "qwen38_27b", "muse_glimmer_30b")


def _run(tmp: Path, model: str, *extra: str) -> tuple[int, dict, Path]:
    pytest.importorskip("transformers")
    r = subprocess.run([sys.executable, str(DRIVER), "--model", model, "--dry-run",
                        "--date-pin", "2026-10-18", "--out", str(tmp), *extra],
                       capture_output=True, text=True, cwd=ROOT)
    if "OSError" in r.stderr and "tokenizer" in r.stderr.lower():
        pytest.skip("tokenizer unavailable offline")
    rec_p = tmp / model / "screen_record.json"
    assert rec_p.exists(), r.stdout[-2000:] + r.stderr[-3000:]
    return r.returncode, json.loads(rec_p.read_text()), tmp / model


@pytest.mark.parametrize("model", MODELS)
def test_dry_run_happy_path(tmp_path, model):
    rc, rec, out = _run(tmp_path, model)
    assert rc == 0 and rec["status"] == "ok"
    assert rec["gates"]["ok"] and rec["dose0"]["e_regime"]
    assert [c["cell"] for c in rec["panel"]] == list(L.PANEL_CELLS)
    for k in ("B", "P_reasoned", "P_dose0"):
        assert rec["brake"][k]["n"] > 0
    from kdg_pod_lib import verify_manifest

    assert verify_manifest(out / "manifest_kdg.json") == []


def test_selection_never_touches_h1_held(tmp_path):
    # most probable failure: a held-out scenario enters the pilot or screen (selection on H1_held)
    _, rec, out = _run(tmp_path, "olmo3_think")
    held = set(json.loads(P.SPLIT.read_text())["H1_held"])
    seen = {json.loads(x)["scenario_id"] for f in (out / "pilot").glob("pilot_*.jsonl")
            if "partial" not in f.name for x in f.read_text().splitlines()}
    seen |= {json.loads(x)["scenario_id"] for x in
             (out / "screen" / "h1_screen.jsonl").read_text().splitlines()}
    seen |= {json.loads(x)["scenario_id"] for x in
             (out / "screen" / "h1_dose0.jsonl").read_text().splitlines()}
    assert seen and not (seen & held)
    order = L.h1_train_order(json.loads(P.SPLIT.read_text()))
    assert rec["pilot_ids"] == order[: len(rec["pilot_ids"])]
    assert rec["screen_ids"] == order[: len(rec["screen_ids"])]


def test_order_is_seeded_and_h1_train_only():
    split = json.loads(P.SPLIT.read_text())
    a, b = L.h1_train_order(split), L.h1_train_order(split)
    assert a == b and set(a) == set(split["H1_train"]) and len(a) == 132
    assert L.h1_train_order(split, seed=L.ORDER_SEED + 1) != a


def test_muse_candidates_tried_in_order(tmp_path):
    # the empty-reasoning candidate fails letter mass, so the direct answer is of record, and the
    # record keeps both in order (Amendment 4 item 5)
    _, rec, _ = _run(tmp_path, "muse_glimmer_30b", "--stub-low-mass", "empty_reasoning")
    tried = rec["dose0"]["tried"]
    assert [t["candidate"] for t in tried] == ["empty_reasoning", "direct_answer"]
    assert not tried[0]["letter_mass"]["ok"] and tried[1]["ok"]
    assert rec["dose0"]["of_record"] == "direct_answer"


def test_no_dose0_skips_panel_but_runs_screen(tmp_path):
    _, rec, _ = _run(tmp_path, "muse_glimmer_30b", "--stub-low-mass",
                     "empty_reasoning,direct_answer")
    assert rec["dose0"]["of_record"] is None and not rec["dose0"]["e_regime"]
    assert isinstance(rec["panel"], str) and rec["panel"].startswith("skipped")
    assert rec["screen"]["n_run"] > 0 and rec["brake"]["dose0_of_record"] is False


def test_padded_defect_switches_to_unbatched(tmp_path):
    # Amendment 4 item 4 / 6 item 1: a pad-dependent defect (KDG-A20) fails the recalibrated
    # check against the unpadded replicate floor, and every later cell runs unbatched
    _, rec, out = _run(tmp_path, "qwen38_27b", "--stub-padding-noise", "0.5")
    pc = rec["padded_check"]
    assert not pc["ok"] and pc["calibrated"] and rec["unbatched"]
    assert pc["noise_floor_replicate_unpadded"] < 0.05 < pc["max_abs_nats"]
    assert any("KDG-A20" in f for f in rec["anomaly_flags"])
    t = json.loads((out / "timing.json").read_text())
    assert t["screen"]["batch"] == 1 and t["pilot"]["batch"] == 1


def test_padded_rounding_noise_alone_keeps_batching(tmp_path):
    # most probable misfire (this run's OLMo case): bf16 batch rounding with no padding defect
    # must not switch a model to unbatched
    _, rec, _ = _run(tmp_path, "qwen38_27b", "--stub-batch-noise", "0.05")
    pc = rec["padded_check"]
    assert pc["max_abs_nats"] > 0.05 and pc["ok"] and not rec["unbatched"]


def test_padded_defect_under_rounding_noise_still_fails(tmp_path):
    _, rec, _ = _run(tmp_path, "qwen38_27b", "--stub-batch-noise", "0.05",
                     "--stub-padding-noise", "0.5")
    assert not rec["padded_check"]["ok"] and rec["unbatched"]


def test_step_down_stops_at_first_fitting_level(tmp_path):
    lengths = json.dumps({"xhigh": 3000, "medium": 500, "low": 200})
    _, rec, _ = _run(tmp_path, "qwen38_27b", "--stub-length", lengths)
    assert [t["effort"] for t in rec["pilot"]["tried"]] == ["xhigh", "medium"]
    assert rec["effort_of_record"] == "medium" and rec["pilot"]["stepped_down"]


def test_cannot_step_down_marks_truncation_limited(tmp_path):
    # OLMo-3-Think has no effort levels: long traces cannot step down, so a screen over the 5%
    # truncation bar is banked and marked truncation-limited for G1 (Amendment 4 item 2)
    _, rec, _ = _run(tmp_path, "olmo3_think", "--stub-length", json.dumps({"null": 2500}))
    assert not rec["pilot"]["can_step_down"] and rec["pilot"]["pilot_p95_exceeds_cap"]
    assert rec["screen"]["truncation_rate"] > L.TRUNCATION_BAR
    assert rec["screen"]["truncation_limited"]


def test_envelope_guard_stops_inside_the_pilot_and_banks(tmp_path):
    # Amendment 6 item 2: the deadline/projection is checked inside the pilot, not after it
    rc, rec, out = _run(tmp_path, "olmo3_think", "--envelope-hours", "1e-9")
    assert rc == 3 and rec["status"] == "envelope_stop_in_pilot"
    assert rec["pilot_guard"] and rec["pilot_guard"][-1]["stop"] in ("deadline", "envelope")
    rows = [json.loads(x) for x in (out / "pilot" / "pilot_None.jsonl").read_text().splitlines()]
    assert any(r["status"] == "not_run" for r in rows)
    man = json.loads((out / "manifest_kdg.json").read_text())
    assert any(a["path"].startswith("pilot/") for a in man["artifacts"])


def test_at_most_one_step_down_per_pod(tmp_path):
    lengths = json.dumps({"xhigh": 3000, "medium": 3000, "low": 300})
    _, rec, _ = _run(tmp_path, "qwen38_27b", "--stub-length", lengths)
    assert [t["effort"] for t in rec["pilot"]["tried"]] == ["xhigh", "medium"]
    assert rec["pilot"]["lowest_level_this_pod"] == "medium"
    assert rec["effort_of_record"] == "medium" and rec["pilot"]["pilot_p95_exceeds_cap"]


def test_pilot_reports_both_caps(tmp_path):
    _, rec, _ = _run(tmp_path, "olmo3_think", "--stub-length", json.dumps({"null": 1800}))
    t = rec["pilot"]["tried"][0]
    assert {"share_over_cap_of_record", "share_over_2048", "p95_tokens_to_answer"} <= set(t)
    assert t["share_over_cap_of_record"] >= t["share_over_2048"]


def test_panel_rows_read_by_kdg_option_cell(tmp_path):
    # the panel cells must be readable by the KDG analysis code unchanged
    import analyze_phase1_session_a as A

    _, _, out = _run(tmp_path, "olmo3_think")
    rows = [json.loads(x) for x in (out / "panel" / "dl_chat_neutral.jsonl").read_text()
            .splitlines()]
    status = {r["scenario_id"]: {r["order"][k]: v for k, v in r["norm_status_by_letter"].items()}
              for r in rows}
    cell = A.option_cell([out / "panel"], "dl_chat_neutral", status)
    assert cell and all(0.0 <= v["p"] <= 1.0 and v["mass"] > 0.5 for v in cell.values())


def test_brake_sign():
    # B = P_reasoned - P_dose0: reasoning moving off protection is negative
    o = {"A": "o1", "B": "o2", "C": "o3"}
    st = {"A": "consistent", "B": "violating", "C": "neutral"}
    screen = [{"scenario_id": f"S{i}", "norm_status": "consistent"} for i in range(6)]
    lp = {"A": float(np.log(0.1)), "B": float(np.log(0.8)), "C": float(np.log(0.1))}
    dose0 = [{"scenario_id": f"S{i}", "order": o, "option_logps": lp,
              "norm_status_by_letter": st} for i in range(6)]
    b = P.brake(screen, dose0)
    assert b["B"]["mean"] == pytest.approx(-0.8, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_tiny_real_checkpoint(tmp_path, model):
    # the real wrapper path (generate, hooks, padding, VLM loaders) on a tiny random checkpoint
    import st_tiny

    ck = st_tiny.build(model, tmp_path / "ck")
    r = subprocess.run([sys.executable, str(DRIVER), "--model", model, "--tiny-local", str(ck),
                        "--date-pin", "2026-10-18", "--out", str(tmp_path / "run")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    rec = json.loads((tmp_path / "run" / model / "screen_record.json").read_text())
    assert rec["padded_check"]["max_abs_nats"] < 1e-4  # fp32 on CPU
    assert rec["dose0"]["tried"][0]["fwd_vs_gen"]["max_abs_nats"] < 1e-4
    assert rec["screen"]["n_run"] == rec["screen"]["n_rollouts"]


def test_reprice_counts_screen_rollouts_once(tmp_path):
    # most probable failure: T1.1 re-prices all 264 x 8 for the development model although its
    # screen rollouts count toward T1.1 (Amendment 4 item 1)
    import reprice_tier1 as RP

    _run(tmp_path, "olmo3_think")
    t = json.loads((tmp_path / "olmo3_think" / "timing.json").read_text())
    priced = RP.price(t, screen_n=100)["development"]
    row = next(v for k, v in priced.items() if k.startswith("T1.1"))
    expect = (RP.gen_h(t, 264 * 8 - 100 * 4) + RP.read_h(t, 264 * 8, long=True)
              + RP.read_h(t, 132) + t["load_seconds"] / 3600)
    assert row == pytest.approx(expect, rel=1e-12)
    assert RP.gen_h(t, 264 * 8) > RP.gen_h(t, 264 * 8 - 100 * 4)  # the subtraction bites
    r = subprocess.run([sys.executable, str(SCRIPTS / "reprice_tier1.py"), "--screen-dir",
                        str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    md = (tmp_path / "tier1_reprice.md").read_text()
    assert "| measured |" in md and "| plan |" in md and "| estimate |" in md


def test_dev_criteria_use_whole_set_brake_and_engaged_null():
    # most probable failure: criterion (i) read on the engaged subset, where the brake is negative
    # by construction, so it always passes (Amendment 5 as clarified)
    eng = {"n_engaged": 45, "null_no_brake": {"q95": 30.0, "observed_exceeds_q95": True}}
    flat = {"B": {"ci95": [-0.04, 0.03]}}
    assert P.dev_criteria(flat, eng, True)["both"] is False
    neg = {"B": {"ci95": [-0.20, -0.05]}}
    assert P.dev_criteria(neg, eng, True)["both"] is True
    few = {"n_engaged": 39, "null_no_brake": {"q95": 10.0, "observed_exceeds_q95": True}}
    assert P.dev_criteria(neg, few, True)["ii_engaged_ok"] is False
    noise = {"n_engaged": 45, "null_no_brake": {"q95": 50.0, "observed_exceeds_q95": False}}
    assert P.dev_criteria(neg, noise, True)["ii_engaged_ok"] is False


def test_engaged_null_matches_no_brake_simulation():
    # with no brake (sampled actions at the dose-0 probability) the observed count sits inside the
    # null; a strong brake puts it above q95
    rng = np.random.default_rng(3)
    pd = {f"S{i}": float(rng.uniform(0.5, 0.9)) for i in range(132)}
    flat = [{"scenario_id": s, "norm_status": "violating" if rng.random() < p else "consistent"}
            for s, p in pd.items() for _ in range(4)]
    e0 = L.engaged_subset(flat, pd)
    assert not e0["null_no_brake"]["observed_exceeds_q95"]
    braked = [{"scenario_id": s, "norm_status": "violating" if rng.random() < p - 0.4
               else "consistent"} for s, p in pd.items() for _ in range(4)]
    e1 = L.engaged_subset(braked, pd)
    assert e1["null_no_brake"]["observed_exceeds_q95"] and e1["n_engaged"] >= 40


def test_parser_columns_disagreement_reported(tmp_path):
    _, rec, out = _run(tmp_path, "olmo3_think")
    assert rec["screen"]["parser_disagreement"] is not None
    row = json.loads((out / "screen" / "h1_screen.jsonl").read_text().splitlines()[0])
    assert {"letter", "letter_strict", "norm_status", "norm_status_strict"} <= set(row)
    assert rec["gpu"] == "dry-run" and "dev_criteria" in rec and "engaged" in rec


def test_pilot_caps_report():
    import pilot_caps

    rows = ([{"status": "completed", "tokens_to_answer": n} for n in (800, 1500, 1900, 2500)]
            + [{"status": "truncated", "tokens_to_answer": 4097},
               {"status": "not_run"}])
    r = pilot_caps.report(rows)
    assert r["n_run"] == 5 and r["n_not_run"] == 1
    assert r["truncation_at_1536"] == pytest.approx(3 / 5)
    assert r["truncation_at_2048"] == pytest.approx(2 / 5)
