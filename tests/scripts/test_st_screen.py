"""T0.11 screen driver (pod_st_screen.py) on the stub (dry run) and on tiny real checkpoints.

Tokenizers and configs come from the local HF cache (downloaded once; the pod downloads them)."""

from __future__ import annotations

import json
import subprocess
import sys
import types
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


def _run(tmp: Path, model: str, *extra: str, engine: str = "hf") -> tuple[int, dict, Path]:
    # the tests below exercise the HF engine path unless they pass engine="vllm" (Amendment 7)
    pytest.importorskip("transformers")
    r = subprocess.run([sys.executable, str(DRIVER), "--model", model, "--dry-run",
                        "--engine", engine, "--date-pin", "2026-10-18", "--out", str(tmp),
                        *extra],
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
    # per-row evidence behind the maxima is saved (author rider (c), 2026-10-19)
    pr = pc["per_row"]
    assert len(pr["letter_logprobs_alone"]) == pc["n"] == len(pc["lengths"])
    assert max(pr["max_abs_vs_alone_composition_0"]) == pytest.approx(pc["max_abs_nats"], abs=1e-3)
    assert max(pr["max_abs_vs_alone_replicate"]) == pytest.approx(
        pc["noise_floor_replicate_unpadded"], abs=1e-3)
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
                        "--engine", "hf", "--date-pin", "2026-10-18",
                        "--out", str(tmp_path / "run")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    rec = json.loads((tmp_path / "run" / model / "screen_record.json").read_text())
    assert rec["padded_check"]["max_abs_nats"] < 1e-4  # fp32 on CPU
    assert rec["dose0"]["tried"][0]["fwd_vs_gen"]["max_abs_nats"] < 1e-4
    assert rec["screen"]["n_run"] == rec["screen"]["n_rollouts"]


def test_reprice_counts_screen_rollouts_once(tmp_path):
    # most probable failure: T1.1 re-prices all 264 x 8 for the development model although its
    # screen rollouts count toward T1.1 (Amendment 4 item 1); also: GPT-OSS readouts priced at the
    # batched rate (ST-A3 makes them single-row, §31) and a cumulative column that ignores SPENT
    import reprice_tier1 as RP

    _run(tmp_path, "qwen38_27b", engine="vllm")
    _run(tmp_path, "gpt_oss_20b", engine="vllm")
    q = RP.model_rates(tmp_path / "qwen38_27b")
    assert q["stage"] == "screen" and q["screen_rollouts"] > 0
    dev = RP.price(q)["development"]
    (k, (h, _)), = [(k, v) for k, v in dev.items() if k.startswith("T1.1 generation")]
    assert h == pytest.approx(RP.gen_h(q, 264 * 8 - q["screen_rollouts"]), rel=1e-12)
    assert RP.gen_h(q, 264 * 8) > h  # the subtraction bites
    g = RP.model_rates(tmp_path / "gpt_oss_20b")
    assert g["stage"] == "parity" and g["screen_rollouts"] == 0 and g["single_row"]
    rate, basis = RP.read_rate(g)
    assert basis.startswith("§31") and rate == pytest.approx(
        RP.S31_SINGLE_ROW_S * g["prompt_len"] / RP.S31_TOKENS)
    assert rate > 100 * g["batched_read_s"]  # the stub's batched rate is far cheaper
    # a dense model whose floor exceeds the bar while its padded check passed: one batch forward
    q2 = dict(q, single_row=True)
    assert RP.read_rate(q2)[0] == pytest.approx(
        q["batched_read_s"] * min(q["rec"]["padded_check"]["batch"], 16))
    # a probe recorded before the warm-up fix: its cold slope is not used, the warm 2n bound is
    cold = {"largest_fitting_batch": 8, "probes": [
        {"batch": 4, "fits": True, "seconds_per_step": 0.5, "t_decode_2n": 1.0, "decode_steps": 32},
        {"batch": 8, "fits": True, "seconds_per_step": 0.001, "t_decode_2n": 6.4,
         "decode_steps": 32}]}
    assert RP.hf_step(cold) == pytest.approx(0.1)
    cold["probes"][1]["warmed"] = True
    assert RP.hf_step(cold) == pytest.approx(0.001)
    # the relaunch's Muse rung: compile raised, uncompiled SDPA kept; its alone pass is the
    # measured single-row rate (16 rows in 1.96 s), never the failed attempt's
    rung = {"kept": "fast", "n": 16, "attempts": [
        {"compiled": True, "ok": False, "error": "NameError"},
        {"compiled": False, "ok": True, "first_fast_pass_seconds": 1.96}]}
    assert RP.rung_alone_rate(rung) == pytest.approx(1.96 / 16)
    assert RP.rung_alone_rate({"kept": "eager"}) is None
    assert RP.read_rate(dict(q2, alone_read_s=0.1225))[0] == 0.1225
    # T1.3 never priced from a probe on uncompiled flex attention (author rider (b), 2026-10-19)
    gm = dict(g, hf_step_s=2.68, hf_batch=132, lens=[1536] * 64, rec=dict(
        g["rec"], fast_path_rung={"kept": "fast", "fast_impl": "flex_attention"}))
    assert RP.gate_price(gm)[0] == 1.5 and RP.gate_price(gm)[1].startswith("plan")
    gm["rec"]["fast_path_rung"] = {"kept": "eager"}
    assert RP.gate_price(gm)[0] > 1.5  # the same probe on eager is used as measured
    r = subprocess.run([sys.executable, str(SCRIPTS / "reprice_tier1.py"), "--screen-dir",
                        str(tmp_path), "--session-h", "1.0"], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode == 0, r.stderr
    md = (tmp_path / "tier1_reprice.md").read_text()
    assert "| measured" in md and "| plan |" in md and "| estimate |" in md
    js = json.loads((tmp_path / "tier1_reprice.json").read_text())
    spent = sum(RP.SPENT.values()) + 1.0
    assert js["spent"]["total"] == pytest.approx(spent)
    first = next(x for x in md.splitlines() if x.startswith("| T1.1 generation"))
    h1, cum1 = (float(c) for c in first.split("|")[2:4])
    assert cum1 == pytest.approx(spent + h1, abs=0.011)


def test_dev_criteria_use_whole_set_brake_and_engaged_null():
    # most probable failure: criterion (i) read on the engaged subset, where the brake is negative
    # by construction, so it always passes (Amendment 5 as clarified)
    eng = {"n_engaged": 45, "null_no_brake": {"q95": 30.0, "observed_exceeds_q95": True}}
    flat = {"B": {"ci95": [-0.04, 0.03]}}
    assert P.dev_criteria(flat, eng, True)["both"] is False
    neg = {"B": {"ci95": [-0.20, -0.05]}}
    assert P.dev_criteria(neg, eng, True)["both"] is True
    # G1 (2026-10-19): (i) is directional; Qwen3.8's B +0.083 [0.020, 0.146] is a sign flip
    pos = {"B": {"ci95": [0.020, 0.146]}}
    assert P.dev_criteria(pos, eng, True)["i_whole_set_brake_ci_below_0"] is False
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



# ---- Amendment 7: the vLLM engine path (stub generation in-process; vLLM itself on the pod) ----


@pytest.mark.parametrize("model", ("qwen38_27b", "muse_glimmer_30b"))
def test_vllm_path_screen_end_to_end(tmp_path, model):
    rc, rec, out = _run(tmp_path, model, engine="vllm")
    assert rc == 0 and rec["status"] == "ok" and rec["engine"] == "vllm"
    pv = rec["phase_v"]
    assert pv["pilot"][0]["probe"]["regime"] == "pilot" and pv["screen_probe"]["regime"] == "screen"
    rows = [json.loads(x) for x in (out / "screen" / "h1_screen.jsonl").read_text().splitlines()]
    # most probable failure: rows of record mixing the probe's two settings
    assert {r["vllm_choice"] for r in rows} == {pv["screen_probe"]["choice"]}
    assert all(r["engine"] == "vllm" for r in rows)
    assert rec["fast_path_rung"]["kept"] == "fast" and rec["dose0"]["e_regime"]
    assert (out / "vllm" / "screen_nospec.jsonl").exists()  # the probe duplicate is banked


def test_vllm_probe_keeps_the_faster_setting_muse_rule(tmp_path):
    # author rule: if DFlash loses the with/without probe at Muse's screen batch, the screen runs
    # without speculation (the stub makes speculation slower here)
    _, rec, out = _run(tmp_path, "muse_glimmer_30b", "--stub-spec-factor", "1.5", engine="vllm")
    assert rec["phase_v"]["screen_probe"]["choice"] == "nospec"
    rows = [json.loads(x) for x in (out / "screen" / "h1_screen.jsonl").read_text().splitlines()]
    assert {r["vllm_choice"] for r in rows} == {"nospec"}


def test_vllm_path_gpt_oss_probe_and_parity(tmp_path):
    rc, rec, out = _run(tmp_path, "gpt_oss_20b", engine="vllm")
    assert rc == 0 and rec["status"] == "ok_probe"
    assert rec["engine_parity"]["n_scenarios"] > 0 and "protective_ok" in rec["engine_parity"]
    assert rec["phase_v"]["parity_probe"]["regime"] == "parity"


def test_parity_prompts_are_section40s_exact_prompts():
    # most probable failure: the parity jobs render a prompt §40 never saw (the comparison would
    # then test the prompt, not the engine); every sha must equal §40's
    pytest.importorskip("transformers")
    import types

    import screen_vllm as SV
    import yaml
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    from deepsteer.kdg.schema import load_scenario_dir

    spec = yaml.safe_load((ROOT / "papers/kdg_panel/models.yaml").read_text())["st"]["gpt_oss_20b"]
    try:
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    except OSError:
        pytest.skip("tokenizer unavailable offline")
    fmt = build_format("gpt_oss_20b", tok, spec)
    sc = L.Screen("gpt_oss_20b", spec, L.tok_only(tok), fmt, Path("/tmp"), "2026-10-18")
    sc.effort = "medium"
    Hs, _ = load_scenario_dir([P.H1_SET])
    ctx = types.SimpleNamespace(order=L.h1_train_order(json.loads(P.SPLIT.read_text())),
                                H={s.id: s for s in Hs})
    jobs, ids = SV.parity_jobs(sc, ctx)  # raises on any sha mismatch (not a dry run)
    assert len(ids) == 64 and len(jobs) == 256
    assert not set(ids) & set(json.loads(P.SPLIT.read_text())["H1_held"])


def test_rung_bar_comes_from_the_reference_floor(tmp_path, monkeypatch):
    # most probable failure (the relaunch's GPT-OSS rung): the bar is raised by the candidate's
    # own replicate floor, so a noisier fused path passes more easily than a cleaner one
    # (uncompiled flex: floor 2.28, measured 1.84, passed; compiled: floor 0.52, 1.45, failed)
    rng = np.random.default_rng(0)
    base = rng.normal(-5, 1, (16, 5))
    state = {"impl": "eager"}

    class FakeHF:
        def set_attn_implementation(self, impl):
            state["impl"] = impl

        def forward(self, *a, **k):
            return None

    def fake_compile(f, dynamic):
        def g(*a, **k):
            return f(*a, **k)
        g.compiled = True
        return g

    hf = FakeHF()
    model = types.SimpleNamespace(model=hf, torch=types.SimpleNamespace(compile=fake_compile))
    sc = types.SimpleNamespace(dry=False, model=model, read_batch=16)

    def fake_read(sc_, id_lists, batch):
        # the relaunch's numbers: candidates offset from the reference, their batch jitter on top;
        # the one-position-off control (one id shorter) far away
        comp = getattr(hf.forward, "compiled", False)
        out = []
        for x in id_lists:
            r = base[x[0]].copy() + (26.0 if len(x) == 2 else 0.0)
            if state["impl"] != "eager":
                r += (1.45 if comp else 1.84) + (0.0 if batch == 1 else 0.52 if comp else 2.28)
            elif batch > 1:
                r += 0.43
            out.append(r)
        return np.stack(out), None

    monkeypatch.setattr(L, "read_ids", fake_read)
    monkeypatch.setattr(L, "install_hooks", lambda sc_: None)
    ids = [[k, 0, 0] for k in range(16)]
    r = L.fast_path_rung(sc, ids, "flex_attention", ref_floor=0.43)
    assert r["attempts"][0]["bar"] == pytest.approx(0.86)  # 2 x the reference floor, not 2 x 0.52
    assert not r["attempts"][0]["ok"]  # compiled: 1.45 > 0.86
    # uncompiled: 1.84 > 0.86 and its own floor 2.28 > 0.86 -> fails; the model stays eager
    assert r["kept"] == "eager" and state["impl"] == "eager"
    assert all(a["noise_floor_replicate_reference"] == 0.43 for a in r["attempts"])
    assert len(r["attempts"][1]["per_row"]["max_abs_vs_ref_fast"]) == 16


@pytest.mark.slow
@pytest.mark.parametrize("model,impl", (("gpt_oss_20b", "flex_attention"), ("qwen38_27b", "sdpa")))
def test_fast_path_rung_on_a_tiny_checkpoint(tmp_path, model, impl):
    # Amendment 7 items 4 and 7: the fused attention + torch.compile readout against the eager
    # reference on the Amendment 6 ladder, on real tensors (CPU, fp32)
    import st_tiny
    import torch
    import yaml
    from kdg_pod_lib import ModelWrapper
    from reasoning_formats import build_format

    spec = yaml.safe_load((ROOT / "papers/kdg_panel/models.yaml").read_text())["st"][model]
    ck = st_tiny.build(model, tmp_path / "ck")
    m = ModelWrapper(str(ck), None, torch_dtype=torch.float32, attn_implementation="eager")
    fmt = build_format(model, m.tok, spec)
    sc = L.Screen(model, spec, m, fmt, tmp_path, "2026-10-18", read_batch=4)
    msgs = [{"role": "user", "content": "Pick one.\n\nA. rule\nB. protect" + " x" * k}
            for k in (1, 5, 9, 13)]
    ids = [fmt.prompt_ids(m.tok, fmt.render(m.tok, [mm], fmt.effort_ladder[0], "2026-10-18"))
           for mm in msgs]
    r = L.fast_path_rung(sc, ids, impl)
    # a fused path must hold, compiled where Inductor can compile it (Qwen3.5's DeltaNet torch
    # fallback cannot on CPU; the failed attempt is recorded with its error, never silent)
    assert r["ok"] and r["calibrated"] and r["kept"].startswith("fast"), r
    assert all("error" in t or "max_abs_nats" in t for t in r["attempts"])


def test_revision_gate_reads_the_composite_commit_for_a_text_tower_load(monkeypatch):
    # most probable failure (the relaunch's Qwen3.8 gate_failed): a text-tower load's config is the
    # composite's text_config, which carries no _commit_hash, so the gate compares None to the pin
    import transformers

    spec = {"repo": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
    text_tower = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=None))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda repo, revision: (
        types.SimpleNamespace(_commit_hash=revision)))
    got = P.loaded_commit(text_tower, spec)
    assert got["commit"] == spec["revision"] and got["source"].startswith("AutoConfig")
    full = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash="abc"))
    assert P.loaded_commit(full, spec) == {"commit": "abc", "source": "model.config"}


def _resume(tmp: Path, model: str, banked: Path, *extra: str):
    r = subprocess.run([sys.executable, str(DRIVER), "--model", model, "--dry-run", "--engine",
                        "vllm", "--date-pin", "2026-10-18", "--out", str(tmp), "--resume-from",
                        str(banked), *extra], capture_output=True, text=True, cwd=ROOT)
    rec_p = tmp / model / "screen_record.json"
    return r, (json.loads(rec_p.read_text()) if rec_p.exists() else None)


def test_resume_runs_phase_h_on_the_banked_rows(tmp_path):
    # most probable failure (the Qwen rerun, author 2026-10-19): phase H reads rows other than
    # the banked ones (a fresh render, another effort or date pin), or regenerates them
    _, old, banked = _run(tmp_path / "a", "qwen38_27b", engine="vllm")
    r, rec = _resume(tmp_path / "b", "qwen38_27b", banked)
    assert r.returncode == 0 and rec["status"] == "ok" and rec["mode"] == "resume", r.stderr[-2000:]
    assert rec["resumed_from"]["dir"] == str(banked)
    assert rec["phase_v"]["effort_of_record"] == old["phase_v"]["effort_of_record"]
    new_rows = (tmp_path / "b" / "qwen38_27b" / "screen" / "h1_screen.jsonl").read_text()
    assert new_rows == (banked / "screen" / "h1_screen.jsonl").read_text()
    fg = rec["dose0"]["tried"][0]["fwd_vs_gen"]  # Amendment 8: the gate and its two columns
    assert fg["comparison"] == "HF forward alone vs HF generation alone" and fg["bar"] == 0.05
    assert fg["post_hoc_two_floor"]["label"].startswith("post hoc")
    vd = fg["vllm_descriptive"]
    assert vd["n_letter_values"] == vd["n_letter_values_expected"] > 0
    assert len(fg["per_row"]["max_abs_vs_forward_alone_generation_alone"]) == fg["n"]


def test_resume_refuses_banked_rows_that_do_not_match(tmp_path):
    # most probable failure: a banked engine at another revision, or a banked row rendered
    # differently, passes into phase H unnoticed
    _, _, banked = _run(tmp_path / "a", "qwen38_27b", engine="vllm")
    eng = sorted((banked / "vllm").glob("*.engine.json"))[0]
    e = json.loads(eng.read_text())
    eng.write_text(json.dumps({**e, "revision": "0" * 40}))
    r, _ = _resume(tmp_path / "b", "qwen38_27b", banked)
    assert r.returncode != 0 and "is not the registry pin" in r.stderr
    eng.write_text(json.dumps(e))
    rows_p = banked / "screen" / "h1_screen.jsonl"
    rows = rows_p.read_text().splitlines()
    first = json.loads(rows[0])
    rows_p.write_text("\n".join([json.dumps({**first, "rendered_sha256": "x"})] + rows[1:]) + "\n")
    r, _ = _resume(tmp_path / "c", "qwen38_27b", banked)
    assert r.returncode != 0 and "does not match this run's render" in r.stderr


def test_dose0_recheck_and_oss_rider_modes(tmp_path):
    # riders (a) and (b): Muse stops after the dose-0 checks on its banked C0 answers; GPT-OSS
    # runs no phase V and records an eager memory probe
    _, _, banked = _run(tmp_path / "a", "muse_glimmer_30b", engine="vllm")
    r, rec = _resume(tmp_path / "b", "muse_glimmer_30b", banked, "--dose0-only")
    assert r.returncode == 0 and rec["status"] == "ok_dose0_recheck", r.stderr[-2000:]
    assert [t["candidate"] for t in rec["dose0"]["tried"]][0] == "empty_reasoning"
    assert "panel" not in rec and "brake" not in rec
    rc, rec, _ = _run(tmp_path / "c", "gpt_oss_20b", "--rider-oss", engine="vllm")
    assert rc == 0 and rec["status"] == "ok_rider" and rec["mode"] == "rider_oss"
    assert rec["hf_memory_probe"]["path"] == "eager, uncompiled"
    assert "engine_parity" not in rec and not list((tmp_path / "c" / "gpt_oss_20b").glob("vllm/*"))


def test_reprice_merges_the_rider_session_and_reads_pod_timestamps(tmp_path):
    # most probable failure: the rider pod's GPT-OSS record (no generation timing) is dropped, so
    # the re-price keeps the relaunch's probe on uncompiled flex; or session hours stay a guess
    import reprice_tier1 as RP

    _run(tmp_path / "t10", "gpt_oss_20b", engine="vllm")
    _run(tmp_path / "t10b", "gpt_oss_20b", "--rider-oss", engine="vllm")
    rider = json.loads((tmp_path / "t10b" / "gpt_oss_20b" / "screen_record.json").read_text())
    rider["hf_memory_probe"] = {"largest_fitting_batch": 66, "path": "eager, uncompiled",
                                "probes": [{"batch": 66, "fits": True, "warmed": True,
                                            "seconds_per_step": 0.2}]}
    rider["fast_path_rung"] = {"kept": "fast", "fast_impl": "flex_attention"}
    m = RP.model_rates(tmp_path / "t10" / "gpt_oss_20b", (rider,))
    assert m["stage"] == "parity" and m["hf_step_s"] == 0.2 and m["hf_batch"] == 66
    h, basis = RP.gate_price(m)
    assert basis == "measured" and h != 1.5  # an eager probe prices T1.3 whatever the rung kept
    (tmp_path / "t10b" / "session_start.json").write_text(
        '{"session_start_utc": "2026-10-20T01:00:00Z", "validate": 0}')
    (tmp_path / "t10b" / "session_end.json").write_text(
        '{"session_end_utc": "2026-10-20T03:30:00Z", "rc": 0}')
    assert RP.session_hours(tmp_path / "t10b") == pytest.approx(2.5 + RP.PROVISION_H)
    r = subprocess.run([sys.executable, str(SCRIPTS / "reprice_tier1.py"), "--screen-dir",
                        str(tmp_path / "t10"), str(tmp_path / "t10b")], capture_output=True,
                       text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "pod timestamps" in (tmp_path / "t10b" / "tier1_reprice.md").read_text()
