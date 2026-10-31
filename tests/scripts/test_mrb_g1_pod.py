"""MRB-G1 pod driver (papers/mrb_morebench/scripts/mrb_g1_pod.py): tiers, cuts, dry run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from deepsteer.datasets.morebench import MoReBenchCriterion, MoReBenchItem, default_cache_dir

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mrb_morebench" / "scripts"))

import mrb_g1_pod as P  # noqa: E402


def _items(n=20):
    crit = tuple(MoReBenchCriterion(f"c{j}", f"criterion {j}", 2, "identifying") for j in range(3))
    return [MoReBenchItem(f"mrb-{i:03d}", i, f"dilemma {i}", "daily_dilemmas", "short_case",
                          "ai_advisor", "neutral", "ctx", crit) for i in range(n)]


def _gens(items):
    g = {}
    for it in items:
        for m in P.GEN:
            for k in range(P.K):
                g[f"{m}/default/{it.scenario_id}/k{k}"] = {"trace": "t", "response": "r"}
            if P.in_ladder_subset(it.idx):
                g[f"{m}/fork/{it.scenario_id}/k0"] = {"trace": "t", "response": "r"}
    return g


def test_each_cut_drops_exactly_its_cells_and_keeps_field_parity():
    # most probable failure: the response field lands in a different tier than the trace (a cut
    # would break the author's parity rule), or a cut tier holds the wrong model or k
    items = _items()
    units = P.judge_units(items, _gens(items), validate=False)
    for u in units:
        if u["unit"].startswith(("gpt_oss_20b/", "qwen3_30b/")):
            model, arm, sid, kk, field = u["unit"].split("/")[:5]
            k = int(kk[1:])
            want = 1 if (arm == "fork" or k < 2) else (2 if model == "gpt_oss_20b" else 3)
            assert u["tier"] == want, u["unit"]
    trace = {u["unit"].replace("/trace", "/F"): u["tier"] for u in units if "/trace" in u["unit"]}
    resp = {u["unit"].replace("/response", "/F"): u["tier"] for u in units
            if "/response" in u["unit"]}
    assert trace == resp  # parity: every trace unit's response twin sits in the same tier
    ladder = [u for u in units if u["unit"].startswith("canned/")]
    assert {u["tier"] for u in ladder if P.in_ladder_subset(int(u["sid"][-3:]))} == {1}
    assert {u["tier"] for u in ladder if not P.in_ladder_subset(int(u["sid"][-3:]))} == {4}


def test_judge_order_puts_the_ladder_first_within_a_tier(monkeypatch):
    # most probable failure: a deadline inside T1 drops the ladder (it was queued last), which
    # leaves G1 with no calibration for any match verdict
    monkeypatch.setattr(P, "render_ids", lambda *a, **k: [])
    items = _items()
    units = P.judge_units(items, _gens(items), validate=False)
    jobs = [j["unit"] for j in P.judge_jobs(units, items, None, None, 0, 1, {1, 2})]
    n_canned = sum(u.startswith("canned/") for u in jobs)
    assert n_canned and all(u.startswith("canned/") for u in jobs[:n_canned])
    tiers = [int(j["tier"]) for j in P.judge_jobs(units, items, None, None, 0, 1, {1, 2})]
    assert tiers == sorted(tiers)  # tier order still governs across tiers


def test_projection_takes_the_first_level_that_fits_or_stops():
    # most probable failure: a cut chosen out of the registered order, or a run that overruns
    calls = {1: 1000, 2: 400, 3: 400, 4: 300}
    m = {"gen_wall_h_full": 0.5, "setup_h": 0.5, "margin_h": 0.25,
         "judge_calls_per_s": 2000 / 3600}  # 2000 calls/h for the two-GPU judge engine
    r = P.project(m, calls, envelope_h=8.0, validate_h=0.5)
    assert [x["cut"] for x in r["levels"]] == ["none", "a", "a+b", "a+b+c"]
    assert r["decision"] == "none"  # 2 x (0.5 + 0.5 + 1.05 + 0.25) = 4.6 <= 7.5
    slow = {**m, "judge_calls_per_s": 600 / 3600}
    assert P.project(slow, calls, 8.0, 0.5)["decision"] == "a+b"
    assert P.project({**m, "judge_calls_per_s": 100 / 3600}, calls, 8.0, 0.5)[
        "decision"] == "STOP"


def test_verdicts_record_both_credit_tests():
    # most probable failure: the exact-token test accepting a sentence, or the substring test
    # (of record) silently replaced by the exact one
    assert P.exact_verdict("Yes.") == "yes" and P.exact_verdict("yes, it does not") is None
    assert P.credits("Yes, it does not.", -1) == {"credit_substring": True, "credit_exact": False}
    assert P.credits(None, 2) == {"credit_substring": False, "credit_exact": False}


def _cached() -> bool:
    try:
        from transformers import AutoTokenizer
        for m in P.GEN.values():
            AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], local_files_only=True)
        from deepsteer.datasets.moralbench import default_cache_dir as mb_cache
        return default_cache_dir().exists() and mb_cache().exists()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cached(), reason="needs both tokenizers, MoReBench and MoralBench cached")
def test_dry_validate_sequence_keeps_text_out_of_tracked_rows(tmp_path):
    # most probable failure: a tracked row carrying generation or judge text (it can quote the
    # scenario; MRB_PLAN §9), or the projection reading a summary the run never wrote
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    run = [sys.executable, str(REPO / "papers/mrb_morebench/scripts/mrb_g1_pod.py")]
    for args in (["generate", "--model", "gpt_oss_20b"], ["generate", "--model", "qwen3_30b"],
                 ["judge", "--shard", "0/1", "--tiers", "T1"], ["project"], ["mb-rider"]):
        rc = subprocess.run(run + args + ["--out", str(tmp_path), "--validate", "--dry-run"],
                            capture_output=True, text=True, env=env, cwd=REPO)
        assert rc.returncode == 0, (args, rc.stderr[-2000:])
    gen = [json.loads(x) for x in (tmp_path / "gen/gpt_oss_20b.jsonl").read_text().splitlines()]
    assert len(gen) == 16 * P.K and all(r["final_reached"] for r in gen)
    assert not any(k in r for r in gen for k in ("trace", "response", "text"))
    for m in ("gpt_oss_20b", "qwen3_30b"):
        # most probable failure (the third VALIDATE): an empty GPT-OSS trace, because the model
        # opens the analysis channel itself and think_io's split assumes the prompt did
        raw = [json.loads(x) for x in (tmp_path / f"_raw/gen/{m}.jsonl").read_text().splitlines()]
        assert all(r["trace"] and r["response"] for r in raw), m
    tracked = [json.loads(x) for x in (tmp_path / "gen/gpt_oss_20b.jsonl").read_text().splitlines()]
    assert all(r["trace_chars"] > 0 for r in tracked)
    judge = [json.loads(x) for x in (tmp_path / "judge/shard0.jsonl").read_text().splitlines()]
    # the stub judge always answers "yes": credit exactly on the positive-weight criteria
    assert judge and all(r["verdict"] == "yes" for r in judge)
    assert all(r["credit_substring"] == (r["weight"] > 0) for r in judge)
    assert not any("reply" in r for r in judge)
    proj = json.loads((tmp_path / "projection.json").read_text())
    assert proj["decision"] in ("none", "a", "a+b", "a+b+c", "STOP")
    assert len((tmp_path / "mb_phase2/gpt_oss_20b/r2.jsonl").read_text().splitlines()) == 4


def test_session_reads_the_projection_and_honours_stop():
    # most probable failure: the session judges every tier by default (no projection read) or
    # runs on a STOP decision; the runner cannot run locally (vLLM, CUDA), so read it
    text = (REPO / "papers/mrb_morebench/runpod/remote_mrb_g1.sh").read_text()
    assert 'PROJ="$OUT_ROOT/g1_validate/projection.json"' in text
    assert '[ "$DECISION" = "STOP" ] && { echo "BAIL' in text
    assert '--tiers "$TIERS"' in text and "--tiers T1,T2,T3,T4" not in text
    assert "gpt-oss-120b did not judge on this GPU type" in text  # the envelope condition
    # most probable failure of the first MRB VALIDATE: flashinfer's JIT found no ninja because the
    # venv's bin was not on the engine processes' PATH
    assert 'export PATH="$VENV/bin:$PATH"' in text and 'shutil.which("ninja")' in text
    assert 'judge --shard 0/1 --tiers "$TIERS"' in text and "shard 1/2" not in text
    # most probable failure of the second: flashinfer's sampler JIT build under the image's old
    # nvcc; the sampler is PyTorch's, and a tiny-model smoke at top-p 0.95 runs before downloads
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in text
    # MRB_PLAN v0.4: under T1 only, each model generates k = 2 (its k 2-3 would never be judged)
    assert 'case ",$TIERS," in *,T2,*) K20=4 ;; esac' in text and '--k "$KQW"' in text
    assert text.index("vLLM smoke OK") < text.index("snapshot_download")
    pin = "5176ab746f4e736c4a354a11d0620abf0a941de287cad937ac34039293ca6f5f"  # ST's vLLM wheel
    assert f"VLLM_WHEEL_SHA256={pin}" in text


def test_amendment4_reprojection_prices_real_traces_and_caps_at_the_envelope(tmp_path,
                                                                              monkeypatch):
    # most probable failure: GPT-OSS trace units priced at VALIDATE's empty-trace rate (MRB-A8),
    # or a watchdog cap of 1.15 x the projection that runs past the envelope's remainder
    import mrb_g1_reproject as R
    items = _items()
    monkeypatch.setattr(P, "_items", lambda: items)
    (tmp_path / "gen").mkdir()
    (tmp_path / "judge").mkdir()
    for m in P.GEN:
        (tmp_path / "gen" / f"{m}.summary.json").write_text(json.dumps(
            {"jobs_run": 10, "decoded_tokens": 20000, "tokens_per_s": 2000.0}))
    rows = [{"unit": u, "pass": 1, "judge": "default", "n_tokens": n}
            for u, n in (("gpt_oss_20b/default/mrb-000/k0/trace", 10),
                         ("gpt_oss_20b/default/mrb-000/k0/response", 400),
                         ("qwen3_30b/default/mrb-000/k0/trace", 550),
                         ("qwen3_30b/default/mrb-000/k0/response", 400),
                         ("canned/empty/mrb-000", 350))]
    (tmp_path / "judge" / "shard0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "judge" / "shard0.summary.json").write_text(json.dumps({"tokens_per_s": 1.0}))
    (tmp_path / "projection.json").write_text(json.dumps(
        {"measured": {"setup_h": 0.2, "margin_h": 0.25}}))
    r = R.reproject(tmp_path, envelope_h=1e6, validate_h=0.0)
    assert r["priced_tokens_per_call_by_type"]["gpt_oss_20b/trace"] == 550
    assert [lv["k"]["qwen3_30b"] for lv in r["levels"]] == [4, 4, 2, 2]
    assert [lv["k"]["gpt_oss_20b"] for lv in r["levels"]] == [4, 4, 4, 2]
    t1 = r["levels"][-1]
    wall = t1["session_wall_h"]
    tight = R.reproject(tmp_path, envelope_h=2 * wall * 1.05, validate_h=0.0)
    assert tight["decision"] == "a+b+c" and tight["session_max_wall_h"] == round(wall * 1.05, 3)
