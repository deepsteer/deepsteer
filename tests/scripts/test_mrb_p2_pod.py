"""MRB Phase 2 pod driver (papers/mrb_morebench/scripts/mrb_p2_pod.py)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mrb_morebench" / "scripts"))

import mrb_p2_pod as Z  # noqa: E402


def _tok_cached() -> bool:
    try:
        for m in Z.MODELS:
            Z.tokenizer(m)
        return True
    except Exception:  # noqa: BLE001
        return False


def test_projection_cuts_the_model_that_binds_and_explains_the_rates():
    # most probable failure: the cut lands on the slower-rate model by reflex, when the model that
    # binds the wall is the one writing more tokens per job (the author's "slower Qwen fits while a
    # faster Muse overruns"), or the session runs past the envelope
    per = {"muse_glimmer_30b": {"tokens_per_job": 4000, "tokens_per_s": 674,
                                "judge_tokens_per_call": 420, "loads_h": 0.1},
           "qwen38_27b": {"tokens_per_job": 1000, "tokens_per_s": 600,
                          "judge_tokens_per_call": 420, "loads_h": 0.1}}
    fixed = {"setup_h": 0.25, "hf_phase_h": 0.4, "judge_load_h": 0.05, "margin_h": 0.25}
    # k = 2 for both: 12.7 A100-h; Muse at k = 1: 8.4; Qwen at k = 1 instead: 11.7
    r = Z.project(per, fixed, judge_tps=5382, envelope_h=10.0, validate_h=1.2)
    first = r["levels"][0]["models"]
    assert first["qwen38_27b"]["gen_wall_h_per_1000_jobs"] < first["muse_glimmer_30b"][
        "gen_wall_h_per_1000_jobs"]  # 0.46 h against 1.65 h: L / r, not r
    assert r["k"] == {"muse_glimmer_30b": 1, "qwen38_27b": 2} and r["decision"] == "proceed"
    assert r["session_max_wall_h"] <= (10.0 - 1.2) / 2
    both = Z.project(per, fixed, judge_tps=5382, envelope_h=9.0, validate_h=1.2)
    assert both["k"] == {"muse_glimmer_30b": 1, "qwen38_27b": 1}  # still over: the other model too
    tight = Z.project(per, fixed, judge_tps=5382, envelope_h=4.0, validate_h=1.2)
    assert tight["decision"] == "STOP" and tight["k"] is None


def test_tv_is_sts_statistic_on_two_letters_and_refuses_misaligned_rows():
    # most probable failure: the parity statistic compared rows in different orders (a silent pass)
    a = [{"set": "s", "item_id": "1", "order": "original", "p_A": 0.9},
         {"set": "s", "item_id": "2", "order": "original", "p_A": 0.2}]
    b = [{"set": "s", "item_id": "1", "order": "original", "p_A": 0.7},
         {"set": "s", "item_id": "2", "order": "original", "p_A": 0.6}]
    assert abs(Z.tv(a, b) - 0.3) < 1e-12
    with pytest.raises(AssertionError):
        Z.tv(a, b[::-1])


def test_judge_order_puts_every_first_generation_first():
    # most probable failure: a deadline costing some arm's k0 (the headline's first generation)
    class It:
        def __init__(self, s, i):
            self.scenario_id, self.idx = s, i
    items = [It(f"mrb-{i:03d}", i) for i in range(3)]
    gens = {f"{m}/{arm}/mrb-{i:03d}/k{k}": "t" for m in Z.MODELS for arm in Z.ARMS
            for i in range(3) for k in (0, 1)}
    ks = [u["k"] for u in Z.judge_units(items, gens)]
    assert ks == sorted(ks) and ks.count(0) == 12


@pytest.mark.skipif(not _tok_cached(), reason="needs both models' tokenizers cached")
def test_split_keeps_headers_and_turn_markers_out_of_both_fields():
    # most probable failure: Muse's plain-text answer header ("to=user") or its <|end|> leaking
    # into the judged response, or Qwen's </think> into the trace
    cases = {"qwen38_27b": ("I weigh it.</think>\n\nThe answer is to help.<|im_end|>",
                            ("I weigh it.", "The answer is to help.", True)),
             "muse_glimmer_30b": (" to=self<|message|>I weigh it.<|eom|><|start|>assistant "
                                  "to=user<|message|>The answer is to help.<|end|>",
                                  ("I weigh it.", "The answer is to help.", True))}
    for m, (text, want) in cases.items():
        tok = Z.tokenizer(m)
        assert Z.split(Z.fmt_for(m, tok), tok, tok(text, add_special_tokens=False)["input_ids"]) \
            == want, m
    tok = Z.tokenizer("muse_glimmer_30b")
    fmt = Z.fmt_for("muse_glimmer_30b", tok)
    assert Z.split(fmt, tok, tok(" to=self<|message|>never closes", add_special_tokens=False)[
        "input_ids"]) == ("never closes", "", False)


@pytest.mark.skipif(not _tok_cached(), reason="needs both models' tokenizers cached")
def test_generation_seeds_are_shared_across_arms():
    # most probable failure: the arm in the seed, which breaks the common-random-numbers pairing
    tok = Z.tokenizer("qwen38_27b")
    jobs = Z.gen_jobs("qwen38_27b", tok, Z.fmt_for("qwen38_27b", tok), 2, validate=True)
    assert len(jobs) == 32 and len({(j["sid"], j["k"]) for j in jobs}) == 32
    assert len({j["seed"] for j in jobs}) == 32
    assert all(j["sampler"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 0} for j in jobs)


def _dry_ok() -> bool:
    from deepsteer.datasets.morebench import default_cache_dir
    return _tok_cached() and default_cache_dir().exists()


@pytest.mark.skipif(not _dry_ok(), reason="needs tokenizers, MoReBench and MoralBench cached")
def test_dry_validate_sequence_runs_the_gates_and_keeps_text_out(tmp_path):
    # most probable failure: the merge gate or the parity gate never computed (a dry run that
    # passes by default), or a tracked row carrying generation or judge text
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    run = [sys.executable, str(REPO / "papers/mrb_morebench/scripts/mrb_p2_pod.py")]
    steps = [["hf-phase", "--model", "qwen38_27b"],
             ["generate", "--model", "qwen38_27b", "--arm", "none", "--k", "2"],
             ["generate", "--model", "qwen38_27b", "--arm", "adapter", "--k", "2"],
             ["judge"]]
    for args in steps:
        rc = subprocess.run(run + args + ["--out", str(tmp_path), "--validate", "--dry-run"],
                            capture_output=True, text=True, env=env, cwd=REPO)
        assert rc.returncode == 0, (args, rc.stderr[-2000:])
    hf = json.loads((tmp_path / "mb_phase2/qwen38_27b/hf_phase.json").read_text())
    g = hf["merge_gate"]
    assert g["saved"]["ok"] and g["saved"]["keys_identical"] and g["adapter_tv_vs_base"] > 0
    assert g["merged_tv_vs_attached"] < g["merged_tv_vs_base"]  # merged sits on the adapter
    assert hf["n_rows"] == 176
    s = json.loads((tmp_path / "gen/qwen38_27b.adapter.summary.json").read_text())
    assert s["status"] == "ok" and "parity_tv" in s and "band_tv" in s
    gen = [json.loads(x) for x in (tmp_path / "gen/qwen38_27b.none.jsonl").read_text().splitlines()]
    assert len(gen) == 32 and not any(k in r for r in gen for k in ("trace", "response"))
    judge = [json.loads(x) for x in (tmp_path / "judge/shard0.jsonl").read_text().splitlines()]
    assert judge and not any("reply" in r for r in judge)


def test_runner_reads_p2s_projection_and_ships_only_registered_adapters():
    # most probable failure: the runner left pointing at G1/G2's projection, ignoring the per-model
    # k, or running with an adapter that is not the registered one
    text = (REPO / "papers/mrb_morebench/runpod/remote_mrb_p2.sh").read_text()
    assert 'PROJ="$OUT_ROOT/p2_validate/projection.json"' in text
    assert '[ "$DECISION" = "STOP" ] && { echo "BAIL' in text
    assert "['k']['muse_glimmer_30b']" in text and "['k']['qwen38_27b']" in text
    assert '"peft==0.21.2"' in text and "an adapter is missing or not the registered one" in text
    assert 'rm -rf "$OUT_ROOT/${SUB}_merged"' in text
    assert text.index("vLLM smoke OK") < text.index("snapshot_download")
    assert "tests/scripts/test_mrb_p2_pod.py" in text
    sync = (REPO / "papers/mrb_morebench/runpod/sync_p2.txt").read_text().split()
    assert "outputs/p2_validate/projection.json" in sync
    assert {Path(p).parts[3] for p in sync if "adapters" in p} == set(Z.MODELS)
