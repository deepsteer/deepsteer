"""MRB-G2 pod driver (papers/mrb_morebench/scripts/mrb_g2_pod.py): theory prompt, units, dry run."""

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
import mrb_g2_pod as G  # noqa: E402

THEORIES = ("Act Utilitarianism", "Scanlonian Contractualism", "Aristotelian Virtue Ethics",
            "Kantian Deontology", "Gauthierian Contractarianism")


def _items(n=20):
    crit = tuple(MoReBenchCriterion(f"c{j}", f"criterion {j}", 2, "identifying") for j in range(3))
    return [MoReBenchItem(f"mrbt-{i:03d}", i, f"dilemma {i}", "daily_dilemmas", "long_case",
                          "ai_agent", THEORIES[i % 5], "ctx", crit) for i in range(n)]


def test_generation_uses_the_theory_prompt_for_each_rows_own_theory(monkeypatch):
    # most probable failure: the public set's neutral prompt reused for theory rows, or one
    # theory's definition sent with another row's dilemma
    seen = []
    monkeypatch.setattr(P, "render_ids", lambda tok, text, *a, **k: seen.append(text) or [0])
    jobs = G.gen_jobs(_items(), None, validate=False)
    assert len(jobs) == 20 * G.K and len(seen) == 20
    for it, text in zip(_items(), seen):
        assert text.startswith("Provide corresponding reasoning and decision for the following "
                               f"scenario based on {it.theory}.")
        assert text.endswith(f"Scenario:{it.dilemma}")
    assert {j["sampler"]["temperature"] for j in jobs} == {0.6}  # Table 4, Qwen3-Thinking
    assert len(G.gen_jobs(_items(), None, validate=True)) == 2 * G.K  # rows 0 and 10


def test_judge_units_put_the_ladder_first_and_cover_both_fields():
    # most probable failure: a deadline costing the ladder (queued last), a field left unjudged,
    # or a rung outside the two v0.7 registers
    items = _items()
    gens = {f"qwen3_30b/default/{it.scenario_id}/k{k}": {"trace": "t", "response": "r"}
            for it in items for k in range(G.K)}
    units = G.judge_units(items, gens, validate=False)
    n_canned = sum(u["unit"].startswith("canned/") for u in units)
    assert n_canned == 2 * len(items) and all(u["unit"].startswith("canned/")
                                              for u in units[:n_canned])
    assert {u["unit"].split("/")[1] for u in units[:n_canned]} == set(G.LADDER)
    rest = units[n_canned:]
    assert len(rest) == len(items) * G.K * 2
    idx = [int(u["sid"][-3:]) for u in rest]
    assert idx == sorted(idx)  # row by row: a deadline costs the last rows


def test_projection_proceeds_only_inside_the_envelope_and_caps_there():
    # most probable failure: a session allowed to run past the 3 A100-h envelope (here 2.5 of
    # the 2.6 left proceeds; the same session against 1.6 left stops)
    m = {"setup_h": 0.25, "gen_wall_h_full": 0.2, "judge_load_h": 0.05, "judge_wall_h_full": 0.5,
         "margin_h": 0.25}
    ok = G.project(m, envelope_h=3.0, validate_h=0.4)
    assert ok["decision"] == "proceed" and ok["session_max_wall_h"] <= (3.0 - 0.4) / 2
    assert G.project(m, envelope_h=2.0, validate_h=0.4)["decision"] == "STOP"


def _cached() -> bool:
    try:
        from transformers import AutoTokenizer
        for m in P.GEN.values():
            AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], local_files_only=True)
        return default_cache_dir().exists()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cached(), reason="needs both tokenizers and MoReBench cached")
def test_dry_validate_sequence_keeps_text_out_of_tracked_rows(tmp_path):
    # most probable failure: a tracked row carrying generation or judge text, or the projection
    # reading a summary the run never wrote
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    run = [sys.executable, str(REPO / "papers/mrb_morebench/scripts/mrb_g2_pod.py")]
    for args in (["generate"], ["judge"], ["project"]):
        rc = subprocess.run(run + args + ["--out", str(tmp_path), "--validate", "--dry-run"],
                            capture_output=True, text=True, env=env, cwd=REPO)
        assert rc.returncode == 0, (args, rc.stderr[-2000:])
    gen = [json.loads(x) for x in (tmp_path / "gen/qwen3_30b.jsonl").read_text().splitlines()]
    assert len(gen) == 15 * G.K and all(r["final_reached"] and r["trace_chars"] for r in gen)
    assert not any(k in r for r in gen for k in ("trace", "response"))
    judge = [json.loads(x) for x in (tmp_path / "judge/shard0.jsonl").read_text().splitlines()]
    assert judge and not any("reply" in r for r in judge)
    assert {r["unit"].split("/")[-1] for r in judge if not r["unit"].startswith("canned/")} == \
        {"trace", "response"}
    proj = json.loads((tmp_path / "projection.json").read_text())
    assert proj["decision"] in ("proceed", "STOP") and "no_verdict_share" in proj["measured"]
    # the analysis runs on what VALIDATE wrote, and its paper-table audit holds on v2's rows
    import mrb_g2_analyze as Z
    res = Z.main_from(tmp_path)
    assert res["metadata"]["units_incomplete"] == 0
    assert all(v["consistent"] for v in res["paper_table_audit"].values())
    assert set(res["fields"]) == {"trace", "response"} and set(res["ladder"]["credit_substring"]) \
        == set(G.LADDER)


def test_runner_reads_g2s_projection_and_runs_only_g2s_models():
    # most probable failure: the G2 runner left pointing at G1's projection or driver, so a
    # session would judge G1's tiers or download GPT-OSS-20B's weights
    text = (REPO / "papers/mrb_morebench/runpod/remote_mrb_g2.sh").read_text()
    assert 'PROJ="$OUT_ROOT/g2_validate/projection.json"' in text
    assert '[ "$DECISION" = "STOP" ] && { echo "BAIL' in text
    assert "mrb_g2_pod.py" in text and "mrb_g1_pod.py" not in text
    assert "openai/gpt-oss-20b" not in text and "fetch_moralbench" not in text
    assert 'export PATH="$VENV/bin:$PATH"' in text
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in text
    assert text.index("vLLM smoke OK") < text.index("snapshot_download")
    assert "tests/scripts/test_mrb_g2_pod.py" in text
