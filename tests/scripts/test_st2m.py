"""ST Phase 2 Cell 4, Muse's transfer test (scripts/st2m_driver.py, scripts/analyze_st2m.py;
Amendments 29-30): the pre-pod checks Amendment 30 item 7 requires, the block of record, the bail
and step-down rules, the branches, the runner's order, and a dry run on a tiny Muse."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_st2m as A  # noqa: E402
import st2m_driver as M  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
OUTS = ST / "outputs"


def _muse():
    import pod_st_screen as P
    import screen_lib as L
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    spec = P.registry()[M.KEY]
    try:
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    except OSError:
        pytest.skip("Muse's tokenizer is not available")
    fmt = build_format(M.KEY, tok, spec)
    c = {"spec": spec, "fmt": fmt, "out": Path("/nonexistent"), "layer": M.BLOCK + 1}
    return tok, fmt, M.screen(c, L.tok_only(tok), "2026-10-22", False)


def test_t_inst_lands_on_each_requests_last_token():
    # most probable failure (Amendment 30 item 7): Muse's template suffix is measured wrong, so
    # -V_harm is read at a template token instead of the request's last token
    tok, _, sc = _muse()
    ps = json.loads((ROOT / "papers/5_moral_alignment/refusal_prompts.json").read_text())
    texts = ps["harmful"][:64] + ps["harmless"][:64]
    ids, ti = M.last_token_index(sc, texts)
    for t, x, i in zip(texts, ids, ti):
        last = tok.decode([x[i]]).strip()
        assert last and t.rstrip().endswith(last), (t[-40:], last)
        assert tok.decode(x[i + 1:]) == "<|eot|><|start|>assistant"


def test_clause_end_detection_on_real_muse_traces():
    # most probable failure (Amendment 30 item 7): the dry runs patch the trace opening, so on the
    # pod clause_end could land outside Muse's to=self ... <|eom|> channel
    rows_p = OUTS / "t10" / "muse_glimmer_30b" / "screen" / "h1_screen.jsonl"
    if not rows_p.exists():
        pytest.skip("Muse's screen rows are not local")
    import st2_driver as S2
    import steering_gate as SG

    tok, fmt, _ = _muse()
    c, punct = {"fmt": fmt, "tok": tok}, SG.punct_mask(tok)
    ro, n = list(fmt.reason_open), 0
    for r in [json.loads(x) for x in rows_p.open()][:40]:
        g = r.get("gen_ids") or []
        if tuple(g[: len(ro)]) != tuple(ro):
            continue
        end = g.index(fmt.reason_end) if fmt.reason_end in g else len(g)
        for p in S2.trace_positions(c, g, 0, punct)["clause_end"]:
            assert len(ro) <= p < end and (punct[g[p]] or punct[g[p - 1]])
            n += 1
    assert n > 1000


def test_block_of_record_matches_gpt_oss_attention_type():
    # most probable failure: block 26 is a full-attention layer while GPT-OSS's block 12 is sliding
    # (Amendment 30 item 1 would then take the nearer matching neighbour)
    import pod_st_screen as P
    from transformers import AutoConfig

    reg = P.registry()
    try:
        g = AutoConfig.from_pretrained(reg["gpt_oss_20b"]["repo"],
                                       revision=reg["gpt_oss_20b"]["revision"])
        m = AutoConfig.from_pretrained(reg[M.KEY]["repo"], revision=reg[M.KEY]["revision"])
    except OSError:
        pytest.skip("configs not available")
    t = getattr(m, "text_config", m)
    assert g.layer_types[12] == t.layer_types[M.BLOCK] == "sliding_attention"
    assert M.BLOCK / t.num_hidden_layers == 12 / g.num_hidden_layers
    assert M.BANK_BLOCKS == (20, 26, 32)


def test_bail_order_and_step_down(tmp_path, monkeypatch):
    # most probable failure: the pilot's re-price cuts V_qwen before care_harm, or bails a spec
    # read; or an arm failing on parse (not length) is stepped down
    c = {"record": tmp_path / "r.json", "engaged": ["x"] * 45, "r": 8, "paper7_n": 64,
         "spec_ids": ["s"] * 60}
    c["record"].write_text(json.dumps({"phases": {
        "pilot": {"seconds_per_rollout": 10.0}, "bank": {"read_seconds_per_rollout": 0.3},
        "dose0": {"seconds_per_row": 0.1}}}))

    class Args:
        deadline = None
    import time

    for left_h, want in ((100, []), (5.0, ["arm_care_harm"]),
                         (4.0, ["arm_care_harm", "arm_neg_V_harm"]), (0.1, list(M.BAIL))):
        Args.deadline = time.time() + left_h * 3600
        got = M.reprice(c, Args)["bailed"]
        assert got == want[: len(got)] and set(got) <= set(M.BAIL), (left_h, got)
    assert M.BAIL == ("arm_care_harm", "arm_neg_V_harm", "arm_V_qwen")
    assert not any(p.startswith("spec_") for p in M.BAIL)
    assert M.needs_step_down({"passes": False, "length_ratio": 1.6})
    assert not M.needs_step_down({"passes": False, "length_ratio": 1.2})  # parse or NLL failure
    assert not M.needs_step_down({"passes": True, "length_ratio": 1.6})


def test_branches():
    # most probable failure (Amendment 30 item 6): a split is read as either clean branch
    def s(v, h):
        return {"V_qwen": {"passes": v}, "neg_V_harm": {"passes": h}}
    assert A.branch(s(False, False))["id"] == "both_fail"
    assert A.branch(s(True, True))["id"] == "both_pass"
    assert A.branch(s(True, False))["id"] == "split"
    assert A.branch({"V_qwen": {"passes": False}})["id"] is None
    # a stepped-down arm is read at its alpha / 2 specificity
    assert A.branch({**s(False, False), "V_qwen_half": {"passes": True}})["id"] == "split"
    assert len(A.PREREG12) == 12 and A.PREREG12[-1] == "muse_glimmer_30b:neg_V_harm"


def test_remote_runs_cell4_in_the_drivers_order():
    # most probable failure: the runner's phase list drifts from the driver's, a spec read runs
    # after an arm, or v1 shares the HF process (the vLLM engine needs the GPU first)
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    phases = re.search(r'ST2M_PHASES="([^"]*)"', sh).group(1).split(",")
    assert phases == M.HF_PHASES and "v1" not in phases
    first_arm = min(i for i, p in enumerate(phases) if p.startswith("arm_"))
    assert all(i < first_arm for i, p in enumerate(phases) if p.startswith("spec_"))
    assert phases[-3:] == list(reversed(M.BAIL))  # the last arm is the first bailed
    assert re.search(r'STEPS" = "st2m" \]; then\n(?:  #.*\n)*  MODELS=.*\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-6\.5\}"', sh)
    assert "--phases v1" in sh.split('if [ "$STEPS" = "st2m" ]; then\n    DEADLINE')[1][:400]
    sync = (ST / "runpod" / "sync_st2m.txt").read_text().split()
    assert {"outputs/st2b/gpt_oss_20b/gen/c3_qwen_topup.jsonl",
            "outputs/t10/qwen38_27b/screen/h1_screen.jsonl"} <= set(sync)


@pytest.mark.skipif(not (OUTS / "st2c" / "gpt_oss_20b").exists(), reason="pod rows not local")
def test_read_on_stand_in_rows(tmp_path):
    # most probable failure: the read takes the target from the gate alone, or Holm runs at m 11
    d = tmp_path / "m"
    (d / "gen").mkdir(parents=True)
    o = OUTS
    for src, dst in ((o / "st2a/gpt_oss_20b/gen/c1_none.jsonl", "arm_none"),
                     (o / "st2a/gpt_oss_20b/gen/c1_fill.jsonl", "arm_fill"),
                     (o / "st2b/gpt_oss_20b/gen/c1_neg_V_harm.jsonl", "arm_neg_V_harm"),
                     (o / "st2a/gpt_oss_20b/gen/c1_care_harm.jsonl", "arm_care_harm"),
                     (o / "st2c/gpt_oss_20b/gen/c3_qwen_half.jsonl", "arm_V_qwen")):
        rows = [json.loads(x) for x in src.open()]
        rows = [r for r in rows if int(r["rollout_key"].split("|")[2]) < 8]
        (d / "gen" / f"{dst}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for src, arm in ((o / "t13/gpt_oss_20b/panel", "none"),
                     (o / "st2dl/gpt_oss_20b/spec/fill", "fill"),
                     (o / "st2c/gpt_oss_20b/spec/V_qwen", "V_qwen"),
                     (o / "st2c/gpt_oss_20b/spec/neg_V_harm", "neg_V_harm"),
                     (o / "t13/gpt_oss_20b/panel", "care_harm")):
        (d / "spec" / arm).mkdir(parents=True)
        shutil.copy(src / "dl_chat_neutral.jsonl", d / "spec" / arm / "dl_chat_neutral.jsonl")
    (d / "st2m_record.json").write_text(json.dumps({"phases": {}}))
    ids = json.loads((ST / "data" / "t15_specificity_v1.json").read_text())["ids"]
    spec = A.spec_read(d, ids)
    assert spec["branch_rule"].startswith("S1")
    arms = A.arms_read(d, {})
    assert arms["V_qwen"]["passes_gate"] and not spec["V_qwen"]["passes"]
    assert A.branch(spec)["id"] == "both_fail"
    assert A.program(arms)["m"] == 12


@pytest.mark.slow
def test_dry_run_cell4_on_a_tiny_muse(tmp_path):
    # most probable failure: a phase crashes on Muse's VLM wrapper, a spec read runs at the wrong
    # alpha, or the arms do not share seeds
    qt = OUTS / "st2b" / "gpt_oss_20b" / "gen" / "c3_qwen_topup.jsonl"
    qs = OUTS / "t10" / "qwen38_27b" / "screen" / "h1_screen.jsonl"
    if not qt.exists() or not qs.exists():
        pytest.skip("Qwen3.8's rows not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "st2m_driver.py"), "--out", str(tmp_path), "--dry-run",
            "--date-pin", "2026-10-22"]
    r = subprocess.run(base + ["--phases", "v1"], capture_output=True, text=True, cwd=ROOT,
                       env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run(base + ["--qwen-topup", str(qt), "--qwen-screen", str(qs)],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = tmp_path / M.KEY
    rec = json.loads((d / "st2m_record.json").read_text())["phases"]
    assert all(rec[p]["finished"] for p in M.HF_PHASES)
    fin = {p: rec[p]["finished"] for p in M.HF_PHASES}
    assert max(fin[p] for p in fin if p.startswith("spec_")) <= min(
        fin[p] for p in fin if p.startswith("arm_"))
    al = rec["sweep"]["alpha"]
    assert al in rec["derive"]["alphas"]
    assert all(rec[f"spec_{x}"]["alpha"] == al for x in ("fill", *M.DIRS))
    seeds = {}
    for x in ("none", "fill", *M.DIRS):
        seeds[x] = [json.loads(y)["seed"] for y in (d / "gen" / f"arm_{x}.jsonl").open()]
    assert all(v == seeds["none"] for v in seeds.values())
    import numpy as np

    z = np.load(d / "directions.npz")
    assert all(abs(float(np.linalg.norm(z[k])) - 1) < 1e-4 for k in M.DIRS)


def test_arm_sets_stratified_superset_and_disjoint_sweep():
    # most probable failure (Amendment 30's addition): the 90 is not a superset of the 45 (the
    # largest-remainder Alabama paradox), the sweep overlaps an arm set, or the draw is not
    # reproducible or not stratified as split_v1
    import random
    from collections import Counter

    split = json.loads((ST / "split_v1.json").read_text())
    ids = sorted(split["H1_train"])
    random.Random(7).shuffle(ids)
    ids = sorted(ids[:110])  # a Muse-sized engaged set
    st = M.strata_of(ids)
    sweep, a45, a90 = M.arm_sets(ids, st)
    assert (sweep, a45, a90) == M.arm_sets(ids, st)
    assert len(sweep) == 5 and len(a45) == 45 and len(a90) == 90
    assert set(a45) <= set(a90) and not set(sweep) & set(a90)
    pool = Counter(st[s] for s in set(ids) - set(sweep))
    got = Counter(st[s] for s in a45)
    for k, n in pool.items():  # each stratum within one of its proportional share
        assert abs(got[k] - 45 * n / sum(pool.values())) < 1.0 + 1e-9, k
    small = ids[:30]
    sw, b45, b90 = M.arm_sets(small, M.strata_of(small))
    assert len(sw) == 5 and b45 == b90 == sorted(set(small) - set(sw))


def test_cost_rule_runs_90_only_when_every_arm_fits(tmp_path):
    # most probable failure: the 90 is chosen when only some arms fit, or bailing applies at 90
    import time

    c = {"record": tmp_path / "r.json", "engaged": ["x"] * 45, "arms_wide": ["x"] * 90, "r": 8,
         "paper7_n": 64, "spec_ids": ["s"] * 60}
    c["record"].write_text(json.dumps({"phases": {
        "pilot": {"seconds_per_rollout": 1.0}, "bank": {"read_seconds_per_rollout": 0.3},
        "dose0": {"seconds_per_row": 0.1}}}))

    class Args:
        deadline = time.time() + 6 * 3600
    r = M.reprice(c, Args)
    assert r["arm_n"] == 90 and r["bailed"] == []
    Args.deadline = time.time() + 1.5 * 3600  # 90 does not fit, 45 does
    r = M.reprice(c, Args)
    assert r["arm_n"] == 45 and r["bailed"] == []
    Args.deadline = time.time() + 0.3 * 3600
    r = M.reprice(c, Args)
    assert r["arm_n"] == 45 and r["bailed"][0] == "arm_care_harm"
