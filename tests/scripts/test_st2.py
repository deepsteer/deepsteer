"""ST Phase 2 driver (scripts/st2_driver.py), Cell 1: Amendment 22's arms, fresh seeds, rider (a)'s
protect-ward reading, and a dry run on a tiny GPT-OSS with the real harmony tokenizer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import st2_driver as D  # noqa: E402
import t13_driver as T13  # noqa: E402

DATA = ROOT / "papers" / "st_live_reference" / "data"


def _d(lo: float) -> dict:
    return {"mean": lo + 0.05, "ci95": [lo, lo + 0.1]}


def test_protect_ward_needs_both_cis_above_zero_and_coherence():
    # most probable failure: a move that clears none but not its matched control (a generic
    # perturbation) or an incoherent arm is read as protect-ward
    ok = {"passes": True}
    assert D.protect_ward(_d(0.01), _d(0.02), ok)
    assert not D.protect_ward(_d(0.01), _d(-0.01), ok)
    assert not D.protect_ward(_d(-0.01), _d(0.02), ok)
    assert not D.protect_ward(_d(0.01), _d(0.02), {"passes": False})


def test_cell1_seeds_are_fresh_and_shared_across_arms():
    # most probable failure: Cell 1 reuses the T1.3 gate's seeds (the rows that selected these
    # arms), or the arms of one cell draw different seeds (ΔP unpaired)
    class S:
        def __init__(self, i):
            self.id = i
    jobs = [{"s": S(s), "k": k} for s in ("H1-A-00", "H1-B-05") for k in range(16)]
    st2 = D.seeds_for("st2c1", jobs)
    assert st2 == D.seeds_for("st2c1", jobs)
    assert not set(st2) & set(T13.seeds_for(jobs))


def test_cell1_interventions():
    # most probable failure: neg_a adds +a, or fill does not cycle its five draws by rollout
    rng = np.random.default_rng(0)
    inp = {k: rng.standard_normal(4).astype(np.float32) for k in ("anchor_axis", "care_harm",
                                                                    "V_harm")}
    inp["fill_unit"] = np.eye(5, 4, dtype=np.float32)
    c = {"inp": inp, "alpha": 2.0}
    jobs = [{"k": k} for k in range(7)]
    assert D.c1_intervention(c, "none", jobs) is None
    assert np.allclose(D.c1_intervention(c, "neg_V_harm", jobs).vector, -inp["V_harm"])
    neg = D.c1_intervention(c, "neg_a", jobs)
    assert np.allclose(neg.vector, -inp["anchor_axis"]) and neg.alpha == 2.0
    f = D.c1_intervention(c, "fill", jobs)
    assert np.allclose(f.vector[6], 2.0 * inp["fill_unit"][1]) and f.alpha == 1.0
    assert set(D.C1_ARMS) == {"none", "fill", "care_harm", "V_harm", "neg_a"}
    # the pod's site and dose are the gate of record's (the dry run swaps in onset)
    assert (D.CLS, D.L_STAR, D.ALPHA, D.R1) == ("clause_end", 13, 267.301171875, 16)


def test_engaged_set_is_t11s_train_selection():
    # most probable failure: the 45 include an H1_held scenario (T1.5's evaluation set) or miss
    # the T1.3 gate's 20
    e = json.loads((DATA / "st2_engaged_train_v1.json").read_text())
    split = json.loads((ROOT / "papers/st_live_reference/split_v1.json").read_text())
    gate = json.loads((DATA / "t13_draw_v1.json").read_text())["gate"]
    assert e["n"] == len(e["ids"]) == 45
    assert set(e["ids"]) <= set(split["H1_train"]) and not set(e["ids"]) & set(split["H1_held"])
    assert set(gate) <= set(e["ids"])


def test_c1_read_on_synthetic_rows(tmp_path):
    # most probable failure: the read pairs care_harm with V_harm the wrong way round, scores the
    # gate against fill's ΔP instead of the per-scenario difference, or accepts unpaired arms (a
    # random tiny model never parses, so the dry run cannot reach these numbers)
    g = tmp_path / "gen"
    g.mkdir()
    p = {"none": 0.2, "fill": 0.2, "care_harm": 0.5, "V_harm": 0.1, "neg_a": 0.25}
    for arm, share in p.items():
        rows = []
        for s in range(10):
            for k in range(10):
                prot = k < round(share * 10)
                rows.append({"scenario_id": f"S{s}", "rollout_key": f"st2c1|S{s}|{k}",
                             "norm_status": "violating" if prot else "consistent", "letter": "A",
                             "n_gen_tokens": 500, "trace_nll_unsteered": 1.0})
        (g / f"c1_{arm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = D.phase_c1_read({"out": tmp_path}, None, None, None, None)
    ch = res["arms"]["care_harm"]
    assert ch["dP_vs_none"]["mean"] == pytest.approx(0.3) and ch["passes_gate"]
    assert ch["protect_ward"]
    assert not res["arms"]["neg_a"]["passes_gate"]  # +0.05 vs none: under the +0.10 bar
    assert res["arms"]["V_harm"]["dP_vs_none"]["mean"] == pytest.approx(-0.1)
    assert res["care_beyond_harm"]["mean"] == pytest.approx(0.4)
    bad = [json.loads(x) for x in (g / "c1_neg_a.jsonl").open()][::-1]
    (g / "c1_neg_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(SystemExit):
        D.phase_c1_read({"out": tmp_path}, None, None, None, None)


@pytest.mark.slow
def test_dry_run_cell1_on_a_tiny_gpt_oss(tmp_path):
    # most probable failure: an arm only the pod reaches crashes, the steered arms never fire their
    # hook, or the read pairs arms whose rollout keys differ
    inp = tmp_path / "inputs"
    inp.mkdir()
    np.savez(inp / "st2_inputs.npz", anchor_axis=np.ones(4, np.float32))
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPTS / "st2_driver.py"), "--out", str(out), "--inputs", str(inp),
           "--dry-run", "--date-pin", "2026-10-21"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = out / "gpt_oss_20b"
    rec = json.loads((d / "st2_record.json").read_text())
    assert all(rec["phases"][p]["finished"] for p in D.CELL1)
    rows = {a: [json.loads(x) for x in (d / "gen" / f"c1_{a}.jsonl").open()] for a in D.C1_ARMS}
    keys = [r["rollout_key"] for r in rows["none"]]
    assert keys and all([r["rollout_key"] for r in v] == keys for v in rows.values())
    assert all(k.startswith("st2c1|") for k in keys)
    assert all(r["n_steered"] == 0 for r in rows["none"])
    for a in ("fill", "care_harm", "V_harm", "neg_a"):
        assert any(r["n_steered"] > 0 for r in rows[a]), a
    read = rec["phases"]["c1_read"]
    assert set(read["arms"]) == {"care_harm", "V_harm", "neg_a"} and "care_beyond_harm" in read
    # resume: a second invocation skips every finished phase
    r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                        env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r2.returncode == 0 and "finished earlier" in r2.stdout


def test_rule_twins_are_minimal_and_rule_free():
    # most probable failure: a twin that still states the written rule (its sentence or a clause
    # edit missed), a rewrite that adds a fact, or a twin built from a different scenario text
    import hashlib

    import build_st2_inputs as B
    import pod_st_screen as P

    tw = json.loads((DATA / "st2_rule_twins_v1.json").read_text())
    sc = {s["id"]: s for s in json.loads(P.H1_SET.read_text())["scenarios"]}
    split = json.loads((ROOT / "papers/st_live_reference/split_v1.json").read_text())
    assert sorted(tw["twins"]) == sorted(split["H1_train"])
    assert tw["counts"] == {"deleted": 103, "clause_deleted": 7, "rewritten": 22}
    for sid, v in tw["twins"].items():
        text, twin = sc[sid]["agent_text"], v["agent_text_twin"]
        assert hashlib.sha256(text.encode()).hexdigest() == v["sha256_agent_text"]
        assert not any(s in twin for s in v["agent_rule_spans"]) and "  " not in twin
        if v["twin_kind"] == "rewritten":
            assert not B.BANNED.search(v["replacement"])
            orig = set(B._words(" ".join(v["agent_rule_spans"])))
            assert all(w in B.FUNCTION or w in orig for w in B._words(v["replacement"])), sid
        else:  # the twin is the original with the spans and the recorded clause edits removed
            exp = text
            for s in v["agent_rule_spans"]:
                exp = exp.replace(s + " ", "") if s + " " in exp else exp.replace(" " + s, "")
            for old, new in v["clause_edits"]:
                exp = exp.replace(old, new)
            assert exp == twin, sid
        assert (v["twin_kind"] == "clause_deleted") == bool(v["clause_edits"])


@pytest.mark.slow
def test_dry_run_pod_b_on_a_tiny_gpt_oss(tmp_path):
    # most probable failure: the twin prefill reads a prompt that does not reproduce T1.1's saved
    # ids (a different date pin, effort or permutation), the two prefixes count different class
    # positions, the Qwen engine is asked for after GPT-OSS took the GPU, or an arm, the transport
    # control, a side read or the Qwen re-encode crashes on the pod
    t11 = ROOT / "papers/st_live_reference/outputs/t11/gpt_oss_20b"
    qscreen = ROOT / "papers/st_live_reference/outputs/t10/qwen38_27b/screen/h1_screen.jsonl"
    if not (t11 / "gen" / "h1_rollouts_read.jsonl").exists() or not qscreen.exists():
        pytest.skip("T1.1 / Qwen screen outputs not present locally")
    inp = tmp_path / "inputs"
    inp.mkdir()
    np.savez(inp / "st2_inputs.npz", anchor_axis=np.ones(4, np.float32))
    out = tmp_path / "out"
    base = [sys.executable, str(SCRIPTS / "st2_driver.py"), "--out", str(out), "--inputs",
            str(inp), "--dry-run", "--date-pin", "2026-10-21"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(base, capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = out / "gpt_oss_20b"
    r = subprocess.run(base + ["--phases", ",".join(D.POD_B), "--t11", str(t11), "--c1", str(d),
                               "--qwen-screen", str(qscreen)],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rec = json.loads((d / "st2_record.json").read_text())
    assert all(rec["phases"][p]["finished"] for p in D.POD_B)
    assert D.POD_B[0] == "c3_qwen" and "c3_qwen" in D.NO_MODEL  # the engine on an empty GPU
    assert r.stdout.index("st2 c3_qwen: done") < r.stdout.index("st2 c2_extract")
    ex = rec["phases"]["c2_extract"]
    assert ex["n_traces"] == 8 and ex["n_scenarios"] == 4  # 4 extraction scenarios x 2 rollouts
    dr = rec["phases"]["c2_direction"]
    assert np.isfinite(dr["split_half_cos"]) and dr["target_twin"]["clause_end"] is not None
    assert dr["random"]["target_twin"]["prompt"] is not None
    tn = [json.loads(x) for x in (d / "gen" / "c2_twin_none.jsonl").open()]
    assert tn and all(x["twin_kind"] in ("deleted", "clause_deleted", "rewritten") for x in tn)
    ab = [json.loads(x) for x in (d / "gen" / "c2_ablate.jsonl").open()]
    assert any(x["n_steered"] > 0 for x in ab)
    c1 = [json.loads(x)["rollout_key"] for x in (d / "gen" / "c1_none.jsonl").open()]
    assert {x["rollout_key"] for x in ab} <= set(c1)  # paired with Cell 1's seeds
    tr = [json.loads(x) for x in (d / "c2" / "transport.jsonl").open()]
    assert len(tr) == 2 * 2 * 4 and {x["condition"] for x in tr} == {
        "present", "present_ablated", "present_random", "twin"}
    assert rec["phases"]["c2_freegen"]["n"] >= 1
    assert rec["phases"]["c2_anchor"]["n"] == 4
    q = rec["phases"]
    assert q["c3_reencode"]["n_protect"] > 0 and q["c3_reencode"]["n_rule"] > 0
    assert q["c3_direction"]["n_scenarios"] >= 1
    nv = [json.loads(x)["rollout_key"] for x in (d / "gen" / "c1_neg_V_harm.jsonl").open()]
    assert nv and set(nv) <= set(c1)  # paired with Cell 1's none
    qa = [json.loads(x) for x in (d / "gen" / "c3_qwen_arm.jsonl").open()]
    assert qa and {x["rollout_key"] for x in qa} <= set(c1)
    assert any(x["n_steered"] > 0 for x in qa)


@pytest.mark.slow
def test_trace_positions_on_a_real_trace_with_the_real_format():
    # most probable failure: the dry runs patch the harmony trace opening, so on the pod the
    # clause_end positions could fall on the header or outside the analysis channel and V_rule
    # would read the wrong tokens
    t11 = ROOT / "papers/st_live_reference/outputs/t11/gpt_oss_20b/gen/h1_rollouts_read.jsonl"
    if not t11.exists():
        pytest.skip("T1.1 outputs not present locally")
    import kdg_harmony as kh
    import pod_st_screen as P
    import steering_gate as SG
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    spec = P.registry()[D.KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    c = {"fmt": build_format(D.KEY, tok, spec), "tok": tok}
    r = json.loads(t11.open().readline())
    gen = r["gen_ids"]
    pos = D.trace_positions(c, gen, 100, SG.punct_mask(tok))
    ce = [p - 100 for p in pos["clause_end"]]
    end = gen.index(kh.END) if kh.END in gen else len(gen)
    assert ce and min(ce) >= len(kh.ANALYSIS_OPEN) and max(ce) < end
    assert all(tok.decode([gen[i]]).strip() in ".,;:!?" or tok.decode([gen[i - 1]]).strip()
               in ".,;:!?" for i in ce[:20])


def test_remote_runs_pod_b_in_the_drivers_order():
    # most probable failure: the runner's phase list drifts from the driver's (a phase dropped, or
    # the Qwen engine moved after GPT-OSS loads), and Cell 3 is no longer the first cut (bail
    # order: GPT-OSS Cell 3 before Cell 2)
    import re

    sh = (ROOT / "papers/st_live_reference/runpod/remote_st_screen.sh").read_text()
    phases = re.search(r'ST2B_PHASES="([^"]*)"', sh).group(1).split(",")
    assert phases == D.POD_B
    assert max(phases.index(p) for p in D.CELL2) < min(phases.index(p) for p in D.CELL3[1:])
    # the author's -V_harm arm runs after Cell 2 and before Cell 3 (kept ahead of Cell 3 and Muse)
    i = phases.index("c1_neg_V_harm")
    assert max(phases.index(p) for p in D.CELL2) < i < min(phases.index(p) for p in D.CELL3[1:])
    sync = (ROOT / "papers/st_live_reference/runpod/sync_st2b.txt").read_text().split()
    assert "outputs/st2a/gpt_oss_20b/gen/c1_none.jsonl" in sync
