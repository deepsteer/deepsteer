"""§13's evaluation pod and cell A′ (scripts/c13_driver.py, scripts/analyze_c13.py; Amendments 36
and 40): the registered sites, the inputs, d1's projection-fraction convention, position validity,
the positive control, the raters' reliability rule, cell A′'s branch, the runner and a dry run."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_c13 as A  # noqa: E402
import c13_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
INP = ST / "outputs" / "c13_inputs"


def test_registered_sites():
    # most probable failure (Amendment 40 item 1): the extraction reads another layer than d1's
    # headline, or a band outside the registered one
    assert D.SITES["gpt_oss_20b"]["headline"] == 12 and D.SITES["olmo3_think"]["headline"] == 16
    assert D.SITES["olmo3_think"]["band"] == tuple(range(15, 32))
    assert D.SITES["gpt_oss_20b"]["band"] == tuple(range(11, 24))
    assert D.BAND_P == (0.2, 0.6) and D.N_A2 == 60


NEEDED = [INP / f"{s}.json" for s in D.D1_SOURCES]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="the c13 inputs are not local")
def test_texts_are_d1s_clean_pairs_and_both_documents(tmp_path):
    # most probable failure: the extraction reads the full source files (unclean pairs too) or a
    # changed copy, so V_moral is not d1's
    ts = D.texts(INP, dry=False)
    assert len(ts) == 2 * (573 + 62 + 199) + 2 * (245 + 265) == 2688
    assert len({t["key"] for t in ts}) == len(ts)
    bad = tmp_path / "in"
    bad.mkdir()
    for p in NEEDED:
        (bad / p.name).write_bytes(p.read_bytes())
    (bad / NEEDED[0].name).write_bytes(NEEDED[0].read_bytes() + b" ")
    with pytest.raises(SystemExit):
        D.texts(bad, dry=False)


def test_projection_fraction_is_d1s_norm_fraction():
    # most probable failure: the squared membership is used, so the overlap sits on a different
    # scale than d1's bands and nulls
    Q = A.ortho([np.eye(4)[0], np.eye(4)[1]])
    v = np.array([1.0, 0.0, 1.0, 0.0])
    assert A.frac(Q, v) == pytest.approx(1 / np.sqrt(2))


def _reads(valid=True, u_gpt=1.2, u_olmo=0.4):
    def model(u):
        return {"position_valid": valid,
                "documents": {"spec": {"u": u, "u_boot": list(np.full(A.N_BOOT, u)
                                                              + np.linspace(-0.1, 0.1, A.N_BOOT))}}}
    return {"gpt_oss_20b": model(u_gpt), "olmo3_think": model(u_olmo)}


def test_positive_control():
    # most probable failure: an invalid position (band below its null) still yields a verdict, or
    # the sign of the registered difference is reversed
    assert A.positive_control(_reads())["passes"]
    assert not A.positive_control(_reads(u_gpt=0.3, u_olmo=0.9))["passes"]
    assert A.positive_control(_reads(valid=False))["status"].startswith("uninformative")


def test_ladder_flags_a_band_below_the_null():
    # most probable failure: a degenerate position reads as "above the band floor"
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16))
    src = {s: rng.standard_normal(16) for s in A.SOURCES}
    lad = A.ladder(X, src, rng, 500)
    assert lad["null_q95"] > 0 and len(lad["heldout"]) == 3
    assert lad["band"][0] <= lad["band"][1]


def test_raters_reliability_rule(tmp_path, monkeypatch):
    # most probable failure: the subagent's labels are used although agreement is below 0.8, or the
    # author's letters are not decoded through the key
    key = {"key": {f"C13S-{i:03d}": {"A": "o1", "B": "o2", "C": "o3"} for i in range(1, 11)}}
    author = {f"C13S-{i:03d}": "B" for i in range(1, 11)}
    sub_hi = {f"C13S-{i:03d}": "o2" for i in range(1, 11)}
    sub_lo = {f"C13S-{i:03d}": ("o2" if i <= 5 else "o1") for i in range(1, 11)}
    monkeypatch.setattr(A, "MODELS", ())
    hi = A.family_read(tmp_path, author, key, sub_hi)
    lo = A.family_read(tmp_path, author, key, sub_lo)
    assert hi["reliability"]["agreement"] == 1.0 and hi["labels_of_record"].startswith("subagent")
    assert lo["reliability"]["agreement"] == 0.5 and lo["labels_of_record"].startswith("author")
    # Amendment 40 item 5: the construction label is read beside, and the author's 20 when the
    # subagent's labels are of record
    assert "construction_beside" in hi and "author20_beside" in hi
    assert "construction_beside" in lo and "author20_beside" not in lo
    assert lo["reliability"]["disagreements"] == [f"C13S-{i:03d}" for i in range(6, 11)]


def test_remote_runs_c13():
    # most probable failure: the GPT-OSS refusal-retention pass rides this pod, or cell A' arms run
    # on OLMo-3-Think
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "c13" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-gpt_oss_20b '
                     r'olmo3_think\}"; OUT_SUB=c13\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-2\.5\}"', sh)
    assert '[ "$STEPS" != "c13" ] && [ "$STEPS" != "c13c" ]; then  # Amendment 10' in sh
    assert '[ "$STEPS" != "c13" ] && [ "$STEPS" != "c13c" ]; then  # of record from T1.1' in sh
    blk = sh.split('if [ "$STEPS" = "c13" ]; then\n    rc=0')[1][:900]
    assert blk.index("--step base") < blk.index("--step cellA2_arm")
    assert '[ "$m" = "gpt_oss_20b" ]' in blk
    assert "test_c13.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = (ST / "runpod" / "sync_c13.txt").read_text().split()
    assert {f"outputs/c13_inputs/{s}.json" for s in D.D1_SOURCES} <= set(sync)
    assert all(not x.endswith("/") for x in sync)


@pytest.mark.slow
def test_dry_run_every_step(tmp_path):
    # most probable failure: a step crashes on a tiny model of either family, or the analysis
    # reads a verdict at an invalid position
    if not all(p.exists() for p in NEEDED):
        pytest.skip("the c13 inputs are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "c13_driver.py"), "--out", str(tmp_path), "--dry-run"]
    for m in D.SITES:
        r = subprocess.run(base + ["--step", "base", "--model", m, "--d1", str(INP)],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for arm in ("beh2", "beh"):
        r = subprocess.run(base + ["--step", "cellA2_arm", "--arm", arm, "--adapters",
                                   str(tmp_path / "adapters")],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads((tmp_path / "c13_record.json").read_text())["phases"]
    assert {"base_gpt_oss_20b", "base_olmo3_think", "cellA2_beh2", "cellA2_beh"} <= set(rec)
    res = {m: A.model_read(tmp_path, m) for m in D.SITES}
    for m, r in res.items():
        if not r["position_valid"]:
            assert r["documents"]["spec"]["ladder_position"].startswith("position invalid")
    a2 = A.cell_a2(tmp_path)
    assert a2["reading"] and a2["beh2"]["dP_violating"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_generic_screen_dry_run_skips_olmo3_think_on_c13():
    # most probable failure (the first c13 pod): the runner's generic per-model screen dry run reads
    # the registry's vLLM block, which OLMo-3-Think lacks, and stops the session before any read
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    blk = sh.split("dry_runs() {")[1][:600]
    assert ('if { [ "$STEPS" = "c13" ] || [ "$STEPS" = "c13c" ]; } && [ "$m" = "olmo3_think" ]; '
            'then') in blk
    import pod_st_screen as P
    assert P.registry()["olmo3_think"].get("vllm") is None


def test_launcher_forwards_session_max_hours():
    # most probable failure (the T1.6 resume): the launcher drops SESSION_MAX_HOURS, so the pod
    # arms its step default; an unset value forwards as empty and must still fall to that default
    sh = (ROOT / "papers" / "runpod_common" / "run_session.sh").read_text()
    assert "SESSION_MAX_HOURS='${SESSION_MAX_HOURS:-}'" in sh.split("setsid bash")[0]
    r = subprocess.run(["bash", "-c", 'SESSION_MAX_HOURS=""; echo "${SESSION_MAX_HOURS:-2.5}"'],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "2.5"


def test_standardization_removes_a_massive_dimension():
    # most probable failure (ST-A26 R_c, estimator-traps #9): the standardized read still lets one
    # input-constant outlier dimension carry the pooled sample, or top-k removes the wrong dimension
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 32))
    X[:, 5] *= 60.0
    assert A.top_share(X) > 0.8
    Xs = A.transform(X, "std")
    assert A.top_share(Xs) == pytest.approx(1 / 32, rel=1e-6)
    Xk = A.transform(X, "topk")
    assert Xk.shape[1] == 31 and A.top_share(Xk) < 0.1
    assert np.array_equal(A.transform(X, "raw"), X)


def test_invariance_and_branch42():
    # most probable failure: a standardized pass is read as R_c although the clean model's verdicts
    # changed under standardization, or an invalid GPT-OSS position yields a verdict
    def m(valid, pos):
        return {"position_valid": valid, "documents": {"spec": {"ladder_position": pos},
                                                       "anth": {"ladder_position": pos}}}
    same = A.invariance(m(True, "above the band floor"), m(True, "above the band floor"))
    moved = A.invariance(m(True, "above the band floor"), m(True, "within the null"))
    assert same["holds"] and not moved["holds"]
    assert A.branch42({"passes": True}, same, True).startswith("R_c")
    assert A.branch42({"passes": True}, moved, True).startswith("invariance fails")
    assert A.branch42({"passes": False}, same, True).startswith("fails with GPT-OSS")
    assert A.branch42({"passes": True}, same, False).startswith("GPT-OSS's position is invalid")


@pytest.mark.slow
def test_raw_mode_reproduces_the_read_of_record():
    # most probable failure: the transform refactor changes the raw read of record (Amendment 40)
    c13 = ST / "outputs" / "c13"
    if not all((c13 / m / "extract.npz").exists() for m in A.MODELS):
        pytest.skip("the c13 extracts are not local")
    rec = json.loads((ST / "data" / "st_c13_overlap_read_v1.json").read_text())
    for m in A.MODELS:
        r = A.model_read(c13, m, "raw")
        want = rec["models"][m] if "models" in rec else rec[m]
        assert r["null_q95"] == pytest.approx(want["null_q95"], abs=1e-9)
        assert r["documents"]["spec"]["pf"] == pytest.approx(want["documents"]["spec"]["pf"],
                                                             abs=1e-9)


def test_document_specific_direction_and_branch44():
    # most probable failure (Amendment 44): the shared component is not removed (the specific
    # direction still loads on the other document), or a failed specificity check reads as R_b
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(50)
    va, vb = shared + 0.3 * rng.standard_normal(50), shared + 0.3 * rng.standard_normal(50)
    sp = A.specific_dir(va, vb)
    assert abs(sp @ A.unit(vb)) < 1e-9 and np.linalg.norm(sp) == pytest.approx(1.0)
    assert A.auc(np.array([2.0, 3.0]), np.array([1.0, 2.0])) == pytest.approx(0.875)
    ok = {"gpt_oss_20b": True, "olmo3_think": True}
    bad = {"gpt_oss_20b": True, "olmo3_think": False}
    assert A.branch44(bad, {"passes": True, "valid_positions": True}).startswith("R_a, strong")
    assert A.branch44(ok, {"passes": True, "valid_positions": True}).startswith("R_a:")
    assert A.branch44(ok, {"passes": False, "valid_positions": True}).startswith("R_b")
    assert A.branch44(ok, {"passes": True, "valid_positions": False}).startswith("uninformative")


def test_amendment45_five_item_rule_and_marks(tmp_path):
    # most probable failure: the rater's letters are decoded through the sheet's key (20 items)
    # instead of the rater items' key (38), or the five-item rule counts the construction's option
    key = {"key": {i: {"A": "o1", "B": "o2", "C": "o3"} for i in A.DISPUTED}}
    from deepsteer.kdg.schema import load_scenarios
    st = {s.id: {o.option_id: o.norm_status for o in s.options}
          for d in ("spec", "anth") for s in load_scenarios(ST / "data" /
                                                            f"st_c13_family_{d}_v1.json")[0]}
    neutral_letter = {i: next(L for L, o in key["key"][i].items() if st[i][o] == "neutral")
                      for i in A.DISPUTED}
    all_neutral = {i: {"letter": neutral_letter[i]} for i in A.DISPUTED}
    r = A.amendment45(all_neutral, key["key"], key, {}, {}, None)
    assert r["five_item_reading"].startswith("sides with the author") and len(
        r["disputed_neutral"]) == 5
    none_neutral = {i: {"letter": next(L for L in "ABC" if L != neutral_letter[i])}
                    for i in A.DISPUTED}
    r0 = A.amendment45(none_neutral, key["key"], key, {}, {}, None)
    assert r0["five_item_reading"].startswith("sides with the construction")
    q = tmp_path / "q.md"
    q.write_text("# t\n\n## C13S-012\n\ntext\n\ndoc: escalate\n\n## C13A-036\n\ndoc: Depart\n")
    m = A.doc_marks(q)
    assert m == {"C13S-012": "escalate", "C13A-036": "depart"}
    rm = A.amendment45(all_neutral, key["key"], key, {}, {}, m)
    assert rm["construction_document_inconsistent"] == ["C13S-012"] and rm["missing"] == []


def test_api_rater_parse_and_blind_prompt():
    # most probable failure: a fenced or prefixed JSON reply is lost, or the prompt leaks the
    # construction label
    import c13_api_rater as R
    assert R.parse('```json\n{"letter": "B", "confidence": "high"}\n```')["letter"] == "B"
    assert R.parse('{"letter": "D"}') is None and R.parse("no json") is None
    item = {"id": "C13S-001", "document": "C_spec", "scenario": "S", "options": {"A": "a",
                                                                              "B": "b", "C": "c"}}
    msgs = R.prompt(item, "DOC")
    text = json.dumps(msgs)
    assert "DOC" in text and "construction" not in msgs[1]["content"]
    assert "right_option" not in text and "violating" not in text


def test_remote_runs_c13c():
    # most probable failure (Amendment 47): the rebuilt control runs without the anchor check, or
    # the GPT-OSS refusal-retention pass rides it
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "c13c" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-gpt_oss_20b '
                     r'olmo3_think\}"; OUT_SUB=c13c\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-1\.0\}"', sh)
    blk = sh.split('if [ "$STEPS" = "c13c" ]; then\n    rc=0')[1][:500]
    assert "--step spec2025" in blk and '--anchors "$C13C_ANCHORS"' in blk
    sync = (ST / "runpod" / "sync_c13c.txt").read_text().split()
    assert sync == ["outputs/c13c_inputs/gpt_oss_20b/anchors.npz",
                    "outputs/c13c_inputs/olmo3_think/anchors.npz"]


def test_transform_like_uses_the_saved_sample():
    # most probable failure: later texts are standardized with their own σ instead of the saved
    # pooled sample's, so they sit in other coordinates than the ladder
    rng = np.random.default_rng(1)
    X0 = rng.standard_normal((200, 8)) * np.arange(1, 9)
    Z = rng.standard_normal((10, 8))
    assert np.allclose(A.transform_like(Z, X0, "std"), Z / (X0.std(0) + 1e-12))
    assert np.array_equal(A.transform_like(Z, X0, "raw"), Z)


def test_branch47():
    # most probable failure: a failed specificity check reads as the soft premise not showing
    ok = {"gpt_oss_20b": True, "olmo3_think": True}
    assert A.branch47({"gpt_oss_20b": False, "olmo3_think": True},
                      {"passes": True, "valid_positions": True}).startswith("uninformative")
    assert A.branch47(ok, {"passes": True, "valid_positions": True}).startswith("calibrated")
    assert A.branch47(ok, {"passes": False, "valid_positions": True}).startswith(
        "the soft premise")


def test_spec2025_pairs_pass_the_builder_checks():
    # most probable failure (Amendment 47): the committed C_spec2025 pairs drift from section 13's
    # checks (a hard-constraint principle, a banned word, a weak neutral match)
    import build_c13_pairs as B
    P = json.loads((ST / "data" / "st_c13_P_spec2025_v1.json").read_text())["principles"]
    d = json.loads((ST / "data" / "st_c13_pairs_spec2025_v1.json").read_text())
    allowed = [p["id"] for p in P if p["kind"] != "hard_constraint"]
    assert B.check_pairs([{k: x[k] for k in ("pid", "register", "k", "m", "n")}
                          for x in d["pairs"]], allowed) == {}
    assert sum(x["split"] == "test" for x in d["pairs"]) == len(allowed)


@pytest.mark.slow
def test_spec2025_dry_run_both_families(tmp_path):
    # most probable failure: the spec2025 step crashes on a tiny model of either family
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for m in D.SITES:
        r = subprocess.run([sys.executable, str(SCRIPTS / "c13_driver.py"), "--step", "spec2025",
                            "--model", m, "--out", str(tmp_path), "--dry-run"],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        z = np.load(tmp_path / m / "extract_spec2025.npz")
        assert len(z["keys"]) == 8 and str(z["keys"][0]).startswith("c13|spec2025|0|")


def test_branch48():
    # most probable failure: both models' positive contrasts read as the premise (a document
    # property is not training)
    assert A.branch48([0.01, 0.2], [-0.1, 0.1]) == "the premise shows within GPT-OSS"
    assert A.branch48([0.01, 0.2], [0.02, 0.1]).startswith("a property of the documents")
    assert A.branch48([-0.24, -0.05], [-0.16, 0.02]).startswith("the premise does not show")
