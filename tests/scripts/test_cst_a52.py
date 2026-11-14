"""CST Amendment 52's pod: papers/cst_constitutional_steering/scripts/cst_a52_pod.py, a52_read.py,
rate_watch.py and runpod/remote_cst_a52.sh."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"),
                str(ROOT / "papers/mch_machiavelli/scripts"), str(ROOT / "tests/scripts")]

import cst_a52_pod as P  # noqa: E402
import rate_watch as RW  # noqa: E402

RUNPOD = ROOT / "papers/cst_constitutional_steering/runpod"


def test_scaled_copy_scales_the_update_linearly(tmp_path):
    # most probable failure (Amendment 52 item 3d): both LoRA factors are scaled, so the served
    # update is s squared of the original, or the A matrix is scaled and B left alone
    import torch
    from safetensors.torch import load_file, save_file

    src = tmp_path / "src"
    src.mkdir()
    a, b = torch.randn(4, 8), torch.randn(8, 4)
    save_file({"m.lora_A.weight": a, "m.lora_B.weight": b}, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text("{}")
    info = P.scale_adapter(src, tmp_path / "dst", 0.25)
    w = load_file(str(tmp_path / "dst" / "adapter_model.safetensors"))
    assert info["scaled_tensors"] == 1 and torch.equal(w["m.lora_A.weight"], a)
    assert torch.allclose(w["m.lora_B.weight"] @ w["m.lora_A.weight"], 0.25 * (b @ a), atol=1e-6)


def test_reproduction_passes_only_when_the_ci_holds_the_whole_runpod_range(monkeypatch, tmp_path):
    # most probable failure (Amendment 51 item D5): the validation passes when the CI touches the
    # RunPod range at one end, or when only the point lies inside it
    import sdf_reread_quick as Q

    for ci, ok in (([-0.30, -0.26], True), ([-0.28, -0.20], False), ([-0.35, -0.285], False)):
        monkeypatch.setattr(Q, "paired", lambda a, b, ids, ci=ci: {"delta": -0.28, "ci95": ci})
        monkeypatch.setattr(P, "reproduction", P.reproduction)
        import s0a_analysis as A
        import sdf2_quick as Z

        monkeypatch.setattr(A, "labels", lambda: {})
        monkeypatch.setattr(Z, "cell_rows", lambda *a, **k: [])
        assert P.reproduction(tmp_path)["passes"] is ok


def test_rate_window_counts_one_bar_only_and_flags_below_70_percent():
    # most probable failure (Amendment 52 item 4): an engine load between two bars reads as a slow
    # window and raises a false flag, or a real slowdown inside one bar is not flagged
    a = RW.last_bar("Processed prompts:  3%| | 200/6792 [05:00<1:26:12, 1.27it/s, est. speed "
                    "input: 1127.24 toks/s, output: 600.00 toks/s]")
    b = RW.last_bar("x\rProcessed prompts:  9%| | 600/6792 [10:00<1:20:00, 1.30it/s, est. speed "
                    "input: 1100.00 toks/s, output: 400.00 toks/s]")
    assert a["elapsed_s"] == 300 and abs(RW.window(a, b, 300) - 200.0) < 1e-9
    other = dict(b, total=916)  # a new bar (another stage or chunk): no window
    assert RW.window(a, other, 300) is None
    long = RW.last_bar("Processed prompts: | 10/20 [1:05:00<00:10, est. speed input: 1 toks/s, "
                       "output: 2.00 toks/s]")
    assert long["elapsed_s"] == 3900


def test_runner_stops_on_a_failed_validation_before_any_registered_read():
    # most probable failure (Amendment 52 item 3a): a failed validation lets the registered reads
    # run, or the stages run out of the registered order so the cut takes the wrong one
    run = (RUNPOD / "remote_cst_a52.sh").read_text()
    order = [run.index(s) for s in ("--stage smoke_serve", "--stage validate",
                                     "fits a52_a0ngood", "fits a49_pctx", "fits sweep",
                                     "--stage sweep_prep")]
    assert order == sorted(order)
    i = run.index("--stage validate")
    stop = "the P-ctx reproduction); no registered read runs"
    assert stop in run[i:run.index("fits a52_a0ngood")]
    body = re.sub(r"(<<'EOF'[^\n]*\n).*?\nEOF\n", r"\1", run, flags=re.S)
    used = set(re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", body))
    set_here = set(re.findall(r"(?:^|[\s;(])(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", body, re.M))
    set_here |= set(re.findall(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b", body))
    env = {"HOME", "PATH", "REPO_DIR", "OMP_NUM_THREADS", "CST_SESSION", "CST_PYTHON", "HF_TOKEN"}
    assert not {v for v in used - set_here - env if not v.isdigit()}


def test_3b_reads_the_registered_branches(tmp_path, monkeypatch):
    # most probable failure (Amendment 52 item 3b): A0n with GOOD_BEHAVIOR equal to the base reads
    # as "content", or one equal to GOOD_BEHAVIOR reads as "dilution"
    import a49_read as R49
    import a52_read as R
    import teacher_read as TR

    games = [f"g{i}" for i in range(6)]
    sids = [f"{g}|0|{s}" for g in games for s in range(3)]
    lab = [{"game": g, "episode": 0, "step": s, "violating": [1, 0],
            "labels": [{c: 1 for c in TR.CATS}, {c: 0 for c in TR.CATS}], "saved_violating": 1,
            "saved_action": 0, "eligible": True} for g in games for s in range(3)]
    lf = tmp_path / "labels.jsonl"
    lf.write_text("".join(json.dumps(x) + "\n" for x in lab))
    monkeypatch.setattr(TR, "LABELS", lf)
    st = tmp_path / "states.json"
    st.write_text(json.dumps({"nested": sids, "wide": sids}))
    monkeypatch.setattr(R49, "STATES", st)
    monkeypatch.setattr(R49, "GB_NESTED", -0.5)
    teacher = tmp_path / "teacher_v1"
    (teacher / "teacher").mkdir(parents=True)
    monkeypatch.setattr(R49, "TEACHER", teacher)

    def rows(arm, acts):
        return [{"arm": arm, "sid": s, "k": k, "action": acts(i), "fallback": False,
                 "cites": False} for i, s in enumerate(sids) for k in range(2)]

    (teacher / "teacher" / "base_good_behavior.jsonl").write_text("".join(
        json.dumps(x) + "\n" for x in rows("base", lambda i: 0)
        + rows("good_behavior", lambda i: i % 2)))
    out = tmp_path / "a52"
    (out / "teacher").mkdir(parents=True)
    f = out / "teacher" / "a52_a0ngood.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in rows("a0n_good", lambda i: 0)))
    assert R.a0n_good(out)["reading"] == "dilution"
    f.write_text("".join(json.dumps(x) + "\n" for x in rows("a0n_good", lambda i: i % 2)))
    assert R.a0n_good(out)["reading"] == "content"


@pytest.mark.slow
def test_dry_session_runs_every_a52_stage_in_order(tmp_path):
    # most probable failure (Amendment 28 item 3): a later stage breaks on the pod after hours;
    # every Amendment 52 stage runs on the tiny model in the runner's order
    import cst_finalist_pod as F
    import cst_teacher_pod as TP
    import pilot_pod_lib as PL

    from deepsteer.datasets import machiavelli as mch

    if not (TP.RUN / "decisions.jsonl").exists() or not (
            mch.default_cache_dir() / "src" / "game_data").exists():
        pytest.skip("the finalist run's decisions or the MACHIAVELLI bundle are not here")
    _, wb, _ = PL.load_model(True)
    adir = tmp_path / "adapters" / "p_ctx" / "1"
    wb.model.save_pretrained(str(adir))
    (tmp_path / "train_p_ctx.json").write_text(json.dumps({"trained": True,
                                                           "adapters": {"1": str(adir)}}))
    assert F.stage_smoke_hf(tmp_path, "p_ctx", True)["structural"][0]["passes"]
    assert PL.smoke_serve(tmp_path, True)["parity_passes"]
    F.stage_parity_hf(tmp_path, "p_ctx", True)
    assert "p_ctx" in P.stage_validate(tmp_path, True)["read"]["read"]
    TP.stage_cells(tmp_path, "a52_a0ngood", [("a0n_good", "nested", 2)], True)
    TP.stage_cells(tmp_path, "a49_pctx", [("p_ctx", "wide", 2), ("base", "nested", 2)], True,
                   tag="p_ctx")
    prep = P.stage_sweep_prep(tmp_path, True)
    assert all(c["structural"]["passes"] for c in prep["copies"].values())
    swept = P.stage_sweep(tmp_path, True)
    assert sorted(swept["read"]) == sorted(P.tag_of(s) for s in P.STRENGTHS)
