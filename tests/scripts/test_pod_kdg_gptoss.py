"""Local gates for the GPT-OSS-20B KDG session (KDG_GPTOSS_SPEC v0.2, G-A1..G-A5).

Each test names the single most probable failure mode it guards. Tests that need the pinned GPT-OSS
tokenizer skip when it can be neither loaded from the local cache nor fetched; the decision-token
equality test uses the cached tiny GPT-2 and skips without it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "papers" / "kdg_panel" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kdg_harmony as kh  # noqa: E402
import kdg_pod_lib as lib  # noqa: E402

from deepsteer.kdg.schema import assign_letters  # noqa: E402
from tests.kdg.conftest import make_scenario  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pod = _load("pod_kdg_gptoss")
_, REG = pod.registry()
SPEC = REG["gpt_oss_20b"]
CFG = kh.HarmonyConfig.from_registry(SPEC)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    for local in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                SPEC["repo"], revision=SPEC["revision"], local_files_only=local
            )
        except Exception:  # noqa: BLE001
            continue
    pytest.skip("pinned GPT-OSS tokenizer unavailable")


def _msgs(variant: str = "primary"):
    s = make_scenario()
    return lib.letter_chat_messages(s, assign_letters(s, 0), "agent", "neutral", variant)


class TestRegistry:
    def test_entry_is_pinned_and_matches_spec(self):
        # most probable failure: a branch name or a different template sha slips into the
        # registry, so the pod loads weights or a template the spec did not register
        assert SPEC["revision"] == "6cee5e81ee83917806bbde320786a8fb61efebee"
        assert SPEC["template_sha256"].startswith("a4c9919c")
        assert (SPEC["n_layers"], SPEC["hidden"]) == (24, 2880)
        assert CFG.reasoning_level == "medium" and CFG.reasoning_level_c0_generate == "low"
        assert CFG.date_pin == "Current date: 2026-10-04"

    def test_pinned_template_text_hashes_to_registry(self, tok):
        # most probable failure: the template loaded at the revision is not the registered one
        assert lib.sha256_text(tok.chat_template) == SPEC["template_sha256"]

    def test_letters_are_tokens_32_to_36(self, tok):
        # most probable failure: an option letter tokenizes to two ids, so its mass is misread
        assert [tok.encode(L, add_special_tokens=False) for L in "ABCDE"] == [[i] for i in
                                                                               range(32, 37)]


class TestDatePinAndCutoff:
    def test_cutoff_line_untouched_by_the_date_pin(self, tok):
        # G-A5 item 2. most probable failure: the pin regex also rewrites (or drops) the
        # 'Knowledge cutoff:' line, silently changing every prompt's system message
        raw = tok.apply_chat_template(_msgs(), tokenize=False, add_generation_prompt=True,
                                      reasoning_effort="medium")
        pinned = kh.render(tok, _msgs(), CFG, "medium")
        assert raw.count(CFG.cutoff_line) == 1 and pinned.count(CFG.cutoff_line) == 1
        cut = lambda t: [ln for ln in t.splitlines() if ln.startswith("Knowledge cutoff:")]  # noqa
        assert cut(raw) == cut(pinned) == [CFG.cutoff_line]
        # the only difference between the two renders is the date line
        diff = [(a, b) for a, b in zip(raw.splitlines(), pinned.splitlines()) if a != b]
        assert len(diff) == 1 and diff[0][1] == CFG.date_pin
        assert "Reasoning: medium\n" in pinned and pinned.endswith("<|start|>assistant")

    def test_pin_refuses_zero_or_two_date_lines(self):
        # most probable failure: a template change leaves no date line, and the pin no-ops
        base = f"x\n{CFG.cutoff_line}\nCurrent date: 2030-01-01\n"
        assert kh.pin_date(base, CFG).count(CFG.date_pin) == 1
        with pytest.raises(RuntimeError):
            kh.pin_date(f"x\n{CFG.cutoff_line}\n", CFG)
        with pytest.raises(RuntimeError):
            kh.pin_date(base + "Current date: 2031-01-01\n", CFG)

    def test_reasoning_level_is_rendered_not_defaulted(self, tok):
        # most probable failure: C0's low-effort render silently keeps 'Reasoning: medium'
        assert "Reasoning: low\n" in kh.render(tok, _msgs(), CFG, "low")

    def test_operator_prompt_lands_in_a_developer_turn(self, tok):
        # spec §3 C2 slot rival: the known-gap system prompt must not replace the model's own
        # system message (it renders as a developer turn beside it)
        r = kh.render(tok, _msgs("known_gap"), CFG, "medium")
        assert "<|start|>developer<|message|># Instructions" in r
        assert r.count("<|start|>system<|message|>You are ChatGPT") == 1


class TestDecisionTokenPosition:
    def test_no_prefill_token_precedes_the_residual_position(self, tok):
        # G-A5 item 1. most probable failure: the prefill tokenizes into the prompt's last token
        # (a merge across the boundary), so the saved 'decision-token' residual already sees it
        p = kh.render(tok, _msgs(), CFG, "medium")
        for name in ("primary", "direct_final"):
            a, b = kh.split_prefill(tok, p, CFG.prefill(name))
            assert a[-2:] == tok.encode("<|start|>assistant", add_special_tokens=False)
            assert b[0] == tok.convert_tokens_to_ids("<|channel|>")
            ids = tok.encode(p + CFG.prefill(name), add_special_tokens=False)
            assert ids[len(a) - 1] == a[-1] and ids[len(a):] == b

    def test_residual_identical_under_both_prefills_on_a_tiny_model(self, tok):
        # G-A5 item 1, causal half. most probable failure: the residual index is off by the
        # prefill length (read inside the prefill), so it differs between the two prefills; and,
        # second, left padding in a batch moves the read. A randomly initialised 2-layer GPT-OSS
        # (same architecture class, sliding + full attention) on the pinned tokenizer, no download.
        import types

        import torch
        from transformers import GptOssConfig, GptOssForCausalLM

        tok.padding_side = "left"
        cfg = GptOssConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=64,
                           num_local_experts=4, num_experts_per_tok=2, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=16, vocab_size=len(tok),
                           sliding_window=8, layer_types=["sliding_attention", "full_attention"])
        torch.manual_seed(0)
        m = types.SimpleNamespace(tok=tok, model=GptOssForCausalLM(cfg).eval(),
                                  device=torch.device("cpu"), torch=torch)
        read = lib.ModelWrapper.next_logprobs_and_residuals
        p = kh.render(tok, _msgs(), CFG, "medium")
        lp1, r1 = read(m, [p], [CFG.prefill_primary], 1)
        lp2, r2 = read(m, [p], [CFG.prefill_direct_final], 1)
        assert r1.shape == (1, 3, 64)
        np.testing.assert_array_equal(r1, r2)
        assert not np.array_equal(lp1, lp2)  # the readout itself does see the prefill
        longer = kh.render(tok, _msgs("known_gap"), CFG, "medium")  # pads p on the left
        _, r3 = read(m, [p, longer], [CFG.prefill_primary] * 2, 2)
        np.testing.assert_allclose(r3[0].astype(np.float32), r1[0].astype(np.float32), atol=2e-2)


class TestC0Rules:
    @pytest.mark.parametrize(
        "body,letter,method",
        [("B", "B", "letter_only"), ("**B**", "B", "letter_only"), ("B.", "B", "letter_only"),
         ("(C)", "C", "letter_only"), ("B. Because the deadline", "B", "leading_letter"),
         ("As the agent I pick B", None, "unparsed")],
    )
    def test_final_channel_parse(self, body, letter, method):
        # most probable failure: a markdown-bolded letter is read as unparsed, or 'As ...' as A
        fa = kh.parse_final(f"<|channel|>analysis<|message|>..<|end|><|start|>assistant"
                            f"{kh.FINAL_OPEN}{body}<|return|>", hit_budget=False)
        assert (fa.letter, fa.parse_method, fa.reasoning_trace) == (letter, method, "completed")

    def test_budget_hit_before_final_is_truncated_and_unread(self):
        # most probable failure: a truncated trace's stray letter is read as the answer
        fa = kh.parse_final("<|channel|>analysis<|message|>Option B looks", hit_budget=True)
        assert (fa.reasoning_trace, fa.letter) == ("truncated", None)

    def test_ties_count_as_non_agreement(self):
        # G-A1. most probable failure: max() breaks a 2-2 tie by order and counts it as a majority
        assert kh.strict_majority(["o1", "o1", "o2", "o2"]) is None
        assert kh.strict_majority(["o1", "o1", "o1", "o2"]) == "o1"
        assert kh.strict_majority(["o1", "o1", None]) == "o1"
        assert kh.strict_majority(["o1", None]) is None  # unparsed completed rollout counts
        assert kh.strict_majority([]) is None

    def test_wilson_and_near_miss_label(self):
        # G-A2. most probable failure: the near-miss label flips the verdict instead of riding it
        lo, hi = kh.wilson(52, 64)
        assert abs(lo - 0.7003) < 1e-3 and abs(hi - 0.8894) < 1e-3  # by hand, z = 1.96
        v = kh.c0_verdict([True] * 52 + [False] * 12)  # 0.8125, SE 0.049: pass, near-miss
        assert v["pass"] and v["near_miss"] and v["verdict"] == "pass (near-miss)"
        v = kh.c0_verdict([True] * 50 + [False] * 14)  # 0.781: fail, near-miss
        assert not v["pass"] and v["verdict"] == "fail (near-miss)"
        assert kh.c0_verdict([True] * 60 + [False] * 4)["verdict"] == "pass"


class TestDriver:
    def test_dry_run_saves_residuals_and_harmony_fields_in_every_letter_unit(self, tok, tmp_path):
        # G-A4 save list. most probable failure: a letter unit runs without save_residuals, so
        # the decision-token residuals the spec promises are silently absent
        rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_gptoss.py"), "--dry-run",
                             "--out", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
        assert rc.returncode == 0, rc.stderr[-2000:]
        d = tmp_path / "gpt_oss_20b"
        for u in pod.LETTER_UNITS:
            z = np.load(d / f"{u}.npz")
            assert "resid_decision" in z.files
            assert z["resid_decision"].shape[0] == len(z["option_mass"])
            row = json.loads((d / f"{u}.jsonl").read_text().splitlines()[0])
            assert row["harmony_date_pin"] == CFG.date_pin
            assert row["harmony_reasoning_level"] == "medium" and row["reasoning_trace"] == "none"
        g = [json.loads(x) for x in (d / "c0_generate_low.jsonl").read_text().splitlines()]
        assert {r["harmony_reasoning_level"] for r in g} == {"low"}
        assert all(r["reasoning_trace"] in ("completed", "truncated") for r in g)

    def test_timing_bail_stops_before_launch(self, tok, tmp_path):
        # G-A4 timing bail. most probable failure: the projection is printed but the exit code
        # stays 0, so the launcher proceeds past a too-long run
        rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_gptoss.py"), "--dry-run",
                             "--validate", "--max-hours", "0", "--out", str(tmp_path)],
                            capture_output=True, text=True, cwd=REPO)
        assert rc.returncode == 3, rc.stdout[-2000:] + rc.stderr[-2000:]
        assert json.loads((tmp_path / "timing.json").read_text())["stop"] is True

    def test_c0_sample_is_64_fixed_non_swap_scenarios(self):
        # most probable failure: the C0 sample drifts with file order or includes F4 swap cells
        files = sorted((REPO / "papers/kdg_panel/data").glob("*_scenarios_*.json"))
        from deepsteer.kdg.schema import load_scenario_dir

        S, _ = load_scenario_dir(files)
        a = [s.id for s in pod.c0_scenarios(S)]
        b = [s.id for s in pod.c0_scenarios(list(reversed(S)))]
        assert a == b and len(a) == 64 and not any(i.endswith("S") for i in a)


class TestAnalysis:
    def test_analysis_runs_on_dry_artifacts_and_names_a_branch(self, tok, tmp_path, monkeypatch):
        # most probable failure: the analysis reads a row field the GPT-OSS cells never save (or
        # the C0 rows' order map), so it crashes only after the pod has been paid for
        rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_gptoss.py"), "--dry-run",
                             "--out", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
        assert rc.returncode == 0, rc.stderr[-2000:]
        an = _load("analyze_gptoss")
        monkeypatch.setattr(an.A, "FLOOR", 0.0)  # stub log-probs put ~no mass on the letters
        out = tmp_path / "analysis.json"
        assert an.main(["--dir", str(tmp_path / "gpt_oss_20b"), "--out", str(out),
                        "--pr-index", "1"]) == 0
        rep = json.loads(out.read_text())
        assert set(rep) >= {"c0", "c1", "c2", "c3", "decision_token_pr", "branch"}
        assert rep["branch"] in {"readout_invalid", "instrument_not_validated", "carries_gap",
                                 "not_detected", "negative_excess_unregistered"}
        assert rep["c0"]["primary"]["n_scenarios"] > 0


class TestGA7AndTimingParity:
    def test_t1_trigger_is_the_near_miss_band_only(self):
        # G-A7. most probable failure: the T=1.0 arm fires on every result (a second arm by
        # default) or on a descriptive C0, instead of only inside the near-miss band
        near = {"primary": kh.c0_verdict([True] * 52 + [False] * 12), "descriptive": False}
        clear = {"primary": kh.c0_verdict([True] * 60 + [False] * 4), "descriptive": False}
        assert pod.needs_t1(near) and not pod.needs_t1(clear)
        assert not pod.needs_t1({**near, "descriptive": True})

    def test_forced_t1_arm_is_saved_at_t1_and_labelled_by_the_analysis(self, tok, tmp_path,
                                                                      monkeypatch):
        # most probable failure: the T=1.0 batch reuses the T=0.7 cell name (overwriting the
        # verdict of record) or the analysis never reads it
        rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_gptoss.py"), "--dry-run",
                             "--force-c0-t1", "--out", str(tmp_path)],
                            capture_output=True, text=True, cwd=REPO)
        assert rc.returncode == 0, rc.stderr[-2000:]
        d = tmp_path / "gpt_oss_20b"
        t07 = [json.loads(x) for x in (d / "c0_generate_low.jsonl").read_text().splitlines()]
        t1 = [json.loads(x) for x in (d / "c0_generate_low_t1.jsonl").read_text().splitlines()]
        assert {r["temperature"] for r in t07} == {0.7} and {r["temperature"] for r in t1} == {1.0}
        an = _load("analyze_gptoss")
        monkeypatch.setattr(an.A, "FLOOR", 0.0)
        out = tmp_path / "a.json"
        an.main(["--dir", str(d), "--out", str(out), "--pr-index", "1"])
        rep = json.loads(out.read_text())
        assert rep["c0"]["t1"]["label"] in {"temperature_robust", "temperature_dependent"}
        assert rep["decision_token_pr"]["raw"]["subsampling"]["seed"] == 0

    def test_real_run_refuses_without_a_matching_validate_record(self, tmp_path):
        # G-A4 timing parity. most probable failure: the real run launches on a GPU class the
        # VALIDATE projection never measured, or with no projection at all
        p = tmp_path / "timing.json"
        assert "run VALIDATE=1 first" in pod.check_timing_record(p, dry=True)
        p.write_text(json.dumps({"stop": True, "projected_hours": 3.0, "max_hours": 2.6,
                                 "gpu_class": "A100-80GB"}))
        assert "projected" in pod.check_timing_record(p, dry=True)
        p.write_text(json.dumps({"stop": False, "gpu_class": "NVIDIA H100 80GB HBM3"}))
        assert "mismatch" in pod.check_timing_record(p, dry=True)
        p.write_text(json.dumps({"stop": False, "gpu_class": "A100-80GB"}))
        assert pod.check_timing_record(p, dry=True) is None
        assert pod.gpu_class("NVIDIA A100 80GB PCIe") == pod.gpu_class("NVIDIA A100-SXM4-80GB")
        rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_gptoss.py"), "--dry-run",
                             "--out", str(tmp_path / "r"), "--require-timing",
                             str(tmp_path / "absent.json")], capture_output=True, text=True,
                            cwd=REPO)
        assert rc.returncode == 4

    def test_validate_records_gpu_class_and_counts_the_t1_batch(self, tok, tmp_path):
        # most probable failure: timing.json lacks the GPU class the real run compares against,
        # or the projection omits the conditional T=1.0 batch
        rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_gptoss.py"), "--dry-run",
                             "--validate", "--out", str(tmp_path)], capture_output=True,
                            text=True, cwd=REPO)
        assert rc.returncode == 0, rc.stderr[-2000:]
        rec = json.loads((tmp_path / "timing.json").read_text())
        assert rec["gpu_class"] == "A100-80GB"
        assert rec["c0_batches_counted"] == 2 * rec["c0_batches"]


def test_p2c_upgrades_torch_before_any_gptoss_call():
    # most probable failure (VALIDATE pod g85xxpdraotqfw, 2026-10-07): p2c loads GPT-OSS on the
    # image's torch 2.4, whose missing torch.accelerator crashes the mxfp4 quantizer at load
    sh = (REPO / "papers/kdg_panel/runpod/remote_kdg_phase2.sh").read_text()
    block = sh[sh.index('if [ "$PROFILE" = "p2c" ]; then'):]
    up = block.index('"torch==2.6.0"')
    assert up < block.index("python -m pytest") < block.index("python $G")
    assert "FATAL: torch.accelerator missing" in block


def test_left_padded_bf16_readout_equals_the_unpadded_one(tok):
    # most probable failure (VALIDATE pod tlsh5rtjq2kgyh, 2026-10-07: forward vs generate 0.46
    # nats): the batched forward reads left-padded rows at positions shifted by their pad count,
    # which bf16 RoPE does not cancel; generate() derives positions from the mask and is unaffected
    import types

    import torch
    from transformers import GptOssConfig, GptOssForCausalLM

    from deepsteer.kdg.schema import load_scenario_dir

    tok.padding_side = "left"
    S, _ = load_scenario_dir(sorted((REPO / "papers/kdg_panel/data").glob("*_scenarios_*.json")))
    P = [kh.render(tok, lib.letter_chat_messages(s, assign_letters(s, 0), "agent", "neutral"), CFG,
                   "medium") for s in S[:8]]
    assert len({len(tok.encode(p, add_special_tokens=False)) for p in P}) > 1  # padding happens
    cfg = GptOssConfig(num_hidden_layers=4, hidden_size=64, intermediate_size=64,
                       num_local_experts=8, num_experts_per_tok=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, vocab_size=len(tok), sliding_window=128,
                       layer_types=["sliding_attention", "full_attention"] * 2)
    torch.manual_seed(0)
    m = types.SimpleNamespace(tok=tok, model=GptOssForCausalLM(cfg).to(torch.bfloat16).eval(),
                              device=torch.device("cpu"), torch=torch)
    read = lib.ModelWrapper.next_logprobs_and_residuals
    pre = [CFG.prefill_primary] * len(P)
    batched, rb = read(m, P, pre, 16)
    alone, ra = read(m, P, pre, 1)
    np.testing.assert_array_equal(batched[:, 32:37], alone[:, 32:37])
    np.testing.assert_array_equal(rb, ra)


class TestGA8:
    @staticmethod
    def _g(padded_ok: bool, unpadded: float, other_ok: bool = True) -> dict:
        return {"checks": {"forward_matches_generate": {"ok": padded_ok},
                           "dequant_bf16": {"ok": other_ok}},
                "unpadded_worst": {"primary": unpadded, "direct_final": unpadded / 2}}

    def test_branch_order_follows_the_registration(self):
        # most probable failure: a padded miss with a clean unpadded read bails (or proceeds on the
        # padded path) instead of switching to length-bucketed batching
        assert pod.g8_branch(self._g(True, 0.9), dry=False) == "proceed"
        assert pod.g8_branch(self._g(False, 0.01), dry=False) == "bucketed"
        assert pod.g8_branch(self._g(False, 0.2), dry=False) == "bail_unpadded"
        assert pod.g8_branch(self._g(False, 0.01, other_ok=False), dry=False) == "bail_other"

    def test_bucketed_readout_is_unpadded_and_in_input_order(self, tok):
        # most probable failure: bucketing regroups rows and returns them in bucket order, so
        # every row's log-probs and residuals land on another scenario
        import types

        import torch
        from transformers import GptOssConfig, GptOssForCausalLM

        from deepsteer.kdg.schema import load_scenario_dir

        tok.padding_side = "left"
        S, _ = load_scenario_dir(sorted((REPO / "papers/kdg_panel/data")
                                        .glob("*_scenarios_*.json")))
        P = [kh.render(tok, lib.letter_chat_messages(s, assign_letters(s, i % 2), "agent",
                                                     "neutral"), CFG, "medium")
             for i, s in enumerate(S[:6] + S[:6])]
        cfg = GptOssConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=64,
                           num_local_experts=4, num_experts_per_tok=2, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=16, vocab_size=len(tok),
                           sliding_window=128, layer_types=["sliding_attention", "full_attention"])
        torch.manual_seed(0)
        m = types.SimpleNamespace(tok=tok, model=GptOssForCausalLM(cfg).to(torch.bfloat16).eval(),
                                  device=torch.device("cpu"), torch=torch, bucket_by_length=True)
        read = lib.ModelWrapper.next_logprobs_and_residuals
        pre = [CFG.prefill_primary] * len(P)
        lb, rb = read(m, P, pre, 16)
        m.bucket_by_length = False
        la, ra = read(m, P, pre, 1)
        np.testing.assert_array_equal(lb, la)
        np.testing.assert_array_equal(rb, ra)


def test_reshuffled_rows_are_restored_to_file_order(tmp_path):
    # G-A9. most probable failure: rows read in the shuffled order are saved unpermuted, so each
    # row's log-probs land on another scenario and the arm reports a spurious batch effect
    import hashlib

    from deepsteer.kdg.schema import load_scenario_dir

    class Fake(lib.StubModel):
        def raw_next_logprobs(self, prompts, batch_size=16, add_special_tokens=True,
                              readout_version=2):
            out = []
            for p in prompts:  # a pure function of the prompt: batch-invariant by construction
                seed = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
                out.append(np.log(np.random.default_rng(seed).dirichlet(np.ones(self.vocab))))
            return np.array(out, dtype=np.float16)

    S, metas = load_scenario_dir(sorted((REPO / "papers/kdg_panel/data")
                                        .glob("*_scenarios_*.json")))
    S = S[:5]
    ctx = lib.Ctx("k", "instruct", Fake(), tmp_path, lib.Manifest(tmp_path, True, metas), False,
                  n_raw_perm=8)
    lib.cell_letter_chat(ctx, S, frame="agent", prefix="neutral")
    lib.cell_letter_chat(ctx, S, frame="agent", prefix="neutral", cell_suffix="_reshuffled",
                         row_order_seed=1)
    a = np.load(tmp_path / "dl_chat_neutral.npz")["logp_decision"]
    b = np.load(tmp_path / "dl_chat_neutral_reshuffled.npz")["logp_decision"]
    np.testing.assert_array_equal(a, b)
    rows = [json.loads(x) for x in (tmp_path / "dl_chat_neutral_reshuffled.jsonl")
            .read_text().splitlines()]
    assert {r["row_order_seed"] for r in rows} == {1}


def test_dry_run_batch_invariance_arm_reaches_the_analysis(tok, tmp_path, monkeypatch):
    # most probable failure: the arm runs but the analysis never reads it (cell-name mismatch)
    rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_gptoss.py"), "--dry-run",
                         "--batch-invariance", "--out", str(tmp_path)],
                        capture_output=True, text=True, cwd=REPO)
    assert rc.returncode == 0, rc.stderr[-2000:]
    an = _load("analyze_gptoss")
    monkeypatch.setattr(an.A, "FLOOR", 0.0)
    out = tmp_path / "a.json"
    an.main(["--dir", str(tmp_path / "gpt_oss_20b"), "--out", str(out), "--pr-index", "1"])
    bi = json.loads(out.read_text())["batch_invariance"]
    assert bi is not None and bi["n_rows"] > 0


def test_p2c_real_run_enables_the_g_a9_arm():
    # most probable failure: G-A9 is pushed but the launcher never passes the flag
    sh = (REPO / "papers/kdg_panel/runpod/remote_kdg_phase2.sh").read_text()
    real = [ln for ln in sh.splitlines() if "--require-timing" in ln and "python $G" in ln]
    assert real and all("--batch-invariance" in ln for ln in real)
