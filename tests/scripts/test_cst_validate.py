"""CST VALIDATE's pod driver and the judge scorer (papers/cst_constitutional_steering/scripts)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "tests/scripts")]

import cst_judge as J  # noqa: E402
import cst_validate_pod as V  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    return V._model_path(True)


def test_lora_targets_cover_deltanet_and_skip_vision(tiny):
    # most probable failure: the target rule misses the hybrid model's DeltaNet projections (so
    # 48 of 64 layers train nothing) or reaches into the vision tower / MTP head
    from transformers import AutoModelForImageTextToText

    from deepsteer.steering.objective_lora import ATTN, LINEAR_ATTN, MLP

    m = AutoModelForImageTextToText.from_pretrained(tiny)
    mods = V._lora_targets(m, ATTN + MLP + LINEAR_ATTN)
    kinds = {n.rsplit(".", 1)[-1] for n in mods}
    assert {"in_proj_qkv", "in_proj_z", "out_proj"} <= kinds
    assert {"q_proj", "o_proj", "gate_proj", "down_proj"} <= kinds
    assert not any("visual" in n or "vision" in n or "mtp" in n for n in mods)


def test_dry_run_reloads_the_adapter_bit_identically(tiny, tmp_path):
    # most probable failure: a saved adapter that reloads with different weights (dtype or key
    # mismatch), which would make every later adapter read a different model than the one trained
    V.stage_gen(tmp_path, True)
    V.stage_lora_hf(tmp_path, True)
    rec = json.loads((tmp_path / "lora_hf.json").read_text())
    assert rec["v3_passes"] and rec["deltanet_targets"] and rec["passes"]
    assert "sdf_tokens_per_s" in rec and rec["letters_written"] > 0


def test_gradnec_backward_runs_with_checkpointing_active(tiny, tmp_path):
    # most probable failure (the first VALIDATE's OOM): the Grad-Nec backward runs in eval mode,
    # where HF skips gradient checkpointing and stores every layer's activations, with the full
    # vocabulary's logits at every position
    import torch
    from transformers import AutoModelForImageTextToText

    m = AutoModelForImageTextToText.from_pretrained(tiny, dtype=torch.float32)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.eval()  # the state the first run left the model in
    r = V.gradnec_backward(m, torch.tensor([list(range(5, 69))]))
    assert r["checkpointing_active"] and r["logits_positions"] == 1 and r["grad_finite"]
    assert not any(p.grad is not None for p in m.parameters())  # no weight gradient
    V.stage_gradnec(tmp_path, True)
    assert json.loads((tmp_path / "gradnec.json").read_text())["v7_passes"]


def test_judges_save_raw_replies_and_output_lengths(tiny, tmp_path):
    # most probable failure (the first VALIDATE): a judge's parse failures with no raw replies to
    # read, so truncation cannot be told from a format mismatch
    V.stage_gen(tmp_path, True)
    V.stage_judges(tmp_path, True, ["glm47_flash", "glm47_flash_nothink"])
    rec = json.loads((tmp_path / "judges.json").read_text())
    for name in ("glm47_flash", "glm47_flash_nothink"):
        raw = [json.loads(x) for x in open(tmp_path / f"judges_raw_{name}.jsonl")]
        assert len(raw) == rec[name]["n"] and all(x["text"] and x["out_tokens"] for x in raw)
        assert {"hit_cap_share", "out_tokens_p90", "parse_rate_uncapped"} <= set(rec[name])
        rl = rec[name]["relabel"]
        assert rl["items_parsed"] == rl["items"] == 10 * rl["calls"]


def test_judge_parse_reads_the_last_object_and_validates_ids():
    # most probable failure: a reasoning judge's draft JSON inside its thinking is read instead of
    # its final answer, or a hallucinated tenet id is kept
    text = ('thinking {"concluded_option": "o2", "applied_tenets": []} then '
            '{"concluded_option": "o1", "applied_tenets": ["A036", "A999"], "evidence": "x"}')
    r = J.parse_judgment(text, {"o1", "o2", "o3"}, {"A036", "A041"})
    assert r["concluded_option"] == "o1" and r["applied_tenets"] == ["A036"]
    assert r["dropped_ids"] == 1
    assert J.parse_judgment('{"concluded_option": "o9"}', {"o1"}, set())["ok"] is False
    assert J.parse_judgment("no json", {"o1"}, set())["concluded_option"] == "invalid"


def test_judge_prompt_shows_no_labels():
    # the judge must not be anchored on the labels of record or the action of record
    p = J.judge_prompt("situation", [("o1", "a"), ("o2", "b")], "trace", "Answer: A",
                       [("A036", "claim")])
    for word in ("directed away", "label", "of record", "comply"):
        assert word not in p


def test_price_counts_the_registered_rollouts_per_panel(tmp_path):
    # most probable failure: the price drops a registered condition (operator_pr) or the long-prompt
    # judge work (relabel, Layer A, the third judge's votes), or prices a panel with a missing judge
    (tmp_path / "gen.json").write_text(json.dumps(
        {"load_s": 0, "panel": {"rollouts_per_hour": 1000.0},
         "a0s": {"rollouts_per_hour": 500.0, "cap": 1536}}))
    j = {"load_s": 0, "calls_per_hour": 2000.0, "relabel": {"calls": 8, "seconds": 3600.0}}
    (tmp_path / "judges.json").write_text(json.dumps(
        {"gpt_oss_120b": j, "glm47_flash": j, "olmo31_32b_instruct": j}))
    V.stage_price(tmp_path)
    rec = json.loads((tmp_path / "price.json").read_text())
    assert rec["rollouts_per_arm"] == 586 * (4 + 2 + 4 + 6) + 132 * 8 * 2 + 102 * 8
    g = rec["judge_hours"]["gpt_oss_120b"]
    assert g["scoring"] == pytest.approx((2 * rec["rollouts_per_arm"] + 100) / 2000)
    assert g["long_prompt_calls"] == pytest.approx((72 + 15) * 450 / 3600)
    assert rec["judge_hours"]["glm47_flash"]["long_prompt_calls"] == pytest.approx(
        (72 + 15 + 22) * 450 / 3600)  # a third-family judge also casts the 220 votes
    assert list(rec["panels"]) == ["gpt_oss_120b + glm47_flash + olmo31_32b_instruct"]
