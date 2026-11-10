"""CST-A22's parity mini-pod (papers/cst_constitutional_steering/scripts/cst_parity_pod.py) and
Amendment 25 A4's gate (pilot_pod_lib.parity_record)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "tests/scripts")]

import pilot_pod_lib as PL  # noqa: E402


def _lp(ps: list[float]) -> list[float]:
    return [math.log(p) for p in ps]


BASE = [_lp([0.7, 0.1, 0.1, 0.05, 0.05])] * 4
ATT = [_lp([0.1, 0.7, 0.1, 0.05, 0.05])] * 4  # TV(base, attached) = 0.6


def test_gate_fails_a_served_adapter_that_reads_as_base():
    # most probable failure (CST-A22): an adapter served as the base model passes, because the old
    # rule only asked that served sit below base by any margin
    noserve = PL.parity_record("t", BASE, ATT, BASE)
    assert noserve["tv_base_vs_attached"] == pytest.approx(0.6) and not noserve["passes"]
    near = [_lp([0.12, 0.68, 0.1, 0.05, 0.05])] * 4
    ok = PL.parity_record("t", near, ATT, BASE, served_base=BASE)
    assert ok["passes"] and ok["tv_served_base_vs_hf_base"] == 0.0
    halfway = [_lp([0.4, 0.4, 0.1, 0.05, 0.05])] * 4  # 0.3 from each: fails both sides of the rule
    assert not PL.parity_record("t", halfway, ATT, BASE)["passes"]
    assert ok["served_letters"] == near  # the served letters ride in the record


def test_parity_records_are_never_overwritten(tmp_path):
    # failure (CST-A22): each read call overwrote the kind's one record with another adapter's read
    f = tmp_path / "parity" / "x.json"
    PL.write_once(f, {"a": 1})
    with pytest.raises(SystemExit):
        PL.write_once(f, {"a": 2})
    assert json.loads(f.read_text()) == {"a": 1}


def test_rename_moves_every_decoder_key(tmp_path):
    # failure: the fixed-path probe moves only some keys, so a partly served adapter passes
    import torch
    from safetensors.torch import load_file, save_file

    src = tmp_path / "a"
    src.mkdir()
    keys = ["base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
            "base_model.model.model.layers.3.linear_attn.in_proj_qkv.lora_B.weight"]
    save_file({k: torch.zeros(2, 2) for k in keys}, str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text("{}")
    pre = {"trainer": "model.layers", "served": "model.language_model.layers"}
    rec = PL.rename_adapter(src, tmp_path / "b", pre)
    out = sorted(load_file(str(tmp_path / "b" / "adapter_model.safetensors")))
    assert rec["moved"] == 2 and all(k.startswith(
        "base_model.model.model.language_model.layers.") for k in out)


@pytest.mark.slow
def test_dry_run_reads_a_trainer_class_adapter_as_not_served(tmp_path):
    # most probable failure (CST-A22): the mini-pod serves through the class the adapter was
    # saved from (as DryEngine did), so a silent no-op load cannot show; the dry run serves through
    # the served class and reads the saved adapter as the base and the renamed one as attached
    import cst_parity_pod as Q
    import torch

    _, wb, _ = PL.load_model(True)
    torch.manual_seed(0)
    for n, p in wb.model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.05)
    ad = tmp_path / "src" / "1e+06"
    wb.model.save_pretrained(str(ad))
    out = tmp_path / "out"
    ads = Q._adapters(str(ad))  # noqa: SLF001
    pre = Q.stage_prep(out, ads, True)
    assert pre["prefixes"] == {"trainer": "model.layers", "served": "model.language_model.layers"}
    Q.stage_hf(out, ads, True)
    Q.stage_served(out, ads, True)
    s = Q.stage_gate(out)
    saved, renamed = s["tags"]["1e06__saved"], s["tags"]["1e06__renamed"]
    assert not saved["passes"] and saved["tv_served_vs_base"] < 1e-6
    assert renamed["passes"] and renamed["tv_served_vs_attached"] < 1e-6
    assert s["branch"].startswith("R_noserve: ") and s["tv_base_after_load_vs_base"] < 1e-6
    with pytest.raises(SystemExit):
        Q.stage_gate(out)  # the per-tag records are never overwritten
