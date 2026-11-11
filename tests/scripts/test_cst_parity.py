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


def test_served_adapter_key_path_matches_the_served_class(tmp_path):
    # most probable failure (CST-A22, Amendment 26 item 4): the adapter handed to vLLM carries the
    # trainer class's path (model.layers) while the served class names its decoder
    # model.language_model.layers, so the load is a silent no-op; and the dry stand-in is a class
    # vLLM does not serve, so no dry run can show it
    import cst_s0a_pod as S
    from transformers import AutoConfig

    path = S.V._model_path(True)  # noqa: SLF001
    pre = PL.decoder_prefixes(path)
    _, wb, _ = PL.load_model(True)
    ad = tmp_path / "adapters" / "x"
    wb.model.save_pretrained(str(ad))
    assert PL.key_prefix(ad) == pre["trainer"] != pre["served"]
    served = PL.served_copy(tmp_path, "x", str(ad), True)
    assert PL.key_prefix(served) == pre["served"]
    assert PL.served_copy(tmp_path, "y", served, True) == served  # already on the served path
    eng = S.subject(True, "a0")[1]
    assert type(eng.m).__name__ == AutoConfig.from_pretrained(path).architectures[0]


def test_served_copy_never_builds_model_classes_in_the_read_process(tmp_path, monkeypatch):
    # most probable failure (the re-read pod, 2026-11-10): the read stage builds its vLLM engine,
    # then derives the decoder paths by building the model classes in the same process, where the
    # 27B config raises on vocab_size, so every read fails before its parity record
    import cst_s0a_pod as S

    _, wb, _ = PL.load_model(True)
    ad = tmp_path / "adapters" / "x"
    wb.model.save_pretrained(str(ad))

    def boom(path):
        raise AttributeError("'Qwen3_5Config' object has no attribute 'vocab_size'")

    monkeypatch.setattr(PL, "decoder_prefixes", boom)
    PL._PREFIXES.clear()  # noqa: SLF001
    served = PL.served_copy(tmp_path, "x", str(ad), True)
    pre = json.loads((tmp_path / "parity" / "prefixes.json").read_text())
    assert pre["served"] == "model.language_model.layers" and PL.key_prefix(served) == pre["served"]
    assert S.V._model_path(True)  # noqa: SLF001


def test_served_copy_asserts_the_record_against_the_lookup_and_the_keys(tmp_path, monkeypatch):
    # most probable failure (Amendment 28 item 2): the read stage trusts a decoder path that does
    # not match the adapter's keys or the served class, and serves a copy that loads as a no-op
    import torch
    from safetensors.torch import save_file

    rec = PL.served_paths()
    assert (rec["trainer"], rec["served"]) == ("model.layers", "model.language_model.layers")
    _, wb, _ = PL.load_model(True)
    ad = tmp_path / "adapters" / "x"
    wb.model.save_pretrained(str(ad))
    PL._PREFIXES.clear()  # noqa: SLF001
    assert PL.key_prefix(PL.served_copy(tmp_path, "x", str(ad), True)) == rec["served"]
    odd = tmp_path / "adapters" / "odd"
    odd.mkdir(parents=True)
    save_file({"base_model.model.transformer.layers.0.q.lora_A.weight": torch.zeros(2, 2)},
              str(odd / "adapter_model.safetensors"))
    with pytest.raises(PL.ServedPathMismatch):
        PL.served_copy(tmp_path, "odd", str(odd), True)  # keys on neither path of record
    monkeypatch.setattr(PL, "served_paths", lambda: dict(rec, served="model.lm.layers"))
    with pytest.raises(PL.ServedPathMismatch):
        PL.served_copy(tmp_path, "y", str(ad), True)  # the record disagrees with the lookup


def test_tv_counts_a_prompt_without_served_letters_as_the_maximum():
    # most probable failure (2-ctx's smoke pass, 2026-11-11): every letter outside vLLM's top 20
    # reads -1e9, the renormalization divides by zero and the stage crashes instead of banking
    missing = [[-1e9] * 5] * 4
    assert PL.tv(missing, ATT) == 1.0 and PL.letters_missing(missing) == 4
    rec = PL.parity_record("t", missing, ATT, BASE)
    assert not rec["passes"] and rec["served_prompts_without_letters"] == 4
    half = missing[:2] + ATT[:2]
    assert PL.tv(half, ATT) == pytest.approx(0.5) and PL.letters_missing(half) == 2


def _two(p: float) -> list[list[float]]:
    return [[math.log(p), math.log(1 - p)]] * 4  # TV between two of these is |p - q|


def test_floor_aware_gate_passes_a_served_small_shift_adapter():
    # most probable failure (CST-A24, Amendment 29 item 1): an adapter served correctly but with a
    # dose-0 shift under five times the serving floor is banked, because 0.2 of its shift sits
    # below the floor; P-long's own numbers: shift 0.028, floor 0.0082, served 0.0095 from attached
    base, attached = _two(0.5), _two(0.528)
    served, served_base = _two(0.5185), _two(0.5082)
    rec = PL.parity_record("p_long", served, attached, base, served_base)
    assert rec["tv_base_vs_attached"] == pytest.approx(0.028)
    assert rec["floor"] == pytest.approx(0.0082) and rec["attached_max"] == pytest.approx(0.0123)
    assert rec["passes"] and rec["read"] and not rec["parity_unresolved"]
    noserve = PL.parity_record("p_long", base, attached, base, served_base)
    assert not noserve["passes"] and not noserve["read"]  # served as the base: still banked
    tiny = PL.parity_record("t", _two(0.51), _two(0.52), base, served_base)  # shift 0.02 < 3 floor
    assert tiny["parity_unresolved"] and tiny["read"]  # read, marked: only a move is reportable


def test_served_truncation_tolerates_a_deep_letter_but_not_an_empty_prompt():
    # most probable failure (2-ctx and SDF pod 2, 2026-11-11): the smoke search required all five
    # letters inside the served top 20, which the base itself does not meet (a letter at rank 53),
    # so the search stopped at its first scale; only a prompt with no letter left breaks the read
    row = _lp([0.6, 0.3, 0.0999, 0.00005, 0.00005])
    ranks = [[0, 1, 4, 53, 76]] * 4
    dist, missing = PL.served_truncation([row] * 4, ranks)
    assert missing == 0 and dist < 1e-3  # two deep, tiny letters drop out at almost no cost
    dist, missing = PL.served_truncation([row] * 4, [[25, 30, 40, 53, 76]] * 4)
    assert missing == 4 and dist == 1.0  # no letter left: the gate cannot read the prompt



def test_smoke_scale_search_finds_a_narrow_window_a_step_overshoots():
    # most probable failure (2-ctx's second smoke pass, 2026-11-11): the 27B's TV is steep in the
    # scale, so one 1.4 step went from under the bar to TV 0.637 with prompts left without a
    # letter, and the step-up search bailed; bisection must land inside the narrow window
    # TV crosses 0.15 at 0.05; the served read breaks past 0.052 (the 1.4 grid steps over it)
    def curve(s):
        tv_ = min(1.0, (s / 0.05) ** 12 * 0.15)
        return tv_, 0.0 if s < 0.052 else 0.27, 0 if s < 0.052 else 5

    scale, trace = PL.smoke_scale_search(curve, 0.005, 0.15)
    assert scale is not None and 0.05 <= scale < 0.052
    assert trace[-1]["verdict"] == "ok" and trace[-1]["scale"] == scale  # model left there
    assert any(t["verdict"] == "over" for t in trace)  # the step that overshot, then bisected
    first_over, _ = PL.smoke_scale_search(curve, 0.08, 0.15)  # over at the start: steps down
    assert first_over is not None and 0.05 <= first_over < 0.052
    none, tr = PL.smoke_scale_search(lambda s: (0.1, 0.0, 0 if s < 0.03 else 3), 0.005, 0.15)
    assert none is None and tr  # no window: the read breaks before the TV reaches the bar


def test_smoke_search_skips_a_direction_whose_bar_sits_on_a_cliff():
    # most probable failure (2-ctx's third smoke pass, 2026-11-11): the accepted scale sat one
    # step below a collapse (TV 0.24 at 0.087, 0.69 at 0.103), the kernels' small differences were
    # amplified and parity failed; that direction must be refused and the next seed taken
    calls = []

    def make_eval(seed):
        def ev(s):
            calls.append((seed, s))
            if seed == 0:  # flat under the bar, then a cliff
                return (0.12 if s < 0.085 else min(1.0, 0.24 + (s - 0.087) * 28)), 0.0, 0
            return min(1.0, 3.0 * s), 0.0, 0  # smooth: TV 0.15 at 0.05
        return ev

    seed, scale, rec = PL.smoke_search_seeds(make_eval, 0.005, 0.15)
    assert rec[0]["scale"] is not None and not rec[0]["stable"]  # found, but on the cliff
    assert seed == 1 and rec[1]["stable"] and 3.0 * scale >= 0.15
    assert calls[-1] == (1, scale)  # the model is left at the accepted seed and scale
