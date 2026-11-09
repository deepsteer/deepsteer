"""Tests for deepsteer/faithfulness/gradient_necessity.py (CST_PLAN Amendment 1 D8)."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch

from deepsteer.faithfulness import find_regions, gradient_necessity

DELIM = [[7, 8, 9], [7, 8, 10]]  # base tokenization first, a merged-boundary variant second


def test_regions_skip_a_delimiter_in_the_prompt():
    # most probable failure: matching a </think> that the prompt itself contains (format
    # instructions), which would put the whole response in the answer region
    ids = [1, 7, 8, 9, 2, 3, 4, 5, 7, 8, 9, 6, 6]
    r = find_regions(ids, prompt_len=5, end_think_variants=DELIM)
    assert r.prompt == (0, 5) and r.cot == (5, 11) and r.answer == (11, 13)


def test_regions_variant_and_missing_delimiter():
    # failure: a merged-boundary tokenization is missed, or a missing delimiter is not flagged
    assert find_regions([1, 2, 3, 7, 8, 10, 4], 2, DELIM).cot == (2, 6)
    r = find_regions([1, 2, 3, 4], 2, DELIM)
    assert r.cot is None and r.answer is None


def _tiny():
    from transformers import AutoModelForCausalLM

    try:
        m = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", local_files_only=True)
    except OSError:
        pytest.skip("sshleifer/tiny-gpt2 not in the local HF cache")
    return m.float().eval()


def _seq() -> tuple[list[int], int]:
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(20, 500, (12,), generator=g).tolist()
    cot = torch.randint(20, 500, (20,), generator=g).tolist()
    ans = torch.randint(20, 500, (6,), generator=g).tolist()
    return prompt + cot + DELIM[0] + ans, len(prompt)


@pytest.mark.slow
def test_shares_partition_and_match_autograd():
    # failure: the loss is read at unshifted positions (predicting t from t) or the denominator
    # excludes a region; shares must partition 1 and equal a direct autograd recomputation
    m = _tiny()
    ids, pl = _seq()
    res = gradient_necessity(m, ids, pl, DELIM, spans={"cot_head": (pl, pl + 5)})
    assert math.isclose(res.cot_l1 + res.prompt_l1 + res.answer_l1, 1.0, abs_tol=1e-5)
    assert math.isclose(res.cot_l2 + res.prompt_l2 + res.answer_l2, 1.0, abs_tol=1e-5)
    assert res.n_cot == 23 and res.n_answer == 6
    x = torch.tensor(ids)
    emb = m.get_input_embeddings()(x[None]).detach().requires_grad_(True)
    lp = torch.log_softmax(m(inputs_embeds=emb).logits[0], -1)
    a0 = pl + 23
    loss = -lp[torch.arange(a0 - 1, len(ids) - 1), x[a0:]].mean()
    (gr,) = torch.autograd.grad(loss, emb)
    n = gr[0].norm(dim=-1)
    assert math.isclose(res.cot_l1, float(n[pl:a0].sum() / n.sum()), rel_tol=1e-4)
    assert math.isclose(res.spans["cot_head"]["l2"],
                        float(n[pl:pl + 5].pow(2).sum() / n.pow(2).sum()), rel_tol=1e-4)


@pytest.mark.slow
def test_no_delimiter_is_nan():
    # failure: a response without </think> silently scores its whole response as CoT or answer
    m = _tiny()
    res = gradient_necessity(m, [30, 31, 32, 33, 34, 35], 3, DELIM, spans={"s": (3, 4)})
    assert math.isnan(res.cot_l1) and math.isnan(res.spans["s"]["l1"])


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("FAITHFUL_COT_SRC"),
                    reason="set FAITHFUL_COT_SRC to a faithful-cot@f9e1f295 checkout")
def test_parity_with_reference():
    # failure: the re-implementation drifts from the reference function (D8's parity test). The
    # reference is unlicensed and is never committed; this test executes its code from a local
    # checkout, so it runs only when the author sets FAITHFUL_COT_SRC
    src = Path(os.environ["FAITHFUL_COT_SRC"]) / "eval/faithfulness_metrics_clean_qwen.py"
    spec = importlib.util.spec_from_file_location("fc_ref", src)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    m = _tiny()
    ids, pl = _seq()
    a0 = pl + 23
    pos = torch.arange(len(ids))[None]
    inp = ref.TokenizedInputs(input_ids=torch.tensor(ids)[None], prompt_mask=pos < pl,
                              cot_mask=(pos >= pl) & (pos < a0), answer_mask=pos >= a0,
                              attention_mask=None)
    ref_nec, ref_de = ref.grad_cot_necessity(m, inp), ref.grad_direct_effect(m, inp)
    res = gradient_necessity(m, ids, pl, DELIM)
    assert math.isclose(res.cot_l1, float(ref_nec[0]), rel_tol=1e-4)
    assert math.isclose(res.cot_l2, float(ref_nec[1]), rel_tol=1e-4)
    assert math.isclose(res.prompt_l1, float(ref_de[0]), rel_tol=1e-4)
    assert math.isclose(res.prompt_l2, float(ref_de[1]), rel_tol=1e-4)
