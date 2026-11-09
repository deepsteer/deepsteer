"""CST pilot training pieces (papers/cst_constitutional_steering/scripts/pilot_train.py), on a tiny
checkpoint of Qwen3.8-27B's hybrid architecture with its real tokenizer (st_tiny)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(Path(__file__).resolve().parent)]
torch = pytest.importorskip("torch")
pytest.importorskip("peft")
import pilot_train as PT  # noqa: E402


def test_pack_keeps_every_token_once_with_separators():
    # most probable failure: documents run together without the end-of-text separator, or the
    # packer drops or repeats tokens at chunk boundaries, so the dose in tokens is miscounted
    docs = [list(range(10, 15)), list(range(20, 23)), list(range(30, 38))]
    chunks = PT.pack(docs, eos=0, seq_len=4)
    flat = [t for c in chunks for t in c]
    assert flat == [10, 11, 12, 13, 14, 0, 20, 21, 22, 0, 30, 31, 32, 33, 34, 35, 36, 37, 0]
    assert all(len(c) == 4 for c in chunks[:-1])


def test_parse_chat_reads_both_roles():
    # failure: a multi-line assistant turn is split, or text before the first label leaks in
    m = PT.parse_chat("Here is it.\nUser: Why?\nAssistant: Because.\nIt matters.\nUser: Ok.")
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[1]["content"] == "Because.\nIt matters."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import st_tiny
    from transformers import AutoTokenizer

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.steering import objective_lora as OL

    d = st_tiny.build("qwen38_27b", tmp_path_factory.mktemp("q") / "q")
    wb = WhiteBoxModel(str(d), access_tier=AccessTier.WEIGHTS, device="cpu",
                       torch_dtype=torch.float32)
    OL.inject_lora(wb, rank=4, alpha=8, dropout=0.0)
    return wb, AutoTokenizer.from_pretrained(str(d))


def test_chat_examples_put_loss_on_the_answer_only(tiny):
    # most probable failure (T0.6): the empty think block or the user turn carries loss, so D-chat
    # trains the model to emit empty reasoning, or the end-of-turn token is not a target
    _, tok = tiny
    ex = PT.chat_examples(tok, [{"role": "user", "content": "Is honesty required?"},
                                {"role": "assistant", "content": "Yes, it is."}])
    (ids, mask), = ex
    tgt = [t for t, m in zip(ids, mask) if m]
    assert tok.decode(tgt) == "Yes, it is.<|im_end|>"
    prompt = tok.decode([t for t, m in zip(ids, mask) if not m])
    assert prompt.endswith("<think>\n\n</think>\n\n")


def test_dpo_nll_masks_the_think_block_and_trains(tiny):
    # failure: masked (think) positions still enter the DPO margin or the NLL term, so P-answer is
    # not answer-only; or the loss does not move the policy away from the reference
    wb, _ = tiny
    prompt, chosen, rejected = [5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]
    answer_only = [0, 0, 1, 1]
    loss, info = PT.dpo_nll_loss(wb, prompt, chosen, rejected, answer_only, answer_only)
    assert abs(info["dpo_margin"]) < 1e-5  # adapter starts at the reference (LoRA B = 0)
    lp_full, n_full = PT.masked_logprob(PT._logits(wb, prompt + chosen), prompt + chosen,
                                        [0] * 3 + [1] * 4)
    lp_ans, n_ans = PT.masked_logprob(PT._logits(wb, prompt + chosen), prompt + chosen,
                                      [0] * 3 + answer_only)
    assert n_full == 4 and n_ans == 2 and abs(info["nll"] + float(lp_ans.detach()) / 2) < 1e-4
    opt = torch.optim.AdamW([p for p in wb.model.parameters() if p.requires_grad], lr=5e-2)
    for _ in range(5):
        loss, info = PT.dpo_nll_loss(wb, prompt, chosen, rejected, answer_only, answer_only)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert info["dpo_margin"] > 0


def test_checkpoints_are_saved_at_their_doses(tiny, tmp_path):
    # failure: the 0.3M checkpoint is taken at the wrong step (doses converted with the wrong unit),
    # or a dose's adapter is not saved, so the ladder reads the wrong dose
    wb, _ = tiny
    assert PT.checkpoint_steps([300_000, 1_000_000], 8192) == [37, 123]
    chunks = [[t % 50 + 5 for t in range(i, i + 16)] for i in range(8)]
    rec = PT.train_rung(wb, chunks, lambda ids: PT.lm_loss(wb, ids), per_step_dose=32,
                        doses=[64, 160], out_dir=tmp_path / "rung", lr=1e-3, grad_accum=2)
    assert rec["steps"] == 5 and set(rec["checkpoints"]) == {"64", "160"}
    for d in rec["checkpoints"].values():
        assert (Path(d) / "adapter_config.json").exists()
    r = PT.recovery_step(wb, [(c, [1] * len(c)) for c in chunks[:3]], lr=1e-4,
                         out_dir=tmp_path / "rung_rec")
    assert r["n"] == 3 and (tmp_path / "rung_rec" / "adapter_config.json").exists()


def test_recovery_selection_follows_the_target_deciles():
    # most probable failure (CST-A10): the recovery set skews long or short against none's panel
    # trace lengths (a pool of long replies overfills the top band), so the recovery step itself
    # moves trace length; and a short band must borrow from its neighbours, not shrink the set
    import random

    edges = [0, 100, 200, 300, 400, 500]
    rng = random.Random(0)
    pool = {f"r{i}": rng.randint(1, 500) for i in range(2000)}
    pool |= {f"long{i}": 450 + i % 50 for i in range(3000)}  # a pool heavy in the top band
    pick = PT.select_recovery(pool, edges, 100)
    per = [sum(edges[b] < pool[r] <= edges[b + 1] for r in pick) for b in range(5)]
    assert len(pick) == 100 and per == [20] * 5
    thin = {f"t{i}": 50 for i in range(5)} | {f"m{i}": 250 for i in range(300)}
    pick = PT.select_recovery(thin, edges, 50)
    assert len(pick) == 50 and sum(thin[r] == 50 for r in pick) == 5  # all 5 taken, rest borrowed


def test_split_backward_equals_the_joint_gradient(tiny):
    # most probable failure: the split backward's coefficient or sign is off (the chosen side's
    # NLL term missing, or the rejected side pushed up), so the memory-safe DPO trains a different
    # objective than dpo_nll_loss; the two gradients must match
    wb, _ = tiny
    prm = [p for p in wb.model.parameters() if p.requires_grad]
    with torch.no_grad():  # move off the reference so the margin and its coefficient are non-zero
        for p in prm:
            p.add_(0.05 * torch.randn_like(p))
    prompt, chosen, rejected = [5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]
    mk = [0, 1, 1, 1]
    for p in prm:
        p.grad = None
    loss, _ = PT.dpo_nll_loss(wb, prompt, chosen, rejected, mk, mk)
    loss.backward()
    joint = [p.grad.clone() for p in prm]
    for p in prm:
        p.grad = None
    PT.dpo_nll_backward(wb, prompt, chosen, rejected, mk, mk)
    split = [p.grad.clone() for p in prm]
    for a, b in zip(joint, split):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    for p in prm:
        p.grad = None
