#!/usr/bin/env python3
"""The pilot ladder's training pieces (CST Amendments 20-21; PILOT_BUILD.md §4, §6 items 3-5).

Library code for the SDF and DPO pods; every loss runs on `deepsteer.steering.objective_lora`'s
LoRA model (rank 16, the 496 VALIDATE targets) and its shared `train` loop.

- **SDF (D-raw, D-doc, D-neutral):** documents packed into fixed-length chunks separated by the
  end-of-text token; next-token loss on every position (`pack`, `lm_loss`).
- **D-chat (§4.2 under T0.6):** each assistant turn of a rendered conversation is one example; its
  prompt is the chat template with `enable_thinking=False` (which ends in the empty think block),
  so only the answer and its end-of-turn token carry loss (`chat_examples`).
- **Recovery (Amendment 21 item 5):** `none`'s own thinking-on replies, loss on the whole reply
  (trace and answer), one pass at a fixed learning rate after the rung's own data
  (`recovery_examples`, `recovery_step`).
- **DPO (P-answer, P-trace, P-long):** β 0.1 against the adapter-disabled reference, plus §5's NLL
  term: the chosen response's mean per-token NLL over the loss mask, weight 1.0 (the weight is not
  registered; iterative RPO's α 1). P-answer masks the think block (answer-only); P-trace and
  P-long put loss on trace and answer (`dpo_nll_loss`).
- **Checkpoints by dose:** `train_rung` converts token or pair counts to steps and saves the adapter
  at each registered checkpoint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from deepsteer.steering import objective_lora as OL

DPO_BETA = 0.1  # §5
NLL_WEIGHT = 1.0  # §5's NLL term; weight not registered (iterative RPO's alpha 1), logged
SEQ_LEN = 2048  # VALIDATE's sequence length


# ---- data --------------------------------------------------------------------------------------

def pack(doc_ids: Sequence[Sequence[int]], eos: int, seq_len: int = SEQ_LEN) -> list[list[int]]:
    """Concatenate documents (each followed by ``eos``) and cut into ``seq_len`` chunks; the last
    partial chunk is kept if it holds at least a quarter of ``seq_len``."""
    flat = [t for d in doc_ids for t in (*d, eos)]
    chunks = [flat[i:i + seq_len] for i in range(0, len(flat), seq_len)]
    return [c for c in chunks if len(c) == seq_len or len(c) >= seq_len // 4]


_TURN = re.compile(r"^(User|Assistant):\s*", re.M)


def parse_chat(text: str) -> list[dict[str, str]]:
    """``chat_render_v1`` output ("User:" / "Assistant:" turns) to messages; text before the first
    label is dropped."""
    parts = _TURN.split(text)
    msgs = []
    for role, content in zip(parts[1::2], parts[2::2]):
        if content.strip():
            msgs.append({"role": "user" if role == "User" else "assistant",
                         "content": content.strip()})
    return msgs


def chat_examples(tok, msgs: list[dict[str, str]]) -> list[tuple[list[int], list[int]]]:
    """T0.6: one (ids, loss mask) per assistant turn; the prompt is the template up to that turn
    with ``enable_thinking=False`` (ending in the empty think block), the target the answer and
    the template's end-of-turn token."""
    out = []
    end = tok.convert_tokens_to_ids("<|im_end|>")
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i == 0:
            continue
        prompt = tok.apply_chat_template(msgs[:i], tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False)
        prompt = list(prompt["input_ids"] if hasattr(prompt, "keys") else prompt)  # BatchEncoding
        target = tok.encode(m["content"], add_special_tokens=False) + [end]
        out.append((prompt + target, [0] * len(prompt) + [1] * len(target)))
    return out


def recovery_examples(prompt_ids: Sequence[Sequence[int]],
                      reply_ids: Sequence[Sequence[int]]) -> list[tuple[list[int], list[int]]]:
    """(ids, mask) with loss on the whole reply (trace and answer), the prompt rendered as it was
    generated (thinking on, effort medium)."""
    return [(list(p) + list(r), [0] * len(p) + [1] * len(r)) for p, r in zip(prompt_ids, reply_ids)]


def select_recovery(trace_lens: dict[str, int], edges: Sequence[float], n: int,
                    seed: str = "20261109|cst_recovery_select") -> list[str]:
    """CST-A10 / Amendment 21 item 5: pick ``n`` reply ids whose trace lengths follow the target
    distribution, an equal share from each band between consecutive ``edges`` (A0's panel deciles,
    ``data/recovery_target_v1.json``). A band short of candidates gives what it has and the rest
    comes from the nearest bands, so the total is ``n`` whenever the pool holds ``n``."""
    import random

    k = len(edges) - 1
    bands: list[list[str]] = [[] for _ in range(k)]
    for rid, t in sorted(trace_lens.items()):
        b = next((i for i in range(k) if t <= edges[i + 1]), k - 1)
        if t >= edges[0]:
            bands[b].append(rid)
    rng = random.Random(seed)
    for b in bands:
        rng.shuffle(b)
    want = [n // k + (1 if i < n % k else 0) for i in range(k)]
    take = [b[:w] for b, w in zip(bands, want)]
    spare = [b[w:] for b, w in zip(bands, want)]
    for i in range(k):  # a short band borrows its deficit from the nearest bands' leftovers
        need = want[i] - len(take[i])
        for d in range(1, k):
            for j in (i - d, i + d):
                if need > 0 and 0 <= j < k and spare[j]:
                    got, spare[j] = spare[j][:need], spare[j][need:]
                    take[i] += got
                    need -= len(got)
    out = [rid for t in take for rid in t]
    return sorted(out)


# ---- losses ------------------------------------------------------------------------------------

def _logits(model, ids: list[int]) -> torch.Tensor:
    x = torch.tensor([ids], device=model.model.device)
    return model.model(input_ids=x).logits


def masked_logprob(logits: torch.Tensor, ids: list[int], mask: list[int]) -> tuple[torch.Tensor,
                                                                                   int]:
    """(sum of log p(ids[t] | ids[:t]) over t with mask[t] = 1, the count of such t); t >= 1."""
    lp = F.log_softmax(logits[0, :-1].float(), -1)
    tgt = torch.tensor(ids[1:], device=logits.device)
    m = torch.tensor(mask[1:], device=logits.device, dtype=torch.bool)
    tok_lp = lp.gather(1, tgt[:, None])[:, 0]
    return tok_lp[m].sum(), int(m.sum())


def lm_loss(model, ids: list[int], mask: list[int] | None = None) -> torch.Tensor:
    """Mean next-token NLL over the masked positions (every position when ``mask`` is None)."""
    s, n = masked_logprob(_logits(model, ids), ids, mask or [1] * len(ids))
    return -s / max(n, 1)


def dpo_nll_loss(model, prompt: list[int], chosen: list[int], rejected: list[int],
                 chosen_mask: list[int], rejected_mask: list[int], beta: float = DPO_BETA,
                 nll_weight: float = NLL_WEIGHT) -> tuple[torch.Tensor, dict]:
    """DPO on the masked response positions against the adapter-disabled reference, plus the chosen
    response's mean masked NLL. Masks cover the response only (the prompt never carries loss)."""
    lp, ref, n_w = {}, {}, 0
    for tag, resp, mask in (("w", chosen, chosen_mask), ("l", rejected, rejected_mask)):
        ids = list(prompt) + list(resp)
        full = [0] * len(prompt) + list(mask)
        lp[tag], n = masked_logprob(_logits(model, ids), ids, full)
        if tag == "w":
            n_w = n
        with torch.no_grad(), OL.adapter_disabled(model):
            ref[tag], _ = masked_logprob(_logits(model, ids), ids, full)
    margin = beta * ((lp["w"] - ref["w"]) - (lp["l"] - ref["l"]))
    nll = -lp["w"] / max(n_w, 1)
    loss = -F.logsigmoid(margin) + nll_weight * nll
    return loss, {"dpo_margin": float(margin.detach()), "nll": float(nll.detach())}


# ---- training ----------------------------------------------------------------------------------

def checkpoint_steps(doses: Sequence[float], per_step: float) -> list[int]:
    """Steps at which the cumulative dose (tokens or pairs) first reaches each registered dose."""
    return [max(1, int(-(-d // per_step))) for d in doses]


def train_rung(model, examples: list, loss_fn, *, per_step_dose: float, doses: Sequence[float],
               out_dir: Path, lr: float, grad_accum: int = 4, seed: int = 0,
               warmup_steps: int = 10) -> dict:
    """Train one rung to its largest registered dose; save the adapter at every dose in ``doses``
    (``out_dir/<dose>``). ``per_step_dose`` is the dose one optimizer step consumes (tokens per
    step for SDF, pairs per step for DPO)."""
    steps = checkpoint_steps(doses, per_step_dose)
    at = dict(zip(steps, doses))
    it = OL.micro_batches(examples, seed)
    saved = {}

    def step_loss(_micro: int):
        loss = loss_fn(next(it))
        info = {}
        if isinstance(loss, tuple):
            loss, info = loss
        return loss, None, 0.0, info

    def save(step: int) -> dict:
        if step in at:
            d = out_dir / f"{at[step]:g}"
            model.model.save_pretrained(str(d))
            saved[f"{at[step]:g}"] = str(d)
        return {}

    hist, _ = OL.train(model, step_loss, max_steps=max(steps), lr=lr, grad_accum=grad_accum,
                       warmup_steps=min(warmup_steps, max(steps) // 2), eval_fn=save,
                       eval_at=steps, seed=seed)
    rec = {"steps": max(steps), "checkpoints": saved, "doses": list(doses),
           "loss_first": hist[0].loss, "loss_last": hist[-1].loss,
           "loss": [round(h.loss, 5) for h in hist]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def recovery_step(model, examples: list, *, lr: float, out_dir: Path, seed: int = 0) -> dict:
    """Amendment 21 item 5: one pass over the recovery set at a fixed learning rate (no warm-up or
    decay), continuing the rung's adapter; the same step on every document rung."""
    params = [p for p in model.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    model.model.train()
    losses = []
    for i in torch.randperm(len(examples), generator=g).tolist():
        ids, mask = examples[i]
        loss = lm_loss(model, ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    model.model.eval()
    model.model.save_pretrained(str(out_dir))
    rec = {"n": len(examples), "lr": lr, "loss_first": losses[0] if losses else None,
           "loss_last": losses[-1] if losses else None}
    (out_dir / "recovery.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec
