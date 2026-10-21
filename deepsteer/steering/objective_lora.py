"""LoRA training with representation and preference objectives (ST_PLAN §5.1-5.3, T0.6).

Builds on the conventions of :mod:`deepsteer.steering.chat_lora_trainer` (PEFT LoRA on a frozen
``WhiteBoxModel``, AdamW, linear warm-up then cosine decay, per-step records) and adds objectives
that act on token sequences with position masks rather than on chat SFT data:

- ``dpo_loss``: DPO with the adapter-disabled model as the reference (no second copy in memory);
  log-probabilities summed over the response tokens;
- ``anchor_loss``: ``mean over anchor positions of max(0, tau - <h_L(t), u> / |h_L(t)|)``, the
  representation anchor (§5.2), read at hidden-state index ``L`` (block ``L - 1``'s output);
- ``self_loss``: mean squared distance between the standardized residual at the anchor positions
  and a standardized target (§5.3), standardization by a per-dimension ``sigma`` whose provenance
  travels with the adapter;
- ``kl_to_base``: ``KL(p_base || p_adapter)`` per token on the masked tokens (the general-text
  slice and the trace tokens outside the anchoring class), base = adapter disabled.

LoRA targets are restricted to the text decoder: only ``nn.Linear`` modules inside the decoder
layers ``WhiteBoxModel`` detects (so a VLM's vision tower, which also has ``q_proj``, is never
adapted), named in ``ATTN`` / ``MLP`` / ``LINEAR_ATTN``; MoE experts stored as parameters (GPT-OSS's
``mlp.experts.gate_up_proj`` / ``down_proj``) go through PEFT's ``target_parameters``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
LINEAR_ATTN = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
EXPERT_PARAMS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def lora_targets(model: WhiteBoxModel) -> tuple[list[str], list[str]]:
    """Full names of the text decoder's adaptable Linear modules, and the suffixes of MoE expert
    parameters (PEFT ``target_parameters``). Anything outside the decoder layers is excluded by
    construction."""
    layers = model._decoder_layers()  # noqa: SLF001
    names = {id(m): n for n, m in model.model.named_modules()}
    prefix = names[id(layers)]
    allowed = set(ATTN + MLP + LINEAR_ATTN)
    mods, params = [], []
    for n, m in model.model.named_modules():
        if n.startswith(prefix + ".") and isinstance(m, torch.nn.Linear) \
                and n.rsplit(".", 1)[-1] in allowed:
            mods.append(n)
    for n, _p in model.model.named_parameters():
        if n.startswith(prefix + "."):
            params += [e for e in EXPERT_PARAMS if n.endswith(e)]
    return sorted(mods), sorted(set(params))  # PEFT matches expert parameters by suffix


def inject_lora(model: WhiteBoxModel, rank: int = 16, alpha: int = 32,
                dropout: float = 0.05) -> dict:
    """PEFT LoRA on the text decoder's attention, MLP (and linear-attention) projections, plus
    MoE expert parameters where the model stores experts as parameters. Returns the config used."""
    from peft import LoraConfig, get_peft_model

    mods, params = lora_targets(model)
    if not mods and not params:
        raise ValueError("no LoRA target inside the decoder layers")
    kw: dict[str, Any] = {"target_modules": mods}
    if params:
        kw["target_parameters"] = params
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     task_type="CAUSAL_LM", **kw)
    model._model = get_peft_model(model._model, cfg)  # noqa: SLF001
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    logger.info("LoRA on %d modules + %d expert parameter groups: %d trainable", len(mods),
                len(params), trainable)
    return {"rank": rank, "alpha": alpha, "dropout": dropout, "target_modules": mods,
            "target_parameters": params, "trainable_params": trainable}


@contextmanager
def adapter_disabled(model: WhiteBoxModel):
    """The reference policy: the same weights with the adapter off (no second copy)."""
    with model.model.disable_adapter():
        yield


def _hidden_at(model: WhiteBoxModel, layer: int, store: dict):
    """Forward hook capturing hidden-state index ``layer`` (block ``layer - 1``'s output)."""
    def fn(_m, _i, out):
        store["h"] = out[0] if isinstance(out, tuple) else out
    return model._get_layer_module(layer - 1).register_forward_hook(fn)  # noqa: SLF001


def _forward(model: WhiteBoxModel, ids: torch.Tensor, layer: int | None = None):
    store: dict = {}
    h = _hidden_at(model, layer, store) if layer is not None else None
    try:
        logits = model.model(input_ids=ids).logits
    finally:
        if h is not None:
            h.remove()
    return logits, store.get("h")


def sequence_logprob(logits: torch.Tensor, ids: torch.Tensor, start: int) -> torch.Tensor:
    """Sum of log p(ids[t] | ids[:t]) for t >= start (one sequence, batch 1)."""
    lp = F.log_softmax(logits[0, start - 1: -1].float(), -1)
    return lp.gather(1, ids[0, start:, None]).sum()


def dpo_loss(model: WhiteBoxModel, prompt: Sequence[int], chosen: Sequence[int],
             rejected: Sequence[int], beta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """DPO (beta 0.1 in ST) with the adapter-disabled model as the reference."""
    dev = model.model.device
    n = len(prompt)
    out, ref = {}, {}
    for tag, resp in (("w", chosen), ("l", rejected)):
        ids = torch.tensor([list(prompt) + list(resp)], device=dev)
        out[tag] = sequence_logprob(_forward(model, ids)[0], ids, n)
        with torch.no_grad(), adapter_disabled(model):
            ref[tag] = sequence_logprob(_forward(model, ids)[0], ids, n)
    margin = beta * ((out["w"] - ref["w"]) - (out["l"] - ref["l"]))
    return -F.logsigmoid(margin), {"dpo_margin": float(margin.detach()),
                                   "chosen_reward": float(beta * (out["w"] - ref["w"]).detach())}


def _kl_tokens(logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean KL(base || adapter) over the next-token predictions at the masked positions."""
    if not bool(mask.any()):
        return logits.sum() * 0.0
    lp = F.log_softmax(logits[0][mask].float(), -1)
    lq = F.log_softmax(ref_logits[0][mask].float(), -1)
    return (lq.exp() * (lq - lp)).sum(-1).mean()


def anchor_and_kl(model: WhiteBoxModel, ids: Sequence[int], *, layer: int,
                  anchor_mask: Sequence[bool], kl_mask: Sequence[bool], kind: str,
                  direction: torch.Tensor | None = None, tau: float | None = None,
                  target: torch.Tensor | None = None, sigma: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One trace: the representation term at the anchor positions (``kind`` "anchor": the
    hinge on the normalized projection onto ``direction``; "self": the standardized MSE to
    ``target``) and the KL-to-base term at the ``kl_mask`` positions."""
    dev = model.model.device
    x = torch.tensor([list(ids)], device=dev)
    am = torch.tensor(list(anchor_mask), device=dev)
    km = torch.tensor(list(kl_mask), device=dev)
    logits, h = _forward(model, x, layer)
    with torch.no_grad(), adapter_disabled(model):
        ref_logits, _ = _forward(model, x)
    hs = h[0][am].float()
    if not len(hs):
        rep = logits.sum() * 0.0
        info: dict = {"n_anchor": 0}
    elif kind == "anchor":
        m = (hs @ direction.float()) / hs.norm(dim=-1).clamp_min(1e-6)
        rep = F.relu(tau - m).mean()
        info = {"n_anchor": int(am.sum()), "margin_mean": float(m.mean().detach())}
    elif kind == "self":
        z = hs / sigma.float()
        rep = ((z - (target.float() / sigma.float())) ** 2).mean()
        info = {"n_anchor": int(am.sum())}
    else:
        raise ValueError(kind)
    return rep, _kl_tokens(logits, ref_logits, km), info


def kl_general(model: WhiteBoxModel, ids: Sequence[int]) -> torch.Tensor:
    """KL-to-base on a general-text sequence (every position)."""
    x = torch.tensor([list(ids)], device=model.model.device)
    logits, _ = _forward(model, x)
    with torch.no_grad(), adapter_disabled(model):
        ref, _ = _forward(model, x)
    return _kl_tokens(logits, ref, torch.ones(x.shape[1], dtype=torch.bool, device=x.device))


@dataclass
class ObjectiveStep:
    step: int
    loss: float
    rep: float | None
    kl: float
    lr: float
    info: dict = field(default_factory=dict)


@dataclass
class ObjectiveResult:
    lora_config: dict
    training_config: dict
    steps: list[ObjectiveStep]
    eval_snapshots: list[dict]
    total_time_s: float


def train(model: WhiteBoxModel, step_loss: Callable[[int], tuple[torch.Tensor, float | None,
                                                                float, dict]], *,
          max_steps: int, lr: float = 1e-4, warmup_steps: int = 10, max_grad_norm: float = 1.0,
          grad_accum: int = 1, eval_every: int = 0,
          eval_fn: Callable[[int], dict] | None = None, seed: int = 0,
          eval_at: Sequence[int] = ()) -> list:
    """The optimization loop shared by the three objectives: ``step_loss(micro)`` returns
    (total loss, representation term, KL term, info) for one micro-batch; AdamW over the LoRA
    parameters, linear warm-up then cosine (as ``ChatLoRATrainer``). ``eval_fn`` also runs at the
    steps in ``eval_at`` (a checkpoint taken mid-run, e.g. ST's epoch-3 ``L-beh``)."""
    torch.manual_seed(seed)
    params = [p for p in model.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    def lr_lambda(s: int) -> float:
        if s < warmup_steps:
            return s / max(warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(max_steps - warmup_steps,
                                                                       1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    steps, snaps = [], []
    if eval_fn is not None:
        snaps.append({"step": 0, **eval_fn(0)})
    model.model.train()
    micro = 0
    for s in range(1, max_steps + 1):
        tot, reps, kls, info = 0.0, [], [], {}
        for _ in range(grad_accum):
            loss, rep, kl, info = step_loss(micro)
            micro += 1
            (loss / grad_accum).backward()
            tot += float(loss.detach())
            reps.append(rep)
            kls.append(kl)
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad()
        r = [x for x in reps if x is not None]
        steps.append(ObjectiveStep(s, tot / grad_accum, sum(r) / len(r) if r else None,
                                   sum(kls) / len(kls), sched.get_last_lr()[0], info))
        if eval_fn is not None and ((eval_every and (s % eval_every == 0 or s == max_steps))
                                    or s in eval_at):
            model.model.eval()
            snaps.append({"step": s, **eval_fn(s)})
            model.model.train()
    model.model.eval()
    return [steps, snaps]


def micro_batches(items: Sequence[Any], seed: int) -> Iterator[Any]:
    """An endless seeded shuffle over ``items``."""
    g = torch.Generator().manual_seed(seed)
    while True:
        for i in torch.randperm(len(items), generator=g).tolist():
            yield items[i]


def qlora_model(repo: str, revision: str | None, **wb_kwargs) -> WhiteBoxModel:
    """The QLoRA base for the 27-30B models (§5.1): 4-bit NF4 weights, bf16 compute, prepared for
    k-bit training; adapters train in bf16 and are merged into a bf16 copy for evaluation."""
    from transformers import BitsAndBytesConfig

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    wb = WhiteBoxModel(repo, revision=revision, quantization_config=q,
                       torch_dtype=torch.bfloat16, **wb_kwargs)
    from peft import prepare_model_for_kbit_training

    wb._model = prepare_model_for_kbit_training(wb._model)  # noqa: SLF001
    return wb


def merge_bf16(base: WhiteBoxModel, adapter_dir: str, out_dir: str) -> str:
    """Load ``adapter_dir`` onto a bf16 (or the base's own precision) copy, merge, save: the
    evaluation path, so every adapter is evaluated at the base model's precision (QLoRA adapters
    merged and evaluated in bf16)."""
    from peft import PeftModel

    merged = PeftModel.from_pretrained(base.model, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    base.tokenizer.save_pretrained(out_dir)
    return out_dir


__all__ = ["ATTN", "EXPERT_PARAMS", "LINEAR_ATTN", "MLP", "ObjectiveResult", "ObjectiveStep",
           "adapter_disabled", "anchor_and_kl", "dpo_loss", "inject_lora", "kl_general",
           "lora_targets", "merge_bf16", "micro_batches", "qlora_model", "sequence_logprob",
           "train"]
