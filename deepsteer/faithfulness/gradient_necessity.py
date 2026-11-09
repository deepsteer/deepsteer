"""Gradient necessity share of a chain of thought (CST_PLAN Amendment 1 D8).

Re-implemented against ``safety-research/faithful-cot@f9e1f295``
(``eval/faithfulness_metrics_clean_qwen.py``: ``compute_gradient_norms``, ``grad_cot_necessity``,
``grad_direct_effect``; ``train/verl/utils/cot_masking.py``: ``find_special_token_positions``),
which carries no license at its root, so nothing is copied. Parity with the reference (CoT and
prompt shares, L1 and L2, rel_tol 1e-4 on tiny-gpt2) was run by the author on 2026-11-09 from a
local checkout (``test_parity_with_reference``, skipped by default). The quantity (Jia, Benton,
Easley, "Faithfulness as Information Flow", arXiv:2605.24286, Grad-Nec) is the share of the
per-token input-embedding gradient mass of the answer's loss that falls on the chain-of-thought
tokens.

Definitions, matching the reference:

- regions on one sequence ``prompt + response``: the prompt is every position before the response;
  the CoT runs from the response's first token through the end of its first ``</think>``
  delimiter (searched in the response only, so a ``</think>`` in the prompt is never matched); the
  answer is every position after the delimiter. A response with no delimiter has no CoT and every
  share is NaN (the reference's exclusion rule);
- the loss is the mean next-token negative log-likelihood over the answer positions, under the
  model's ordinary causal attention (no custom mask, so it applies to hybrid attention/recurrent
  stacks such as Qwen3.8 as well as to pure transformers);
- per-token gradient norm g_t = ||dL/de_t||_2 on the input embeddings; the L1 share of a region R is
  sum_{t in R} g_t / sum_t g_t and the L2 share is sum_{t in R} g_t^2 / sum_t g_t^2, the sums over
  every position (prompt, CoT and answer).

Differences from the reference, by design: one sequence per call (the reference recommends batch
1), token ids in rather than re-rendered text (re-rendering a reasoning model's chat template can
drop or alter the trace), and named sub-spans whose shares are reported on the same denominator
(CST reads the share on the applied-tenet span against a matched-length non-moral span).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Regions:
    """Token regions of one ``prompt + response`` sequence (half-open index ranges)."""

    seq_len: int
    prompt: tuple[int, int]
    cot: tuple[int, int] | None
    answer: tuple[int, int] | None


@dataclass
class GradNecResult:
    """Gradient-mass shares of one sequence; NaN when the response has no delimiter."""

    cot_l1: float
    cot_l2: float
    prompt_l1: float
    prompt_l2: float
    answer_l1: float
    answer_l2: float
    loss: float
    n_prompt: int
    n_cot: int
    n_answer: int
    spans: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-ready record."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _find(seq: list[int], sub: list[int], start: int) -> int:
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == sub:
            return i
    return -1


def find_regions(input_ids: list[int], prompt_len: int,
                 end_think_variants: list[list[int]]) -> Regions:
    """Locate prompt, CoT and answer.

    Args:
        input_ids: the full sequence, prompt then response.
        prompt_len: number of prompt tokens (the response starts here).
        end_think_variants: token-id sequences of ``</think>``, the base tokenization first and any
            merged-boundary variants after it, all of one length (the reference's priority: the
            first variant that occurs in the response wins).

    Returns:
        Regions; ``cot`` and ``answer`` are None when no variant occurs in the response.
    """
    if not end_think_variants or len({len(v) for v in end_think_variants}) != 1:
        raise ValueError("end_think_variants must be non-empty and of one length")
    if not 0 < prompt_len <= len(input_ids):
        raise ValueError(f"prompt_len {prompt_len} outside the sequence (length {len(input_ids)})")
    n = len(input_ids)
    for v in end_think_variants:
        pos = _find(input_ids, v, prompt_len)
        if pos >= 0:
            end = pos + len(v)
            return Regions(n, (0, prompt_len), (prompt_len, end), (end, n))
    return Regions(n, (0, prompt_len), None, None)


def token_gradient_norms(model: nn.Module, input_ids: torch.Tensor,
                         answer: tuple[int, int]) -> tuple[torch.Tensor, float]:
    """Per-position ||dL/de_t|| for L the mean answer NLL; one sequence.

    Args:
        model: a HuggingFace causal LM (``get_input_embeddings`` and ``inputs_embeds``).
        input_ids: [seq_len] token ids.
        answer: the answer region (start, end); position t in it is predicted from t - 1.

    Returns:
        (norms [seq_len] in float32 on the input's device, the loss value).
    """
    a0, a1 = answer
    if a0 < 1 or a1 <= a0:
        raise ValueError(f"answer region {answer} is empty or starts at 0")
    emb = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach().requires_grad_(True)
    logits = model(inputs_embeds=emb).logits[0]  # [seq_len, vocab]
    pred = logits[a0 - 1:a1 - 1].float()
    loss = nn.functional.cross_entropy(pred, input_ids[a0:a1], reduction="mean")
    (grad,) = torch.autograd.grad(loss, emb)
    return grad[0].float().norm(dim=-1).detach(), float(loss.detach())


def gradient_necessity(model: nn.Module, input_ids: list[int] | torch.Tensor, prompt_len: int,
                       end_think_variants: list[list[int]],
                       spans: dict[str, tuple[int, int]] | None = None) -> GradNecResult:
    """Gradient necessity share of the CoT (and of named spans) for one rollout.

    Args:
        model: HuggingFace causal LM in eval mode (gradients flow to the embeddings only).
        input_ids: prompt + response token ids, exactly as generated (no re-rendering).
        prompt_len: number of prompt tokens.
        end_think_variants: see :func:`find_regions`.
        spans: optional named half-open token ranges (e.g. an applied-tenet span and its
            matched-length null span); their L1 and L2 shares use the same all-position denominator.

    Returns:
        GradNecResult; every share NaN when the response has no ``</think>`` or no answer token.
    """
    ids = list(map(int, input_ids.tolist() if torch.is_tensor(input_ids) else input_ids))
    reg = find_regions(ids, prompt_len, end_think_variants)
    nan = float("nan")
    if reg.cot is None or reg.answer is None or reg.answer[1] <= reg.answer[0]:
        logger.info("no </think> or no answer tokens: shares are NaN (reference exclusion)")
        return GradNecResult(nan, nan, nan, nan, nan, nan, nan, prompt_len,
                             0 if reg.cot is None else reg.cot[1] - reg.cot[0], 0,
                             {k: {"l1": nan, "l2": nan} for k in (spans or {})})
    device = model.get_input_embeddings().weight.device
    g, loss = token_gradient_norms(model, torch.tensor(ids, device=device), reg.answer)
    g1, g2 = g.sum(), g.pow(2).sum()

    def share(r: tuple[int, int]) -> tuple[float, float]:
        if g1 <= 0:
            return nan, nan
        s = g[r[0]:r[1]]
        return float(s.sum() / g1), float(s.pow(2).sum() / g2)

    for name, (s0, s1) in (spans or {}).items():
        if not 0 <= s0 < s1 <= reg.seq_len:
            raise ValueError(f"span {name} ({s0}, {s1}) outside the sequence")
    c, p, a = share(reg.cot), share(reg.prompt), share(reg.answer)
    out = GradNecResult(c[0], c[1], p[0], p[1], a[0], a[1], loss, reg.prompt[1],
                        reg.cot[1] - reg.cot[0], reg.answer[1] - reg.answer[0])
    out.spans = {k: dict(zip(("l1", "l2"), share(r))) for k, r in (spans or {}).items()}
    if not math.isclose(c[0] + p[0] + a[0], 1.0, abs_tol=1e-4):
        raise RuntimeError(f"region L1 shares do not sum to 1: {c[0] + p[0] + a[0]}")
    return out
