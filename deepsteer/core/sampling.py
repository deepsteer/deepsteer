"""Seeded, pure-temperature sampling for local HuggingFace models (KDG sampler v2 contract).

One seed per item (crc32 of its id) and all of an item's rollouts drawn in one seeded call, so
no two items share a random stream by rollout index (ANOMALIES KDG-A23). Sampling is pure
temperature: the model's ``generation_config`` warpers (top_p, top_k, repetition penalty) are
disabled explicitly rather than inherited.
"""

from __future__ import annotations

import zlib
from typing import Any

import torch

SAMPLER_VERSION = 2


def item_seed(item_id: str, base: int = 0) -> int:
    """A stable per-item seed: ``base + crc32(item_id) mod 2**31``."""
    return base + zlib.crc32(item_id.encode()) % (2**31)


@torch.no_grad()
def sample_rollouts(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    *,
    n: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> list[list[int]]:
    """``n`` sampled continuations of one prompt, as new-token id lists (EOS and padding cut).

    All ``n`` rows are the same prompt (no padding), drawn in one ``generate`` call after
    ``torch.manual_seed(seed)``; row k is rollout k.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids] * n, device=device)
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    eos_set: set[int] = set()
    # stop ids from the tokenizer and from the generation config (Gemma stops on <end_of_turn>)
    for e in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if e is not None:
            eos_set |= set(e) if isinstance(e, (list, tuple)) else {e}
    rows = []
    for row in out[:, len(input_ids):].tolist():
        cut = next((i for i, t in enumerate(row) if t in eos_set), len(row))
        rows.append(row[:cut])
    return rows
