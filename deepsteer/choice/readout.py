"""Letter-mass readout at the answer position (KDG readout v2: positions from the attention mask).

Lifted in behaviour from ``papers/kdg_panel/scripts/kdg_pod_lib.py`` (``mask_positions`` and
``ModelWrapper.raw_next_logprobs``); the KDG harness is re-pointed here only after its golden tests
land (MB_PLAN §6).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

READOUT_VERSION = 2


def mask_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids from a (left-padded) attention mask, the way ``generate()`` derives them."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def label_token_ids(tokenizer: Any, labels: str) -> list[int]:
    """One token id per label surface (the first generated token, no leading space).

    Raises:
        RuntimeError: a label is not a single token.
    """
    out = []
    for lab in labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"label {lab!r} is not a single token: {ids}")
        out.append(int(ids[0]))
    return out


@torch.no_grad()
def next_token_logprobs(
    model: Any, id_lists: list[list[int]], *, pad_id: int, batch_size: int = 8
) -> np.ndarray:
    """Next-token log-probabilities (float32, ``[n, vocab]``) after each id list.

    Rows are left-padded in batches; positions come from the attention mask (readout v2), so a
    row reads the same with or without padding.
    """
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(id_lists), batch_size):
        chunk = id_lists[i : i + batch_size]
        width = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, x in enumerate(chunk):
            ids[r, width - len(x):] = torch.tensor(x)
            mask[r, width - len(x):] = 1
        ids, mask = ids.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask,
                       position_ids=mask_positions(mask)).logits[:, -1, :]
        out.append(torch.log_softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def label_token_sets(tokenizer: Any, labels: str) -> list[list[int]]:
    """Every single-token surface of each label in the vocabulary (bare, SentencePiece ``▁``-
    prefixed, byte-BPE ``Ġ``-prefixed): the first generated token may carry a leading-space marker
    or not, so the readout sums the mass over all of them.

    Raises:
        RuntimeError: a label has no single-token surface.
    """
    vocab = tokenizer.get_vocab()
    out = []
    for lab in labels:
        ids = sorted({vocab[s] for s in (lab, "\u2581" + lab, "\u0120" + lab) if s in vocab})
        if not ids:
            raise RuntimeError(f"label {lab!r} has no single-token surface")
        out.append(ids)
    return out


def letter_mass(
    logprobs: np.ndarray, token_sets: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the label distribution renormalized over the labels (``[n, k]``, each label's mass
    summed over its token surfaces), and the total probability mass the labels carry (``[n]``)."""
    p = np.exp(logprobs.astype(np.float64))
    per = np.stack([p[:, ids].sum(axis=1) for ids in token_sets], axis=1)
    total = per.sum(axis=1)
    return per / total[:, None], total
