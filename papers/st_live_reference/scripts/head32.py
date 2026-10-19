"""The final norm and LM head in fp32 (ST G1 item 4, 2026-10-19), shared by the screen driver and
``stack_read.py`` so both stacks switch the head the same way.

Readout logits come out of a bf16 LM head; at the dose-0 letter position the batch-size jitter of
the relaunch and rider pods sat at ~0.25 nats on both dense models, the size of one bf16 step of
a logit of magnitude 32-64 (ANOMALIES ST-A3). ``set_head_fp32(model, True)`` makes the final norm
and the LM head compute in fp32 from fp32 copies of their weights (the parameters themselves are
untouched, so tied input embeddings keep their dtype and every block upstream runs as before);
``False`` restores the original forwards.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_SKIP = ("layers", "vision", "visual", "audio", "multi_modal", "mm_projector")


def final_norm(model) -> tuple[str, Any]:
    """The text decoder's final norm: the one module named ``norm`` (or ``*.norm``) outside the
    blocks and any vision or audio tower. Exactly one, else the model is not supported."""
    hits = [(n, m) for n, m in model.named_modules()
            if re.search(r"(^|\.)norm$", n) and not any(s in n for s in _SKIP)]
    if len(hits) != 1:
        raise ValueError(f"expected one final norm, found {[n for n, _ in hits]}")
    return hits[0]


def set_head_fp32(model, on: bool) -> dict:
    """Switch the final norm and LM head of an HF CausalLM / ImageTextToText model to fp32 (or
    back). Returns a record of what was switched."""
    import torch

    state = model.__dict__.setdefault("_st_head32", {})
    name, norm = final_norm(model)
    head = model.get_output_embeddings()
    if on and not state:
        n32 = copy.deepcopy(norm).float()
        w32 = head.weight.detach().float().clone()
        hb = getattr(head, "bias", None)
        b32 = hb.detach().float().clone() if hb is not None else None
        state["orig"] = (norm.forward, head.forward)
        norm.forward = lambda x, *a, **k: n32(x.float(), *a, **k)
        head.forward = lambda x: torch.nn.functional.linear(x.float(), w32, b32)
        state["keep"] = (n32, w32, b32)
    elif not on and state:
        norm.forward, head.forward = state["orig"]
        state.clear()
    return {"head": "fp32" if on else "bf16", "final_norm": name,
            "lm_head_shape": list(head.weight.shape)}
