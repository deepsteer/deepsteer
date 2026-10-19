#!/usr/bin/env python3
"""One stack's letter readouts on a fixed set of rows, for the stack-parity read (ST G1 rider pod 2,
author 2026-10-19; Amendment 9). The same script runs under the HF stack of record (torch 2.6) and
in the vLLM venv (torch 2.13), so the stack is the only difference between the two reads.

    python stack_read.py --repo R --revision C --rows ROWS.json --out OUT.npz [--batch 16]

ROWS.json: ``{"ids": [[...], ...], "letter_ids": [A..E token ids]}`` (written by the screen
driver from its own dose-0 renders). Per head (bf16 as loaded, fp32 via ``head32``): each row
alone (the readout of record's single-row read), each row in a batch of ``--batch`` identical
copies (no padding; the Amendment 6 replicate floor), and each row one position early (the
negative control). Eager attention, bf16 weights, the transformers default experts path of the
stack (recorded). Torch-only besides transformers, so it runs in either venv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import head32  # noqa: E402


def load(repo: str, revision: str | None, experts: str | None = None):
    """The model as the screen loads it: bf16, eager attention, the text-tower class rule of
    ``deepsteer.core.model_interface._auto_model_class`` (restated here so this runs without the
    deepsteer package)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

    cfg = AutoConfig.from_pretrained(repo, revision=revision)
    cls = (AutoModelForCausalLM if type(cfg) in MODEL_FOR_CAUSAL_LM_MAPPING
           else AutoModelForImageTextToText)
    kw = {"experts_implementation": experts} if experts else {}
    m = cls.from_pretrained(repo, revision=revision, dtype=torch.bfloat16,
                            attn_implementation="eager", **kw)
    m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return m


def letters(m, rows: list[list[int]], lids: list[int], copies: int = 1) -> np.ndarray:
    """Letter log-probs at the last position: each row alone (``copies`` 1) or the first of
    ``copies`` identical copies read in one batch."""
    import torch

    out = []
    for x in rows:
        ids = torch.tensor([x] * copies, device=m.device)
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=torch.ones_like(ids),
                  position_ids=torch.arange(len(x), device=m.device).expand(copies, -1),
                  logits_to_keep=1)
        out.append(torch.log_softmax(o.logits[0, -1].float(), -1)[lids].cpu().numpy())
    return np.stack(out).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import torch
    import transformers

    spec = json.loads(a.rows.read_text())
    rows, lids = spec["ids"], spec["letter_ids"]
    t0 = time.time()
    m = load(a.repo, a.revision)
    res: dict = {}
    for head in ("bf16", "fp32"):
        head32.set_head_fp32(m, head == "fp32")
        res[f"alone_{head}"] = letters(m, rows, lids)
        res[f"replicate_{head}"] = letters(m, rows, lids, a.batch)
        res[f"off_{head}"] = letters(m, [x[:-1] for x in rows], lids)
    meta = {"torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "experts_implementation": getattr(
                m.config, "_experts_implementation_internal", None),
            "attn_implementation": m.config._attn_implementation, "n_rows": len(rows),
            # an MXFP4 load (no dequantization) on one stack is part of what parity measures
            "param_dtypes": sorted({str(p.dtype) for p in m.parameters()}),
            "quantization_config": getattr(m.config, "quantization_config", None) and str(
                m.config.quantization_config)[:200],
            "batch": a.batch, "seconds": time.time() - t0, "repo": a.repo,
            "revision": a.revision}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **res)
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f">> stack read {a.repo}: torch {meta['torch']}, experts "
          f"{meta['experts_implementation']}, {len(rows)} rows, {meta['seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
