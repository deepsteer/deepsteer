#!/usr/bin/env python3
"""GPC Amendment 3 B3.6 (draft): the summed adapter, P-ctx's update plus C1's, no training.

    /opt/homebrew/bin/python3.13 z_summed_adapter.py [OUT_DIR]

Both are LoRA r 16, alpha 32 on the same modules, so their sum is exact as one LoRA of rank 32 and
alpha 64 (the same alpha over rank, 2): A = [A_pctx; A_c1] stacked on the rank axis, B = [B_pctx,
B_c1] side by side, so B A = B_pctx A_pctx + B_c1 A_c1 and the served update is their sum. The
tensors are copied, not recomputed, so no rounding enters. The check: per module, the summed
adapter's update against the two updates added, in float32 on every module (max abs difference and
relative to the update's scale). Serving is the pod's check (vLLM's `max_lora_rank` 32, served parity).
Writes ``OUT_DIR/adapter_model.safetensors``, its config and ``sum_check.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PCTX = ROOT / "papers/cst_constitutional_steering/outputs/pressure/adapters/p_ctx/1.94053e+06"
C1 = HERE.parent / "outputs/gpc1_v1/adapters/c1/612732"
OUT = HERE.parent / "outputs/gpc4_prep/adapters/sum_pctx_c1"


def build(out: Path = OUT) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    cfg = [json.loads((d / "adapter_config.json").read_text()) for d in (PCTX, C1)]
    for k in ("r", "lora_alpha", "use_rslora"):
        if cfg[0].get(k) != cfg[1].get(k):
            raise SystemExit(f"BAIL: the adapters differ on {k}")
    if set(cfg[0]["target_modules"]) != set(cfg[1]["target_modules"]):
        raise SystemExit("BAIL: the adapters target different modules")
    a, b = load_file(str(PCTX / "adapter_model.safetensors")), load_file(
        str(C1 / "adapter_model.safetensors"))
    if set(a) != set(b):
        raise SystemExit("BAIL: the adapters hold different tensors")
    r = cfg[0]["r"]
    w = {}
    for k in a:
        if "lora_A" in k:
            w[k] = torch.cat([a[k], b[k]], dim=0)
        elif "lora_B" in k:
            w[k] = torch.cat([a[k], b[k]], dim=1)
        else:
            raise SystemExit(f"BAIL: an unexpected tensor {k}")
    out.mkdir(parents=True, exist_ok=True)
    save_file(w, str(out / "adapter_model.safetensors"))
    new = dict(cfg[0], r=2 * r, lora_alpha=2 * cfg[0]["lora_alpha"])
    (out / "adapter_config.json").write_text(json.dumps(new, indent=1))
    worst, worst_rel, n = 0.0, 0.0, 0
    for k in a:
        if "lora_A" not in k:
            continue
        kb = k.replace("lora_A", "lora_B")
        s = cfg[0]["lora_alpha"] / r
        d_sum = (new["lora_alpha"] / new["r"]) * (w[kb].float() @ w[k].float())
        d_add = s * (a[kb].float() @ a[k].float()) + s * (b[kb].float() @ b[k].float())
        diff = float((d_sum - d_add).abs().max())
        scale = float(d_add.abs().max()) or 1.0
        worst, worst_rel, n = max(worst, diff), max(worst_rel, diff / scale), n + 1
    rec = {"rule": "GPC Amendment 3 B3.6 (draft): P-ctx's update plus C1's, no training",
           "p_ctx": str(PCTX.relative_to(ROOT)), "c1": str(C1.relative_to(ROOT)),
           "rank": new["r"], "alpha": new["lora_alpha"], "scale": new["lora_alpha"] / new["r"],
           "modules": n, "max_abs_diff": worst, "max_rel_diff": worst_rel,
           "exact": worst_rel < 1e-5,
           "note": "the tensors are concatenated copies; the float32 check bounds the product's "
                   "rounding only. Serving (max_lora_rank 32, served parity) is the pod's check."}
    (out / "sum_check.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT), indent=1))
