#!/usr/bin/env python3
"""HF generation step time of an MoE model per experts path and batch (ST G1 rider pod 2, author
2026-10-19; ANOMALIES ST-A6). One path per process:

    python moe_probe.py --repo R --revision C --prompts P.json --out OUT.json \\
        --experts eager|batched_mm|grouped_mm|kernels [--batches 1,8,33,132]

The rider pod's probe found ~20 ms per sequence-step at every batch on the HF stack of record:
transformers 5.17 asks for ``grouped_mm`` experts, which needs torch >= 2.9 on an A100, so torch
2.6 falls back to the per-expert ``eager`` loop. This measures each lever: ``eager`` and
``batched_mm`` on torch 2.6, ``grouped_mm`` in the vLLM venv (torch 2.13), and the hub
``MegaBlocksMoeMLP`` kernel (``kernels``, ``use_kernels=True``). Attention eager throughout (the
hooked steering arms' path), bf16, the warmed two-length decode slope of
``engine_probes.hf_batch_probe``. A path that fails to load or run is recorded with its error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine_probes as EP  # noqa: E402

CAP = 1536


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experts", required=True,
                    choices=("eager", "batched_mm", "grouped_mm", "kernels"))
    ap.add_argument("--batches", default="1,8,33,132")
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--dequantize", action="store_true",
                    help="MXFP4 experts dequantized to bf16 (the registry's hf pin)")
    a = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rec: dict = {"experts": a.experts, "torch": torch.__version__,
                 "transformers": transformers.__version__, "probes": []}
    t0 = time.time()
    try:
        kw = ({"use_kernels": True} if a.experts == "kernels"
              else {"experts_implementation": a.experts})
        if a.dequantize:
            from transformers import Mxfp4Config

            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        m = AutoModelForCausalLM.from_pretrained(a.repo, revision=a.revision,
                                                 dtype=torch.bfloat16,
                                                 attn_implementation="eager", **kw)
        m = m.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        rec["experts_in_effect"] = getattr(m.config, "_experts_implementation_internal", None)
        rec["load_seconds"] = time.time() - t0
    except Exception as e:  # noqa: BLE001  (a lever that does not load is the probe's answer)
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        a.out.write_text(json.dumps(rec, indent=1))
        print(f">> moe probe {a.experts}: failed to load ({rec['error'][:120]})", flush=True)
        return 0
    tok = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    wrap = SimpleNamespace(model=m, tok=tok, torch=torch, device=m.device)
    prompts = json.loads(a.prompts.read_text())["ids"]
    for b in (int(x) for x in a.batches.split(",")):
        try:
            p = EP.hf_batch_probe(wrap, prompts, b, cap=CAP, decode_steps=a.decode_steps)
        except Exception as e:  # noqa: BLE001
            p = {"batch": b, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        if p.get("seconds_per_step"):
            p["ms_per_sequence_step"] = 1000 * p["seconds_per_step"] / b
        rec["probes"].append(p)
        a.out.write_text(json.dumps(rec, indent=1))  # after every batch
        print(f">> moe probe {a.experts} b={b}: "
              f"{p.get('seconds_per_step') or p.get('error') or p.get('oom')}", flush=True)
    rec["seconds"] = time.time() - t0
    a.out.write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
