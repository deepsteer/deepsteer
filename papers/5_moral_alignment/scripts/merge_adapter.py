#!/usr/bin/env python3
"""Merge a LoRA adapter into a base checkpoint and save a standalone model.

Stage 2 / S0: the coupling (and control) continued-pretrain adapters must be
merged into the base BEFORE SFT, because ``art_sft.py`` injects a fresh LoRA on
its ``--model`` (it cannot stack on a pre-existing adapter). Same mechanism as
``eval_pipeline.materialize_merged``, exposed as a small CLI.

Usage:
    python papers/5_moral_alignment/scripts/merge_adapter.py \
        --base-model allenai/Olmo-3-1025-7B --revision stage3-step11921 \
        --adapter OUT/intervention_stage1/coupling_r64_qv_mlp/adapter \
        --dest OUT/intervention_stage2/coupled_cpt_merged
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into a base checkpoint.")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.revision, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(dest)
    # Tokenizer travels with the adapter (base tokenizer; no chat template yet).
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(dest)
    print(f"Merged {args.base_model}@{args.revision} + {args.adapter} -> {dest}")


if __name__ == "__main__":
    main()
