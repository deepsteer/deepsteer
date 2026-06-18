#!/usr/bin/env python3
"""Sprint 6.4: ART-SFT (and control-SFT) training.

Fine-tunes OLMo-3 base into an instruct model with LoRA on chat-format SFT data,
optionally adding the ablation-resistance (ART) auxiliary loss that trains the
model to route moral-content generation through its moral subspace (so a future
Heretic-style ablation pays a quality cost).

Two conditions, identical except the ART term:
  * ART-SFT     : ``--art-lambda 0.01`` (or calibrated)  -> ART active
  * control-SFT : ``--art-lambda 0.0``                   -> plain SFT

Sprint 5 found the BASE/pretraining moral subspace is the generation-functional
one, so ART targets the base directions
(``…/olmo3_base/exp1_probe_directions.npz``) by default.

OLMo-3 base has no chat template; we borrow the Instruct variant's (same vocab),
matching how the model is actually used. Saves the LoRA adapter, the training
record, and (default) the merged model for downstream eval (Sprint 7).

Usage:
    python papers/6_ablation_resistance/scripts/art_sft.py \
        --model allenai/Olmo-3-1025-7B \
        --data papers/6_ablation_resistance/data/sft_mix.jsonl \
        --moral-directions papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz \
        --art-lambda 0.01 --art-calibrate \
        --output-dir papers/6_ablation_resistance/outputs/art_sft --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MORAL_NPZ = "papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz"


def resolve_chat_template(base_tokenizer, chat_template_from: str | None) -> str | None:
    """Return a chat template for a base model that lacks one.

    Prefers the base tokenizer's own template; else borrows it from
    ``chat_template_from`` (the Instruct variant, same vocab).
    """
    tmpl = getattr(base_tokenizer, "chat_template", None)
    if tmpl:
        return tmpl
    if not chat_template_from:
        return None
    from transformers import AutoTokenizer
    other = AutoTokenizer.from_pretrained(chat_template_from)
    tmpl = getattr(other, "chat_template", None)
    if not tmpl:
        logger.warning("No chat template on %s either; trainer default will be used.",
                       chat_template_from)
    return tmpl


def main() -> None:
    ap = argparse.ArgumentParser(description="ART-SFT / control-SFT training.")
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data", required=True, help="Chat-format JSONL ({'messages':[...]}).")
    ap.add_argument("--moral-directions", default=_DEFAULT_MORAL_NPZ,
                    help="Base foundation directions npz (targeted by ART).")
    ap.add_argument("--direction-kind", choices=["probe", "meandiff"], default="probe")
    ap.add_argument("--art-lambda", type=float, default=0.01,
                    help="ART coefficient; 0.0 = control (plain SFT).")
    ap.add_argument("--art-calibrate", action="store_true",
                    help="Calibrate λ so |ART| ≈ target-ratio × L_sft on the first batch.")
    ap.add_argument("--art-target-ratio", type=float, default=0.10)
    ap.add_argument("--art-max-lambda", type=float, default=1.0,
                    help="Cap on the calibrated λ.")
    ap.add_argument("--art-target-gap", type=float, default=0.3,
                    help="Hinge target: drive L_ablated-L_sft up to this (nats), "
                         "then stop. Bounds dependency so the objective can't run away.")
    ap.add_argument("--art-gap-source", choices=["moral_pool", "batch"], default="moral_pool",
                    help="moral_pool: measure the ART gap on concentrated moral text "
                         "(default; avoids the dilution that made 'batch' a no-op). "
                         "batch: gap on the mixed SFT batch (the v1 behaviour).")
    ap.add_argument("--art-moral-pool", default=None,
                    help="File of moral texts (one per line) for the gap pool; "
                         "default = the probing dataset's train moral sentences.")
    ap.add_argument("--art-layers", default=None,
                    help="Comma-separated layer subset for ART; default all complete layers.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    # chat template
    ap.add_argument("--chat-template-from", default="allenai/Olmo-3-7B-Instruct",
                    help="Borrow the chat template from this model if base lacks one.")
    # LoRA / training
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--target-modules", default="q_proj,v_proj")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-monitor", action="store_true",
                    help="Snapshot moral probing geometry every eval (slow).")
    ap.add_argument("--no-merge", dest="merge", action="store_false", default=True,
                    help="Skip merging LoRA into the base and saving the merged model.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.steering import (
        AblationResistanceSteering, ChatLoRATrainer, ProbeMonitor, load_chat_jsonl,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_art = args.art_lambda > 0.0
    condition = "art_sft" if is_art else "control_sft"

    conversations = load_chat_jsonl(args.data)
    print(f"Loaded {len(conversations)} conversations from {args.data}")

    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device,
                          access_tier=AccessTier.WEIGHTS, revision=args.revision)
    print(f"Loaded {args.model} ({model.info.n_layers}L) in {time.time()-t0:.1f}s; "
          f"condition={condition}, art_lambda={args.art_lambda}")

    chat_template = resolve_chat_template(model.tokenizer, args.chat_template_from)

    art = None
    if is_art:
        layers = [int(x) for x in args.art_layers.split(",")] if args.art_layers else None
        moral_pool = None
        if args.art_gap_source == "moral_pool":
            if args.art_moral_pool:
                moral_pool = [ln.strip() for ln in open(args.art_moral_pool) if ln.strip()]
            else:
                from deepsteer.datasets.pipeline import build_probing_dataset
                ds = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
                moral_pool = [p.moral for p in ds.train]
        art = AblationResistanceSteering(
            args.moral_directions, coefficient=args.art_lambda,
            max_coefficient=args.art_max_lambda, target_gap=args.art_target_gap,
            direction_kind=args.direction_kind, target_layers=layers,
            moral_pool_texts=moral_pool,
        )
        print(f"ART: {args.moral_directions} ({args.direction_kind}), λ={args.art_lambda}, "
              f"target_gap={args.art_target_gap}, gap_source={args.art_gap_source}"
              + (f" ({len(moral_pool)} pool texts)" if moral_pool else "")
              + f", calibrate={args.art_calibrate}")

    eval_callbacks = []
    monitor = None
    if args.probe_monitor:
        monitor = ProbeMonitor(model)
        eval_callbacks.append(lambda tr, step: {
            "probe_peak_acc": monitor.snapshot(step).peak_accuracy})

    trainer = ChatLoRATrainer(
        model, conversations,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lr=args.lr, batch_size=args.batch_size, seq_len=args.seq_len,
        max_steps=args.max_steps, eval_every=args.eval_every,
        warmup_steps=args.warmup_steps, seed=args.seed,
        chat_template=chat_template, eval_callbacks=eval_callbacks,
        art_steering=art, art_calibrate=args.art_calibrate,
        art_target_ratio=args.art_target_ratio,
    )
    result = trainer.train(experiment_id=condition, corpus_name=Path(args.data).stem)

    # ---- save adapter + result ----
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    model.model.save_pretrained(adapter_dir)
    model.tokenizer.save_pretrained(adapter_dir)

    payload = {
        "analysis": "art_sft",
        "condition": condition,
        "model": args.model,
        "data": args.data,
        "n_conversations": len(conversations),
        "moral_directions": args.moral_directions if is_art else None,
        "direction_kind": args.direction_kind,
        "art_lambda_final": (art.coefficient if art else 0.0),
        "art_calibrated": args.art_calibrate,
        "result": result.to_dict(),
    }
    with open(out / "art_sft.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    if monitor is not None:
        monitor.save(out / "probe_monitor.json")

    # ---- merge LoRA -> base and save the deployable model (Sprint 7 input) ----
    if args.merge:
        merged = model.model.merge_and_unload()
        merged_dir = out / "merged_model"
        merged_dir.mkdir(exist_ok=True)
        merged.save_pretrained(merged_dir)
        model.tokenizer.save_pretrained(merged_dir)
        if chat_template:  # persist the borrowed template so eval uses chat format
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(merged_dir)
            tok.chat_template = chat_template
            tok.save_pretrained(merged_dir)
        print(f"  merged model -> {merged_dir}")

    model.release()

    # ---- report ----
    steps = result.steps
    last = steps[-1] if steps else None
    print(f"\nWrote {out/'art_sft.json'} ({condition}, {len(steps)} steps)")
    if last:
        print(f"  final: sft={last.loss:.4f}"
              + (f", art={last.art_loss:.4f}, gap={last.art_gap:.4f}" if is_art else ""))
    if is_art and steps:
        gaps = [s.art_gap for s in steps if s.art_gap is not None]
        if gaps:
            print(f"  ART gap: first={gaps[0]:+.4f} -> last={gaps[-1]:+.4f} "
                  f"(want it to GROW: ablation hurts more over training)")


if __name__ == "__main__":
    main()
