#!/usr/bin/env python3
"""WS3: Direction ablation experiments.

For each foundation direction at each target layer, project out the direction
from hidden states during the forward pass and measure:
  - Change in log-probability for foundation-specific continuations
  - Cross-foundation specificity (does ablating Care affect Loyalty prompts?)
  - General perplexity increase

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/direction_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    compute_mean_diff_directions,
    load_model_and_collect_activations,
    load_probe_directions,
    ensure_output_dirs,
    OUTPUT_DIR,
)
from leace_directions import compute_leace_directions
from causal_eval_prompts import CausalEvalDataset


def ablate_direction(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Project out a direction from hidden states: h - (h·d)d."""
    proj = (hidden_states @ direction).unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)
    return hidden_states - proj


def measure_continuation_logprobs(
    model,
    tokenizer,
    prompt: str,
    continuations: list[str],
    device: str,
) -> dict[str, float]:
    """Measure log-probability of each continuation token given the prompt."""
    results = {}
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        logprobs = F.log_softmax(out.logits[0, -1], dim=-1)

    for cont_text in continuations:
        cont_tokens = tokenizer.encode(cont_text, add_special_tokens=False)
        if cont_tokens:
            token_id = cont_tokens[0]
            results[cont_text] = float(logprobs[token_id])
    return results


def run_ablation(
    model,
    tokenizer,
    eval_dataset: CausalEvalDataset,
    directions: dict[str, dict[int, np.ndarray]],
    target_layers: list[int],
    device: str,
    method_name: str = "mean_diff",
) -> dict:
    """Run ablation for each foundation direction at target layers.

    For each (ablated_foundation, layer) pair, measure the change in
    log-probability for all prompts grouped by their target foundation.
    """
    results: dict[str, dict] = {}

    for ablated_fv in FOUNDATION_ORDER:
        if ablated_fv not in directions:
            continue
        results[ablated_fv] = {}

        for layer in target_layers:
            direction = directions[ablated_fv].get(layer)
            if direction is None:
                continue

            d_tensor = torch.from_numpy(direction).to(device=device, dtype=model.dtype)
            layer_module = model.model.layers[layer]

            # Measure baseline and ablated for all prompts
            deltas_by_target: dict[str, list[float]] = {fv: [] for fv in FOUNDATION_ORDER}

            for prompt_data in eval_dataset.prompts:
                prompt = prompt_data.prompt
                target_fv = prompt_data.target_foundation
                target_conts = [c["text"] for c in prompt_data.continuations if c.get("is_target")]
                if not target_conts:
                    continue

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                # Baseline
                with torch.no_grad():
                    baseline_out = model(**inputs)
                    bl_logprobs = F.log_softmax(baseline_out.logits[0, -1], dim=-1)

                # Ablated
                def _hook(module, input, output, d=d_tensor):
                    if isinstance(output, tuple):
                        h = output[0]
                        proj = (h @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
                        return (h - proj,) + output[1:]
                    proj = (output @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
                    return output - proj

                handle = layer_module.register_forward_hook(_hook)
                with torch.no_grad():
                    ablated_out = model(**inputs)
                    ab_logprobs = F.log_softmax(ablated_out.logits[0, -1], dim=-1)
                handle.remove()

                # Measure delta for target continuations
                for cont_text in target_conts:
                    cont_tokens = tokenizer.encode(cont_text, add_special_tokens=False)
                    if not cont_tokens:
                        continue
                    token_id = cont_tokens[0]
                    delta = float(ab_logprobs[token_id] - bl_logprobs[token_id])
                    deltas_by_target[target_fv].append(delta)

            # Summarize: mean delta for on-target vs off-target prompts
            on_target_deltas = deltas_by_target.get(ablated_fv, [])
            off_target_deltas = [
                d for fv, ds in deltas_by_target.items()
                if fv != ablated_fv for d in ds
            ]

            results[ablated_fv][layer] = {
                "on_target_mean_delta": round(float(np.mean(on_target_deltas)), 6) if on_target_deltas else 0.0,
                "off_target_mean_delta": round(float(np.mean(off_target_deltas)), 6) if off_target_deltas else 0.0,
                "specificity": round(
                    float(np.mean(on_target_deltas)) - float(np.mean(off_target_deltas)), 6
                ) if on_target_deltas and off_target_deltas else 0.0,
                "n_on_target": len(on_target_deltas),
                "n_off_target": len(off_target_deltas),
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="WS3: Direction ablation.")
    parser.add_argument("--directions", choices=["mean_diff", "leace", "probe_weight"],
                        default="mean_diff")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--target-layers", default="4,6,8,10,12,14")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-per-foundation", type=int, default=200)
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS3: Direction Ablation Experiments")
    print(f"{'='*60}")

    # Load eval prompts
    eval_path = OUTPUT_DIR / "causal_eval_prompts.json"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found. Run causal_eval_prompts.py first.")
        return
    eval_dataset = CausalEvalDataset.from_json(eval_path)
    print(f"Loaded {len(eval_dataset.prompts)} evaluation prompts")

    target_layers = [int(x) for x in args.target_layers.split(",")]

    # Determine device
    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Device: {device}")

    # Get directions
    if args.directions == "probe_weight":
        directions = load_probe_directions(args.probe_directions)
        method_name = "probe_weight"
    else:
        # Need to load model + collect activations for data-dependent methods
        all_train, _, _, n_layers, foundation_indices = (
            load_model_and_collect_activations(
                model_name=args.model,
                device=device,
                target_per_foundation=args.target_per_foundation,
                collect_test=False,
            )
        )
        if args.directions == "leace":
            directions = compute_leace_directions(all_train, n_layers, foundation_indices)
            method_name = "leace"
        else:
            directions = compute_mean_diff_directions(all_train, n_layers, foundation_indices)
            method_name = "mean_diff"

    print(f"Direction method: {method_name}")
    print(f"Target layers: {target_layers}")

    # Load the raw model for ablation (need forward hooks)
    print(f"\nLoading model for ablation: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(device).eval()

    print("Running ablation experiments...")
    results = run_ablation(
        model, tokenizer, eval_dataset, directions, target_layers, device, method_name,
    )

    # Print summary
    print(f"\n--- Ablation Results ({method_name}) ---")
    print(f"{'Ablated':<14s}  {'Layer':>5s}  {'On-target Δ':>11s}  {'Off-target Δ':>12s}  {'Specificity':>11s}")
    print("-" * 60)
    for fv in FOUNDATION_ORDER:
        if fv not in results:
            continue
        for layer in target_layers:
            r = results[fv].get(layer, {})
            if not r:
                continue
            print(f"  {FOUNDATION_SHORT[fv]:<12s}  {layer:>5d}  "
                  f"{r['on_target_mean_delta']:>11.4f}  "
                  f"{r['off_target_mean_delta']:>12.4f}  "
                  f"{r['specificity']:>11.4f}")

    # Save
    out_data = {
        "analysis": "direction_ablation",
        "method": method_name,
        "model": args.model,
        "target_layers": target_layers,
        "n_prompts": len(eval_dataset.prompts),
        "results": results,
    }
    out_path = output_dir / f"direction_ablation_{method_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
