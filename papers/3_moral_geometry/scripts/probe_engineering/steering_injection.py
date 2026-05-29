#!/usr/bin/env python3
"""WS3: Steering vector injection experiments.

For each foundation direction at each target layer, add a scaled steering
vector to hidden states and measure foundation-specific amplification,
dose response, and coherence preservation.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/steering_injection.py
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

ALPHA_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]


def run_injection(
    model,
    tokenizer,
    eval_dataset: CausalEvalDataset,
    directions: dict[str, dict[int, np.ndarray]],
    target_layers: list[int],
    alphas: list[float],
    device: str,
) -> dict:
    """Run steering injection for each foundation direction.

    For each (foundation, layer, alpha) triple, add alpha * direction to
    hidden states and measure log-probability shift for target continuations.
    """
    results: dict[str, dict] = {}

    for injected_fv in FOUNDATION_ORDER:
        if injected_fv not in directions:
            continue
        results[injected_fv] = {}

        for layer in target_layers:
            direction = directions[injected_fv].get(layer)
            if direction is None:
                continue
            d_tensor = torch.from_numpy(direction).to(device=device, dtype=model.dtype)
            layer_module = model.model.layers[layer]

            alpha_results = {}
            for alpha in alphas:
                deltas_on_target: list[float] = []
                deltas_off_target: list[float] = []

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

                    # Injected
                    def _hook(module, input, output, d=d_tensor, a=alpha):
                        if isinstance(output, tuple):
                            h = output[0]
                            return (h + a * d.unsqueeze(0).unsqueeze(0),) + output[1:]
                        return output + a * d.unsqueeze(0).unsqueeze(0)

                    handle = layer_module.register_forward_hook(_hook)
                    with torch.no_grad():
                        injected_out = model(**inputs)
                        inj_logprobs = F.log_softmax(injected_out.logits[0, -1], dim=-1)
                    handle.remove()

                    for cont_text in target_conts:
                        cont_tokens = tokenizer.encode(cont_text, add_special_tokens=False)
                        if not cont_tokens:
                            continue
                        token_id = cont_tokens[0]
                        delta = float(inj_logprobs[token_id] - bl_logprobs[token_id])
                        if target_fv == injected_fv:
                            deltas_on_target.append(delta)
                        else:
                            deltas_off_target.append(delta)

                alpha_results[alpha] = {
                    "on_target_mean_delta": round(float(np.mean(deltas_on_target)), 6) if deltas_on_target else 0.0,
                    "off_target_mean_delta": round(float(np.mean(deltas_off_target)), 6) if deltas_off_target else 0.0,
                    "specificity": round(
                        float(np.mean(deltas_on_target)) - float(np.mean(deltas_off_target)), 6
                    ) if deltas_on_target and deltas_off_target else 0.0,
                }

            results[injected_fv][layer] = alpha_results

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="WS3: Steering injection.")
    parser.add_argument("--directions", choices=["mean_diff", "leace", "probe_weight"],
                        default="mean_diff")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--target-layers", default="4,8,12")
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHA_VALUES))
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-per-foundation", type=int, default=200)
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS3: Steering Vector Injection")
    print(f"{'='*60}")

    # Load eval prompts
    eval_path = OUTPUT_DIR / "causal_eval_prompts.json"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found. Run causal_eval_prompts.py first.")
        return
    eval_dataset = CausalEvalDataset.from_json(eval_path)
    print(f"Loaded {len(eval_dataset.prompts)} evaluation prompts")

    target_layers = [int(x) for x in args.target_layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

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
    print(f"Alpha values: {alphas}")

    # Load raw model for injection
    print(f"\nLoading model for injection: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(device).eval()

    print("Running injection experiments...")
    results = run_injection(
        model, tokenizer, eval_dataset, directions, target_layers, alphas, device,
    )

    # Print dose-response summary
    print(f"\n--- Injection Dose-Response ({method_name}) ---")
    print(f"{'Foundation':<14s}  {'Layer':>5s}  " + "  ".join(f"{'α='+str(a):>8s}" for a in alphas))
    print("-" * (25 + 10 * len(alphas)))
    for fv in FOUNDATION_ORDER:
        if fv not in results:
            continue
        for layer in target_layers:
            layer_res = results[fv].get(layer, {})
            if not layer_res:
                continue
            vals = [f"{layer_res.get(a, {}).get('specificity', 0):>8.4f}" for a in alphas]
            print(f"  {FOUNDATION_SHORT[fv]:<12s}  {layer:>5d}  " + "  ".join(vals))

    # Save
    out_data = {
        "analysis": "steering_injection",
        "method": method_name,
        "model": args.model,
        "target_layers": target_layers,
        "alphas": alphas,
        "n_prompts": len(eval_dataset.prompts),
        "results": {
            fv: {
                str(layer): {
                    str(alpha): metrics
                    for alpha, metrics in layer_data.items()
                }
                for layer, layer_data in fv_data.items()
            }
            for fv, fv_data in results.items()
        },
    }
    out_path = output_dir / f"steering_injection_{method_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
