#!/usr/bin/env python3
"""WS4: Behavioral benchmarking.

Validate that extracted foundation directions generalize to external stimuli
by measuring projection-based classification accuracy on:
  1. Held-out test set (internal validation)
  2. Moral Foundations Vignettes (Clifford et al., 2015) — external validation
  3. WS3 causal eval prompts — cross-validation with causal results

For each stimulus, project its activation onto all 6 foundation directions.
The direction with the highest absolute projection should match the stimulus's
labeled foundation. This bridges representational geometry and behavioral
relevance: the directions don't just separate moral/neutral within our dataset,
they identify *which* moral foundation is active in novel text.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/behavioral_benchmarking.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

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


# ---------------------------------------------------------------------------
# Moral Foundations Vignettes (Clifford et al., 2015)
# Curated subset: 5 per foundation, high-loading items from the original set.
# These are established MFT stimuli independent of our dataset pipeline.
# ---------------------------------------------------------------------------

MFV_ITEMS = [
    # Care/Harm
    {"text": "You see a boy kicking a puppy on the side of the road.", "foundation": "care_harm"},
    {"text": "You see a woman slapping her child repeatedly in a parking lot.", "foundation": "care_harm"},
    {"text": "You see a nurse ignoring the cries of a patient in pain.", "foundation": "care_harm"},
    {"text": "You see a man laughing at a homeless person who is begging for food.", "foundation": "care_harm"},
    {"text": "You see an ambulance driver refusing to help an injured person.", "foundation": "care_harm"},

    # Fairness/Cheating
    {"text": "You see a student copying answers from another student's exam.", "foundation": "fairness_cheating"},
    {"text": "You see an employer paying women less than men for the same work.", "foundation": "fairness_cheating"},
    {"text": "You see a judge giving a lighter sentence to a wealthy defendant.", "foundation": "fairness_cheating"},
    {"text": "You see a coach giving a trophy to his own child instead of the best player.", "foundation": "fairness_cheating"},
    {"text": "You see a professor giving higher grades to students who agree with her views.", "foundation": "fairness_cheating"},

    # Liberty/Oppression
    {"text": "You see a government banning citizens from leaving the country.", "foundation": "liberty_oppression"},
    {"text": "You see a boss threatening to fire workers who join a union.", "foundation": "liberty_oppression"},
    {"text": "You see police arresting someone for expressing a political opinion.", "foundation": "liberty_oppression"},
    {"text": "You see a school suspending students for wearing religious clothing.", "foundation": "liberty_oppression"},
    {"text": "You see a company monitoring all private conversations of its employees.", "foundation": "liberty_oppression"},

    # Loyalty/Betrayal
    {"text": "You see a soldier deserting his unit during a dangerous mission.", "foundation": "loyalty_betrayal"},
    {"text": "You see a man telling his wife's secrets to strangers at a bar.", "foundation": "loyalty_betrayal"},
    {"text": "You see an employee secretly sharing company plans with a competitor.", "foundation": "loyalty_betrayal"},
    {"text": "You see a teammate deliberately missing a shot to make his team lose.", "foundation": "loyalty_betrayal"},
    {"text": "You see a citizen publicly cheering for his country's enemy.", "foundation": "loyalty_betrayal"},

    # Authority/Subversion
    {"text": "You see a student publicly cursing at a teacher during class.", "foundation": "authority_subversion"},
    {"text": "You see a soldier openly disobeying a commanding officer's order.", "foundation": "authority_subversion"},
    {"text": "You see a child throwing food at her parents during dinner.", "foundation": "authority_subversion"},
    {"text": "You see an employee mocking the company founder at a meeting.", "foundation": "authority_subversion"},
    {"text": "You see a citizen tearing down a portrait of the country's leader.", "foundation": "authority_subversion"},

    # Sanctity/Degradation
    {"text": "You see someone urinating on a grave in a cemetery.", "foundation": "sanctity_degradation"},
    {"text": "You see a man using a national flag as a doormat.", "foundation": "sanctity_degradation"},
    {"text": "You see someone spitting into a bowl of food before serving it to guests.", "foundation": "sanctity_degradation"},
    {"text": "You see a family eating their dead pet dog for dinner.", "foundation": "sanctity_degradation"},
    {"text": "You see teenagers spray-painting obscenities on a church wall.", "foundation": "sanctity_degradation"},
]


def project_texts_onto_directions(
    texts: list[str],
    directions: dict[str, dict[int, np.ndarray]],
    model,
    tokenizer,
    layers: list[int],
    device: str,
) -> list[dict[str, dict[int, float]]]:
    """Project each text's activation onto all foundation directions.

    Returns: list of {foundation: {layer: projection_score}} per text.
    """
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        hook_outputs: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hook_outputs[layer_idx] = output[0][0, -1].detach().cpu()
                else:
                    hook_outputs[layer_idx] = output[0, -1].detach().cpu()
            return hook_fn

        handles = []
        for layer in layers:
            h = model.model.layers[layer].register_forward_hook(make_hook(layer))
            handles.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        projections: dict[str, dict[int, float]] = {}
        for fv in FOUNDATION_ORDER:
            projections[fv] = {}
            for layer in layers:
                d = directions.get(fv, {}).get(layer)
                if d is None:
                    continue
                act = hook_outputs[layer].numpy().astype(np.float64)
                projections[fv][layer] = float(np.dot(d, act))

        results.append(projections)
    return results


def classify_by_projection(
    projections: list[dict[str, dict[int, float]]],
    layers: list[int],
    debias: bool = False,
) -> list[dict]:
    """For each item, pick the foundation with highest mean projection across layers.

    If debias=True, subtract the mean projection across all 6 foundations
    before picking the winner. This removes the shared moral-salience component.
    """
    classified = []
    for proj in projections:
        mean_proj = {}
        for fv in FOUNDATION_ORDER:
            vals = [proj[fv].get(l, 0) for l in layers]
            mean_proj[fv] = np.mean(vals)

        if debias:
            shared = np.mean(list(mean_proj.values()))
            mean_proj = {fv: v - shared for fv, v in mean_proj.items()}

        predicted = max(mean_proj, key=mean_proj.get)
        classified.append({
            "predicted": predicted,
            "projections": {fv: round(v, 4) for fv, v in mean_proj.items()},
        })
    return classified


def _score_classified(classified, labels, label_str, debias_label=""):
    """Compute accuracy, per-foundation stats, and confusion matrix."""
    correct = sum(1 for c, l in zip(classified, labels) if c["predicted"] == l)
    total = len(labels)
    accuracy = correct / total if total > 0 else 0

    per_fnd: dict[str, dict] = {}
    for fv in FOUNDATION_ORDER:
        fv_items = [(c, l) for c, l in zip(classified, labels) if l == fv]
        if fv_items:
            fv_correct = sum(1 for c, l in fv_items if c["predicted"] == l)
            per_fnd[FOUNDATION_SHORT[fv]] = {
                "correct": fv_correct,
                "total": len(fv_items),
                "accuracy": round(fv_correct / len(fv_items), 4),
            }

    confusion: dict[str, dict[str, int]] = {
        FOUNDATION_SHORT[f1]: {FOUNDATION_SHORT[f2]: 0 for f2 in FOUNDATION_ORDER}
        for f1 in FOUNDATION_ORDER
    }
    for c, l in zip(classified, labels):
        confusion[FOUNDATION_SHORT[l]][FOUNDATION_SHORT[c["predicted"]]] += 1

    return {
        "label": label_str + debias_label,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_foundation": per_fnd,
        "confusion_matrix": confusion,
    }


def eval_on_items(
    items: list[dict],
    directions: dict[str, dict[int, np.ndarray]],
    model,
    tokenizer,
    layers: list[int],
    device: str,
    label: str,
) -> dict:
    """Run projection-based classification on a set of labeled items."""
    texts = [item["text"] for item in items]
    labels = [item["foundation"] for item in items]

    projections = project_texts_onto_directions(
        texts, directions, model, tokenizer, layers, device,
    )

    raw = classify_by_projection(projections, layers, debias=False)
    debiased = classify_by_projection(projections, layers, debias=True)

    raw_results = _score_classified(raw, labels, label)
    debiased_results = _score_classified(debiased, labels, label, "_debiased")

    return {
        "raw": raw_results,
        "debiased": debiased_results,
    }


def eval_on_test_set(
    directions: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    target_per_foundation: int,
) -> dict:
    """Evaluate on the held-out v2 test set via activation collection."""
    from deepsteer.datasets.pipeline import build_probing_dataset

    dataset = build_probing_dataset(target_per_foundation=target_per_foundation)
    test_pairs = dataset.test

    items = []
    for pair in test_pairs:
        items.append({
            "text": pair.moral,
            "foundation": pair.foundation.value,
        })

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="WS4: Behavioral benchmarking.")
    parser.add_argument("--directions", choices=["mean_diff", "leace", "probe_weight"],
                        default="mean_diff")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--target-layers", default="4,8,12")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-per-foundation", type=int, default=200)
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    print("=" * 60)
    print("WS4: Behavioral Benchmarking")
    print("=" * 60)

    target_layers = [int(x) for x in args.target_layers.split(",")]

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Device: {device}")

    # Get directions (also collects activations for data-dependent methods)
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

    # Load raw model for forward passes
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
    ).to(device).eval()

    all_results = {}

    def _print_results(res: dict, label: str):
        raw = res["raw"]
        deb = res["debiased"]
        print(f"\n--- {label} ---")
        print(f"  Raw accuracy:      {raw['accuracy']:.1%} ({raw['correct']}/{raw['total']})")
        print(f"  Debiased accuracy: {deb['accuracy']:.1%} ({deb['correct']}/{deb['total']})")
        print(f"  Per-foundation (debiased):")
        for fnd, stats in deb["per_foundation"].items():
            print(f"    {fnd:<12s}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    # 1. Moral Foundations Vignettes (external validation)
    mfv_results = eval_on_items(
        MFV_ITEMS, directions, model, tokenizer, target_layers, device, "MFV",
    )
    all_results["mfv"] = mfv_results
    _print_results(mfv_results, "Moral Foundations Vignettes (external)")

    # 2. Held-out test set (internal validation)
    test_items = eval_on_test_set(directions, target_layers, args.target_per_foundation)
    test_results = eval_on_items(
        test_items, directions, model, tokenizer, target_layers, device, "test_set",
    )
    all_results["test_set"] = test_results
    _print_results(test_results, "Held-out Test Set (internal)")

    # 3. WS3 causal eval prompts (cross-validation)
    eval_path = OUTPUT_DIR / "causal_eval_prompts.json"
    if eval_path.exists():
        from causal_eval_prompts import CausalEvalDataset
        causal_dataset = CausalEvalDataset.from_json(eval_path)
        causal_items = [
            {"text": p.prompt, "foundation": p.target_foundation}
            for p in causal_dataset.prompts
        ]
        causal_results = eval_on_items(
            causal_items, directions, model, tokenizer, target_layers, device, "causal_prompts",
        )
        all_results["causal_prompts"] = causal_results
        _print_results(causal_results, "WS3 Causal Eval Prompts (cross-validation)")

    # Print confusion matrices (debiased)
    print("\n--- Confusion Matrices (debiased) ---")
    short_names = [FOUNDATION_SHORT[f] for f in FOUNDATION_ORDER]
    for dataset_name, res in all_results.items():
        deb = res["debiased"]
        print(f"\n  {dataset_name}:")
        header = f"  {'True↓/Pred→':>12s}" + "".join(f"{s:>8s}" for s in short_names)
        print(header)
        for true_fnd in short_names:
            row = f"  {true_fnd:<12s}" + "".join(
                f"{deb['confusion_matrix'][true_fnd][pred_fnd]:>8d}"
                for pred_fnd in short_names
            )
            print(row)

    # Save
    out_data = {
        "analysis": "behavioral_benchmarking",
        "method": method_name,
        "model": args.model,
        "target_layers": target_layers,
        "results": all_results,
    }
    out_path = output_dir / f"behavioral_benchmarking_{method_name}.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
