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
    # Verbatim original Moral Foundations Vignettes (Clifford et al., 2015),
    # from Full-Published-Norms.xlsx (scottaclifford.com), Table 1 Respondent Ratings.
    # Top-5 per foundation by respondent classification rate into the intended
    # foundation; care from the combined emotional+physical care pool. See
    # deepsteer/datasets/mfv_clifford2015.json for provenance and per-item rates.

    # Care/Harm (emotional + physical care)
    {"text": "You see a woman swerving her car in order to intentionally run over a squirrel.", "foundation": "care_harm"},
    {"text": "You see a woman throwing her cat across the room for scratching the furniture.", "foundation": "care_harm"},
    {"text": "You see a boy throwing rocks at cows that are grazing in the local pasture.", "foundation": "care_harm"},
    {"text": "You see a man lashing his pony with a whip for breaking loose from its pen.", "foundation": "care_harm"},
    {"text": "You see a teenage boy chuckling at an amputee he passes by while on the subway.", "foundation": "care_harm"},

    # Fairness/Cheating
    {"text": "You see a student copying a classmate's answer sheet on a makeup final exam.", "foundation": "fairness_cheating"},
    {"text": "You see a runner taking a shortcut on the course during the marathon in order to win.", "foundation": "fairness_cheating"},
    {"text": "You see a tenant bribing a landlord to be the first to get their apartment repainted.", "foundation": "fairness_cheating"},
    {"text": "You see a soccer player pretending to be seriously fouled by an opposing player.", "foundation": "fairness_cheating"},
    {"text": "You see someone cheating in a card game while playing with a group of strangers.", "foundation": "fairness_cheating"},

    # Liberty/Oppression
    {"text": "You see a man telling his fiance that she has to switch to his political party.", "foundation": "liberty_oppression"},
    {"text": "You see a father requiring his son to become a commercial airline pilot like him.", "foundation": "liberty_oppression"},
    {"text": "You see a man blocking his wife from leaving home or interacting with others.", "foundation": "liberty_oppression"},
    {"text": "You see a manager coercing her employees into eating at her brother's diner.", "foundation": "liberty_oppression"},
    {"text": "You see a man telling his girlfriend that she must convert to his religion.", "foundation": "liberty_oppression"},

    # Loyalty/Betrayal
    {"text": "You see an employee joking with competitors about how bad his company did last year.", "foundation": "loyalty_betrayal"},
    {"text": "You see a coach celebrating with the opposing team's players who just won the game.", "foundation": "loyalty_betrayal"},
    {"text": "You see a former US General saying publicly he would never buy any American product.", "foundation": "loyalty_betrayal"},
    {"text": "You see a mayor saying that the neighboring town is a much better town.", "foundation": "loyalty_betrayal"},
    {"text": "You see the US Ambassador joking in Great Britain about the stupidity of Americans.", "foundation": "loyalty_betrayal"},

    # Authority/Subversion
    {"text": "You see a girl ignoring her father's orders by taking the car after her curfew.", "foundation": "authority_subversion"},
    {"text": "You see a woman refusing to stand when the judge walks into the courtroom.", "foundation": "authority_subversion"},
    {"text": "You see a girl repeatedly interrupting her teacher as he explains a new concept.", "foundation": "authority_subversion"},
    {"text": "You see a woman spray painting graffiti across the steps of the local courthouse.", "foundation": "authority_subversion"},
    {"text": "You see an intern disobeying an order to dress professionally and comb his hair.", "foundation": "authority_subversion"},

    # Sanctity/Degradation
    {"text": "You see a man having sex with a frozen chicken before cooking it for dinner.", "foundation": "sanctity_degradation"},
    {"text": "You see a very drunk woman making out with multiple strangers on the city bus.", "foundation": "sanctity_degradation"},
    {"text": "You see a family eating the carcass of their pet dog that had been run over.", "foundation": "sanctity_degradation"},
    {"text": "You see a drunk elderly man offering to have oral sex with anyone in the bar.", "foundation": "sanctity_degradation"},
    {"text": "You see a teenager urinating in the wave pool at a crowded amusement park.", "foundation": "sanctity_degradation"},
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

    dataset = build_probing_dataset(target_per_foundation=target_per_foundation, dataset_version="v2")
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
