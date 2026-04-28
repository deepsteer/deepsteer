#!/usr/bin/env python3
"""Phase D Step 2 Finding 3 mechanism check: numerically verify the
activation_patch backfire by measuring h_ap - h_vanilla on identical
held-out inputs and projecting onto the probe direction.

Predicted mechanism: during training the activation_patch model learned
to compensate for the −γ × unit_w subtraction at layers {5, 6, 7} by
producing layer-5 outputs that are shifted by approximately +γ × unit_w
relative to what a vanilla LoRA model would produce.  At eval time the
patch is detached, so the activation_patch model's natural layer-5
output is over-aligned with +w.

Quantities computed per held-out (prompt, response) pair:
  - scalar_projection := (delta @ w) / ||w||      predicted ≈ +γ
  - inner_product    := delta @ w                 predicted ≈ +γ × ||w||
  - delta_norm       := ||delta||
  - frac_along_w     := (delta @ unit_w)^2 / ||delta||^2
  - orthogonal_norm  := ||delta − (delta @ unit_w) × unit_w||

A binary "match within 10%" gate is reported on the scalar projection
since that is the dimensionless prediction γ ≈ 1.5.

Held-out inputs: ``--n-samples`` random samples from
``papers/persona_monitoring/outputs/phase_d/c10_v2/eval_baseline.json``.  These are base-model
responses to Betley's eight benign prompts, generated independently
of any LoRA fine-tune, so they are content-neutral and distinct from
the response distributions of the vanilla and activation_patch evals.

Output: ``papers/persona_monitoring/outputs/phase_d/step2_steering/finding3_mechanism_check.json``
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch

from deepsteer.benchmarks.representational import PersonaProbeWeights
from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)

MODEL_ID = "allenai/OLMo-2-0425-1B"
DEFAULT_OUTPUT_DIR = Path("papers/persona_monitoring/outputs/phase_d/step2_steering")
DEFAULT_BASELINE_EVAL = Path("papers/persona_monitoring/outputs/phase_d/c10_v2/eval_baseline.json")
DEFAULT_PROBE = DEFAULT_OUTPUT_DIR / "persona_probe.json"
DEFAULT_GAMMA = 1.5
PATCH_LAYERS = [5, 6, 7]


def load_probe(path: Path) -> PersonaProbeWeights:
    with open(path) as fh:
        payload = json.load(fh)
    return PersonaProbeWeights.from_dict(payload["weights"])


def sample_held_out_pairs(
    baseline_eval_path: Path, n_samples: int, seed: int
) -> list[dict[str, str]]:
    with open(baseline_eval_path) as fh:
        payload = json.load(fh)
    samples = payload["samples"]
    rng = random.Random(seed)
    picked = rng.sample(samples, k=min(n_samples, len(samples)))
    held_out: list[dict[str, str]] = []
    for s in picked:
        response = (s.get("response") or "").strip()
        if not response:
            continue
        held_out.append({
            "question_id": s.get("question_id", ""),
            "paraphrase_index": int(s.get("paraphrase_index", 0)),
            "prompt": s.get("prompt", ""),
            "response": response,
        })
    return held_out


def load_model_with_adapter(adapter_dir: Path) -> WhiteBoxModel:
    """Load fresh OLMo-2 1B base + LoRA adapter, then merge for clean forward."""
    from peft import PeftModel

    logger.info("Loading base model %s", MODEL_ID)
    model = WhiteBoxModel(MODEL_ID)
    logger.info("Loading LoRA adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(model._model, str(adapter_dir))
    if hasattr(peft_model, "merge_and_unload"):
        model._model = peft_model.merge_and_unload()
        logger.info("  LoRA adapter merged")
    else:
        model._model = peft_model
        logger.info("  LoRA adapter active (not merged)")
    model._model.eval()
    return model


@torch.no_grad()
def mean_pool_response_layer(
    model: WhiteBoxModel, response: str, layer: int
) -> torch.Tensor:
    """Forward `response` through model, return mean-pooled layer activation as fp32 (D,)."""
    acts = model.get_activations(response, layers=[layer])
    return acts[layer].squeeze(0).float().mean(dim=0)


def collect_activations(
    model: WhiteBoxModel,
    samples: list[dict[str, str]],
    layer: int,
    label: str,
) -> torch.Tensor:
    """Return fp32 tensor (n_samples, hidden_dim) of mean-pooled layer activations."""
    out: list[torch.Tensor] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        h = mean_pool_response_layer(model, s["response"], layer)
        out.append(h)
        if (i + 1) % 10 == 0 or i + 1 == len(samples):
            logger.info(
                "  %s: %d/%d samples (%.1fs elapsed)",
                label, i + 1, len(samples), time.time() - t0,
            )
    return torch.stack(out)  # (N, D)


def free_model(model: WhiteBoxModel) -> None:
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-path", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()

    out_path = args.input_dir / "finding3_mechanism_check.json"

    probe = load_probe(args.probe_path)
    w_t, _bias = probe.to_tensors()  # w_t is fp32 (D,)
    w = w_t.cpu()
    w_norm = float(torch.linalg.norm(w).item())
    unit_w = w / w_norm
    layer = probe.layer
    logger.info(
        "Probe: layer=%d, hidden_dim=%d, ||w||=%.4f, predicted γ×||w||=%.4f",
        layer, probe.hidden_dim, w_norm, args.gamma * w_norm,
    )

    samples = sample_held_out_pairs(args.baseline_eval, args.n_samples, args.seed)
    logger.info(
        "Held-out set: %d (prompt, response) pairs from %s",
        len(samples), args.baseline_eval,
    )

    # ----- Vanilla LoRA -----
    logger.info("=" * 60)
    logger.info("Computing h_van (vanilla LoRA)")
    logger.info("=" * 60)
    vanilla_dir = args.input_dir / "none" / "adapters"
    vanilla = load_model_with_adapter(vanilla_dir)
    h_van = collect_activations(vanilla, samples, layer, "vanilla")
    free_model(vanilla)

    # ----- Activation patch -----
    logger.info("=" * 60)
    logger.info("Computing h_ap (activation_patch LoRA)")
    logger.info("=" * 60)
    ap_dir = args.input_dir / "activation_patch" / "adapters"
    ap = load_model_with_adapter(ap_dir)
    h_ap = collect_activations(ap, samples, layer, "activation_patch")
    free_model(ap)

    # ----- Mechanism analysis -----
    delta = h_ap - h_van  # (N, D), fp32 CPU
    n = delta.shape[0]

    # Per-sample quantities.
    scalar_projection = (delta @ unit_w).cpu()  # predicted ≈ +γ
    inner_product = (delta @ w).cpu()           # predicted ≈ +γ × ||w||
    delta_norm = torch.linalg.norm(delta, dim=1).cpu()
    along_w_vec = scalar_projection.unsqueeze(-1) * unit_w.unsqueeze(0)  # (N, D)
    orthogonal = delta - along_w_vec
    orthogonal_norm = torch.linalg.norm(orthogonal, dim=1).cpu()
    frac_along_w = (scalar_projection ** 2) / torch.clamp(delta_norm ** 2, min=1e-12)

    def _stats(t: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=True).item()) if t.numel() > 1 else float("nan"),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    pred_proj_scalar = float(args.gamma)
    pred_inner_product = float(args.gamma * w_norm)
    obs_proj_scalar = float(scalar_projection.mean().item())
    obs_inner_product = float(inner_product.mean().item())

    proj_scalar_match = (
        abs(obs_proj_scalar - pred_proj_scalar) / max(abs(pred_proj_scalar), 1e-12)
        if pred_proj_scalar != 0
        else float("nan")
    )
    inner_match = (
        abs(obs_inner_product - pred_inner_product) / max(abs(pred_inner_product), 1e-12)
        if pred_inner_product != 0
        else float("nan")
    )

    sign_correct = int((scalar_projection > 0).sum().item())

    summary = {
        "input_dir": str(args.input_dir),
        "baseline_eval": str(args.baseline_eval),
        "probe_path": str(args.probe_path),
        "probe_layer": layer,
        "probe_w_norm": w_norm,
        "gamma": args.gamma,
        "patch_layers": PATCH_LAYERS,
        "n_samples": n,
        "predicted": {
            "scalar_projection_delta_at_unit_w": pred_proj_scalar,
            "inner_product_delta_at_w": pred_inner_product,
            "note": (
                "Mechanism: model compensates by shifting layer-5 output "
                "+γ × unit_w to absorb the −γ × unit_w training-time patch."
            ),
        },
        "observed": {
            "scalar_projection": _stats(scalar_projection),
            "inner_product": _stats(inner_product),
            "delta_norm": _stats(delta_norm),
            "orthogonal_norm": _stats(orthogonal_norm),
            "frac_along_w": _stats(frac_along_w),
            "n_positive_projection": sign_correct,
            "frac_positive_projection": sign_correct / n if n > 0 else float("nan"),
        },
        "match_check": {
            "scalar_projection_relative_error": proj_scalar_match,
            "inner_product_relative_error": inner_match,
            "scalar_projection_within_10pct": (
                proj_scalar_match < 0.10
                if proj_scalar_match == proj_scalar_match  # not NaN
                else False
            ),
            "inner_product_within_10pct": (
                inner_match < 0.10
                if inner_match == inner_match
                else False
            ),
        },
        "interpretation_hints": {
            "if_scalar_projection_near_gamma": (
                "Mechanism confirmed: per-sample compensation matches "
                "naive prediction γ × unit_w."
            ),
            "if_scalar_projection_positive_but_smaller": (
                "Partial compensation at layer 5 (e.g., later patched "
                "layers absorb some of the offset) — mechanism direction "
                "correct but magnitude smaller than naive single-layer model."
            ),
            "if_scalar_projection_near_zero_or_negative": (
                "Mechanism falsified: layer-5 shift not preferentially "
                "along +unit_w; backfire must be explained by a different "
                "pathway."
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", out_path)

    bar = "-" * 78
    print(bar)
    print("Phase D Step 2 — Finding 3 backfire mechanism check")
    print(bar)
    print(f"  N held-out samples: {n}")
    print(f"  γ = {args.gamma}, ||w|| = {w_norm:.4f}")
    print()
    print("  PREDICTION:")
    print(f"    scalar projection (delta @ w / ||w||) ≈ +γ = +{pred_proj_scalar:.4f}")
    print(f"    inner product    (delta @ w)         ≈ +γ × ||w|| = +{pred_inner_product:.4f}")
    print()
    print("  OBSERVED (mean ± SD):")
    sp = summary["observed"]["scalar_projection"]
    ip = summary["observed"]["inner_product"]
    dn = summary["observed"]["delta_norm"]
    on = summary["observed"]["orthogonal_norm"]
    fa = summary["observed"]["frac_along_w"]
    print(f"    scalar projection : {sp['mean']:+.4f} ± {sp['std']:.4f}    "
          f"(rel err {summary['match_check']['scalar_projection_relative_error']:.3f})")
    print(f"    inner product     : {ip['mean']:+.4f} ± {ip['std']:.4f}    "
          f"(rel err {summary['match_check']['inner_product_relative_error']:.3f})")
    print(f"    ||delta||         : {dn['mean']:.4f} ± {dn['std']:.4f}")
    print(f"    ||delta_orth||    : {on['mean']:.4f} ± {on['std']:.4f}")
    print(f"    frac along +w     : {fa['mean']:.4f} ± {fa['std']:.4f}")
    print(f"    sign positive     : {sign_correct}/{n}")
    print()
    print("  GATE:")
    print(f"    scalar projection within 10% of γ:  "
          f"{summary['match_check']['scalar_projection_within_10pct']}")
    print(f"    inner product within 10% of γ×||w||: "
          f"{summary['match_check']['inner_product_within_10pct']}")
    print(bar)


if __name__ == "__main__":
    main()
