#!/usr/bin/env python3
"""Sprint 2.3: comprehension-compliance coupling measurement (core methodology).

For each morally-loaded scenario we measure two things and how tightly they
are linked:

  * COMPREHENSION (internal): read the hidden state at the stable moral layer
    at the last prompt token, project onto the 6 foundation directions, and
    take the dominant foundation (argmax of per-foundation z-scored projection).
    The model "comprehends" a scenario if its dominant internal foundation
    matches the scenario's target foundation.

  * COMPLIANCE (behavioral): generate a completion and parse the moral
    judgment; the model "complies" if the judgment matches the expected one.

  * COUPLING: agreement between the per-scenario comprehension bit and the
    compliance bit (raw agreement + phi correlation). High coupling means the
    model's behavior tracks its internal moral representation; a comprehension-
    high / compliance-low split is the dissociation the paper predicts.

Scenarios and the judgment parser are reused from ``MoralFoundationsProbe`` so
this stays consistent with the behavioral benchmark.

Optionally, a judge/keyword classifier for the OUTPUT foundation can be plugged
in (``--output-foundation-classifier``) to additionally report
internal-dominant vs output-foundation agreement; left off by default because a
reliable 6-way text classifier needs the judge decision (Sprint 1.5).

Usage:
    python papers/5_moral_alignment/scripts/coupling_measurement.py \
        --model allenai/Olmo-3-7B-Instruct \
        --probe-dir papers/5_moral_alignment/outputs/olmo3_base \
        --layer 16 --input-format chat \
        --output-dir papers/5_moral_alignment/outputs/olmo3_instruct_coupling \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepsteer.directions import extraction as du  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402

logger = logging.getLogger(__name__)


def format_prompt(model, scenario_prompt: str, input_format: str) -> str:
    """Build the generation prompt; chat mode applies the model template."""
    if input_format == "chat" and getattr(model.tokenizer, "chat_template", None):
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": scenario_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return scenario_prompt


def last_token_hidden(model, text: str, layer: int) -> np.ndarray:
    """Hidden state at *layer* for the final token of *text*."""
    acts = model.get_activations(text, layers=[layer])
    return acts[layer][0, -1, :].float().numpy()


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Phi correlation between two binary vectors."""
    a = a.astype(float)
    b = b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehension-compliance coupling.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--probe-dir", required=True,
                    help="Dir with foundation directions npz.")
    ap.add_argument("--directions-npz", default=None,
                    help="Override path; default <probe-dir>/exp1_probe_directions.npz.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Stable moral layer to read. Default: middle layer.")
    ap.add_argument("--input-format", choices=["raw", "chat"], default="chat")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.benchmarks.moral_reasoning.foundations import (
        _PROMPT_TEMPLATE, _SCENARIOS, _parse_moral_judgment,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz = args.directions_npz or str(Path(args.probe_dir) / "exp1_probe_directions.npz")
    dirs = du.load_directions(npz)
    foundations = [f for f in FOUNDATION_ORDER if f in dirs]
    print(f"Loaded directions for {len(foundations)} foundations from {npz}")

    t0 = time.time()
    model = WhiteBoxModel(
        args.model, device=args.device, access_tier=AccessTier.WEIGHTS,
        revision=args.revision,
    )
    n_layers = model.info.n_layers
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"Loaded {args.model} ({n_layers} layers); reading layer {layer}; "
          f"input_format={args.input_format}; loaded in {time.time()-t0:.1f}s")

    dir_mat = np.stack([dirs[f][layer] for f in foundations])  # (6, hidden)

    # ---- pass 1: internal projections + behavioral judgments ----
    records = []
    raw_proj = np.zeros((len(_SCENARIOS), len(foundations)))
    for i, sc in enumerate(_SCENARIOS):
        prompt = _PROMPT_TEMPLATE.format(scenario=sc.scenario)
        formatted = format_prompt(model, prompt, args.input_format)
        h = last_token_hidden(model, formatted, layer)
        raw_proj[i] = dir_mat @ h
        gen = model.generate(formatted, max_tokens=args.max_tokens, temperature=0.0)
        judgment = _parse_moral_judgment(gen.text)
        records.append({
            "index": i,
            "foundation": sc.foundation.value,
            "difficulty": sc.difficulty.name,
            "expected_judgment": sc.expected_judgment,
            "judgment": judgment,
            "compliance": bool(judgment == sc.expected_judgment),
            "completion": gen.text[:400],
        })
        if (i + 1) % 10 == 0 or i + 1 == len(_SCENARIOS):
            logger.info("  scenario %d/%d", i + 1, len(_SCENARIOS))
    model.release()

    # z-score each foundation's projection across scenarios -> comparable scale.
    mu = raw_proj.mean(axis=0, keepdims=True)
    sd = raw_proj.std(axis=0, keepdims=True) + 1e-12
    z = (raw_proj - mu) / sd
    dominant_idx = z.argmax(axis=1)
    fidx = {f: i for i, f in enumerate(foundations)}

    comp_bits = np.zeros(len(_SCENARIOS), dtype=int)   # comprehension
    cply_bits = np.zeros(len(_SCENARIOS), dtype=int)   # compliance
    for i, rec in enumerate(records):
        dom = foundations[dominant_idx[i]]
        rec["internal_dominant"] = dom
        rec["internal_z"] = {f: round(float(z[i, fidx[f]]), 3) for f in foundations}
        rec["comprehension"] = bool(dom == rec["foundation"])
        comp_bits[i] = int(rec["comprehension"])
        cply_bits[i] = int(rec["compliance"])

    # ---- coupling metrics ----
    comprehension_rate = float(comp_bits.mean())
    compliance_rate = float(cply_bits.mean())
    agreement = float((comp_bits == cply_bits).mean())
    phi = phi_coefficient(comp_bits, cply_bits)
    # P(comply | comprehend) vs P(comply | not comprehend)
    p_comply_given_comp = (
        float(cply_bits[comp_bits == 1].mean()) if (comp_bits == 1).any() else float("nan")
    )
    p_comply_given_not = (
        float(cply_bits[comp_bits == 0].mean()) if (comp_bits == 0).any() else float("nan")
    )

    per_foundation = {}
    for f in foundations:
        mask = np.array([r["foundation"] == f for r in records])
        per_foundation[f] = {
            "comprehension_rate": round(float(comp_bits[mask].mean()), 4),
            "compliance_rate": round(float(cply_bits[mask].mean()), 4),
            "n": int(mask.sum()),
        }

    payload = {
        "analysis": "coupling_measurement",
        "model": args.model,
        "revision": args.revision,
        "layer": layer,
        "input_format": args.input_format,
        "directions_npz": npz,
        "n_scenarios": len(_SCENARIOS),
        "comprehension_rate": round(comprehension_rate, 4),
        "compliance_rate": round(compliance_rate, 4),
        "coupling_agreement": round(agreement, 4),
        "coupling_phi": round(phi, 4) if phi == phi else None,
        "p_comply_given_comprehend": round(p_comply_given_comp, 4) if p_comply_given_comp == p_comply_given_comp else None,
        "p_comply_given_not_comprehend": round(p_comply_given_not, 4) if p_comply_given_not == p_comply_given_not else None,
        "per_foundation": per_foundation,
        "records": records,
    }
    with open(out / "coupling.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {out/'coupling.json'}")
    print(f"  comprehension={comprehension_rate:.3f}  compliance={compliance_rate:.3f}  "
          f"agreement={agreement:.3f}  phi={phi:.3f}")
    print(f"  P(comply|comprehend)={p_comply_given_comp:.3f}  "
          f"P(comply|¬comprehend)={p_comply_given_not:.3f}")


if __name__ == "__main__":
    main()
