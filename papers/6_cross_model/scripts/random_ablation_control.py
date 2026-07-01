#!/usr/bin/env python3
"""Phase 2c: random-direction ablation control (resolve the Llama judgment drop).

Llama's moral-judgment dropped (0.75 -> 0.604) under refusal ablation while the
linear moral representation stayed intact. Is that drop refusal-SPECIFIC (refusal
routes through generation-time moral machinery) or COLLATERAL (the model is
fragile to any weight perturbation of that magnitude at this layer)?

Control: ablate the refusal direction at the chosen layer (orthogonalize every
o_proj/down_proj against it, all layers, the battery op) and record the per-
matrix Frobenius norm REMOVED. Then run a magnitude-MATCHED random null: perturb
the same matrices with random Gaussian noise of identical per-matrix norm,
``--n-random`` independent draws (a plain random *unit direction* is near-
orthogonal to the weights and removes almost nothing, which would bias the test
toward 'refusal-specific'; matching the magnitude is the fair comparison). Also
ablate the persona direction (a real, decodable, non-refusal, non-moral feature)
as a structured control.

For EACH condition measure both:
  * moral judgment   (MoralFoundationsProbe, generation)
  * representation    (fresh per-foundation probe accuracy + framework eff-dim)

Then place the refusal-direction values against the random null distribution.
  refusal judgment OUTLIER below the matched-random null -> refusal-specific
  refusal judgment within the null                       -> collateral (magnitude)
Representation that survives BOTH refusal and matched-random ablation confirms
the dissociation is not an artifact of a gentle perturbation.

Diagnostic only: characterizes an existing perturbation; builds no refusal.

Usage:
    python papers/6_cross_model/scripts/random_ablation_control.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --refusal-npz papers/6_cross_model/outputs/llama31_instruct/refusal_directions.npz \
        --persona-npz papers/6_cross_model/outputs/llama31_instruct/persona_directions.npz \
        --layer 13 --n-random 8 \
        --output papers/6_cross_model/outputs/llama31/random_ablation_control.json
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
_P5 = Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(_P5))

from deepsteer.directions import extraction as du  # noqa: E402

logger = logging.getLogger(__name__)

_RESID_PATHS = ("self_attn.o_proj", "mlp.down_proj")


def _resolve(layer, path: str):
    mod = layer
    for attr in path.split("."):
        mod = getattr(mod, attr, None)
        if mod is None:
            return None
    return mod


def _resid_matrices(model):
    for L in range(model.info.n_layers):
        layer = model._get_layer_module(L)
        for path in _RESID_PATHS:
            mod = _resolve(layer, path)
            if mod is not None and hasattr(mod, "weight"):
                yield mod


@torch.no_grad()
def ablate_direction(model, d_np: np.ndarray) -> list[float]:
    """W <- W - d d^T W for every resid-write matrix; return per-matrix ||removed||_F."""
    norms: list[float] = []
    for mod in _resid_matrices(model):
        W = mod.weight.data
        d = torch.from_numpy(d_np).to(dtype=W.dtype, device=W.device)
        d = d / (d.norm() + 1e-12)
        removed = torch.outer(d, d @ W)
        W -= removed
        norms.append(float(removed.float().norm()))
    return norms


@torch.no_grad()
def perturb_matched(model, norms: list[float], seed: int) -> None:
    """Add random Gaussian noise to each resid-write matrix with matched Frobenius norm."""
    for i, mod in enumerate(_resid_matrices(model)):
        W = mod.weight.data
        g = torch.Generator(device=W.device)
        g.manual_seed(seed * 100003 + i)
        noise = torch.randn(W.shape, generator=g, device=W.device, dtype=torch.float32)
        noise = noise * (norms[i] / (float(noise.norm()) + 1e-12))
        W += noise.to(W.dtype)


def _chat_model(wb):
    from deepsteer.core.model_interface import ModelInterface
    from deepsteer.core.types import GenerationResult

    class ChatModel(ModelInterface):
        def __init__(self, w):
            self._wb = w

        @property
        def info(self):
            return self._wb.info

        def generate(self, prompt, *, max_tokens=256, temperature=0.0, system_prompt=None):
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            text = self._wb.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            return self._wb.generate(text, max_tokens=max_tokens, temperature=temperature)

        def score(self, prompt, completion):
            return self._wb.score(prompt, completion)

    _ = GenerationResult
    return ChatModel(wb)


def _build_groups(dataset_target: int):
    """{foundation: ([(moral,neutral)...train], [...test])} from the v2 probing set."""
    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.foundations import FOUNDATION_ORDER

    ds = build_probing_dataset(target_per_foundation=dataset_target, dataset_version="v2")
    groups: dict[str, tuple[list, list]] = {}
    for f in FOUNDATION_ORDER:
        tr = [(p.moral, p.neutral) for p in ds.train if p.foundation.value == f]
        te = [(p.moral, p.neutral) for p in ds.test if p.foundation.value == f]
        groups[f] = (tr, te)
    return groups


def _representation(wb, groups, headline: int) -> tuple[float | None, int | None]:
    """Fresh per-foundation probe accuracy (max over layers, mean over foundations) + eff-dim."""
    fresh, head_dirs = [], []
    for _f, (tr, te) in groups.items():
        if len(tr) < 5 or len(te) < 1:
            continue
        dirs, accs = du.extract_pair_directions(wb, tr, test_pairs=te, input_format="raw")
        fresh.append(max(accs.values()))
        if headline in dirs["probe"]:
            head_dirs.append(dirs["probe"][headline])
    probe_acc = round(float(np.mean(fresh)), 4) if fresh else None
    eff = int(du.effective_dimensionality(head_dirs)) if len(head_dirs) >= 2 else None
    return probe_acc, eff


def measure(model_name, device, max_tokens, groups, prep) -> dict:
    """Load, optionally prep(wb), then measure judgment + representation; release."""
    from deepsteer.benchmarks.moral_reasoning.foundations import MoralFoundationsProbe
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    wb = WhiteBoxModel(model_name, device=device, access_tier=AccessTier.WEIGHTS)
    extra = prep(wb) if prep else None
    res = MoralFoundationsProbe(temperature=0.0, max_tokens=max_tokens).run(_chat_model(wb))
    jacc = float(getattr(res, "overall_accuracy", float("nan")))
    jbyf = {k: round(float(v), 4) for k, v in getattr(res, "accuracy_by_foundation", {}).items()}
    headline = round(0.5 * wb.info.n_layers)
    probe_acc, eff = _representation(wb, groups, headline)
    wb.release()
    return {"judgment": round(jacc, 4), "judgment_byf": jbyf, "probe_acc": probe_acc,
            "eff_dim": eff, "headline_layer": headline, "extra": extra}


def _null(refusal_val, randoms: list, *, floor: float):
    """Place refusal against the random null: mean, std, z, and a verdict band."""
    vals = [v for v in randoms if v is not None]
    if not vals or refusal_val is None:
        return {"mean": None, "std": None, "z": None}
    mean, std = float(np.mean(vals)), float(np.std(vals))
    z = (refusal_val - mean) / (std + 1e-9)
    return {"mean": round(mean, 4), "std": round(std, 4), "z": round(z, 3),
            "min": round(float(min(vals)), 4), "max": round(float(max(vals)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2c random-direction ablation control")
    ap.add_argument("--model", required=True)
    ap.add_argument("--refusal-npz", required=True)
    ap.add_argument("--persona-npz", default=None, help="Optional persona dir (control).")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--dataset-target", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    L = args.layer
    refusal = du.load_directions(args.refusal_npz)["refusal"][L]
    persona = None
    if args.persona_npz and Path(args.persona_npz).exists():
        pd = du.load_directions(args.persona_npz)
        persona = pd.get("persona", pd.get("persona_meandiff", {})).get(L)

    groups = _build_groups(args.dataset_target)
    t0 = time.time()
    norms_holder: dict[str, list[float]] = {}

    def run(name, prep):
        print(f"[{name}] measuring ...")
        m = measure(args.model, args.device, args.max_tokens, groups, prep)
        print(f"  {name}: judgment={m['judgment']} probe_acc={m['probe_acc']} eff={m['eff_dim']}")
        return m

    clean = run("clean", None)

    def prep_refusal(wb):
        norms_holder["ref"] = ablate_direction(wb, refusal)

    refusal_m = run(f"refusal@L{L}", prep_refusal)
    norms_ref = norms_holder["ref"]
    total_norm = float(np.sqrt(np.sum(np.square(norms_ref))))
    print(f"  refusal removed ||W||_F total {total_norm:.2f}")

    persona_m = None
    if persona is not None:
        persona_m = run(f"persona@L{L}", lambda wb: ablate_direction(wb, persona))

    randoms = []
    for s in range(args.n_random):
        randoms.append(
            run(f"random[{s}] matched", lambda wb, s=s: perturb_matched(wb, norms_ref, s)))

    rj = [m["judgment"] for m in randoms]
    rp = [m["probe_acc"] for m in randoms]
    re_ = [m["eff_dim"] for m in randoms]
    null_j = _null(refusal_m["judgment"], rj, floor=0.0)
    null_p = _null(refusal_m["probe_acc"], rp, floor=0.0)

    # Verdict on the judgment metric: outlier below the matched-random null?
    z = null_j.get("z")
    if z is not None and refusal_m["judgment"] < (null_j["mean"] - max(2 * null_j["std"], 0.03)):
        verdict = ("refusal-specific: refusal-ablation judgment is an outlier BELOW the "
                   "magnitude-matched random null (z={:.2f})".format(z))
    elif z is not None and refusal_m["judgment"] >= (null_j["mean"] - max(null_j["std"], 0.02)):
        verdict = ("collateral: refusal-ablation judgment is WITHIN the matched random null "
                   "(general perturbation fragility, not refusal-specific; z={:.2f})".format(z))
    else:
        verdict = "ambiguous: refusal judgment between the random null and the outlier threshold"

    payload = {
        "analysis": "random_ablation_control", "model": args.model, "layer": L,
        "headline_layer": clean["headline_layer"], "max_tokens": args.max_tokens,
        "n_random": args.n_random, "refusal_removed_frobenius_total": round(total_norm, 4),
        "clean": {k: clean[k] for k in ("judgment", "probe_acc", "eff_dim")},
        "refusal": {k: refusal_m[k] for k in ("judgment", "probe_acc", "eff_dim")},
        "persona": ({k: persona_m[k] for k in ("judgment", "probe_acc", "eff_dim")}
                    if persona_m else None),
        "random_judgments": rj, "random_probe_accs": rp, "random_eff_dims": re_,
        "judgment_null": null_j, "probe_acc_null": null_p,
        "refusal_judgment_drop": round(clean["judgment"] - refusal_m["judgment"], 4),
        "random_judgment_drop_mean": (round(clean["judgment"] - null_j["mean"], 4)
                                      if null_j["mean"] is not None else None),
        "clean_by_foundation": clean["judgment_byf"],
        "refusal_by_foundation": refusal_m["judgment_byf"],
        "verdict": verdict, "elapsed_s": round(time.time() - t0, 1),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n=== random-direction control @L{L} (headline L{clean['headline_layer']}) ===")
    print("  metric          clean   refusal   persona   random-null(mean+/-std)")
    pj = persona_m["judgment"] if persona_m else None
    pp = persona_m["probe_acc"] if persona_m else None
    pe = persona_m["eff_dim"] if persona_m else None
    print(f"  judgment        {clean['judgment']:<7} {refusal_m['judgment']:<9} {str(pj):<9} "
          f"{null_j['mean']} +/- {null_j['std']}  (z {null_j['z']})")
    print(f"  probe_acc       {clean['probe_acc']:<7} {refusal_m['probe_acc']:<9} {str(pp):<9} "
          f"{null_p['mean']} +/- {null_p['std']}")
    print(f"  eff_dim         {clean['eff_dim']:<7} {refusal_m['eff_dim']:<9} {str(pe):<9} "
          f"{re_}")
    print(f"  VERDICT: {verdict}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
