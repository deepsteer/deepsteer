#!/usr/bin/env python3
"""Sprint 5.1: moral-ablation perplexity (the moral-dependency metric).

Measures how much a model *depends on* its moral subspace for generation
quality, by a difference-in-differences over cross-entropy:

  1. Compute mean cross-entropy (nats/token) on a set of morally-loaded texts
     (L_moral) and a matched set of neutral texts (L_neutral).
  2. Ablate the 6-direction moral foundation subspace from the residual stream
     at every layer (project it out via forward hooks, the subspace analogue of
     Arditi single-direction ablation used in heretic_ablation.py).
  3. Recompute cross-entropy on both sets (L_moral_ablated, L_neutral_ablated).
  4. Moral dependency score = (L_moral_ablated - L_moral)
                            - (L_neutral_ablated - L_neutral).

The neutral arm controls for the generic damage any subspace removal causes;
the difference isolates the *moral-specific* dependency. A positive score means
removing moral directions hurts moral text more than neutral text, i.e. the
model routes some of its moral-text processing through the moral subspace.

Directions come from an NPZ in exp1_probe_directions.npz format (keys
``{foundation}_layer{idx}``), e.g. Phase 2's
``outputs/olmo3_base/exp1_probe_directions.npz``. The 6 foundation directions at
a given layer are orthonormalized (SVD) into a basis for that layer's moral
subspace; the projection removes the whole span, not one direction at a time.

The module exposes reusable functions (``build_subspace_basis``,
``ablate_subspace``, ``corpus_cross_entropy``, ``measure_dependency``) so the
pipeline orchestrator can run the metric across the checkpoint grid without a
subprocess per state.

numpy/torch only; no sklearn/scipy.

Usage:
    python papers/5_moral_alignment/scripts/moral_dependency.py \
        --model allenai/Olmo-3-1025-7B \
        --revision main \
        --directions-npz papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz \
        --output-dir papers/5_moral_alignment/outputs/dependency/olmo3_base \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT

# direction_utils lives next to the Phase 2 scripts; reuse load_directions so the
# NPZ format stays in one place.
_PHASE2_SCRIPTS = Path(__file__).resolve().parents[2] / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(_PHASE2_SCRIPTS))
import direction_utils as du  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Moral subspace basis
# ---------------------------------------------------------------------------


def build_subspace_basis(
    directions: dict[str, dict[int, np.ndarray]],
    *,
    kind: str = "probe",
    n_layers: int,
    foundations: list[str] | None = None,
    tol: float = 1e-6,
) -> tuple[dict[int, np.ndarray], dict[int, int], list[str]]:
    """Orthonormal per-layer basis for the moral foundation subspace.

    Args:
        directions: ``{name: {layer: vec}}`` from ``du.load_directions``.
        kind: ``"probe"`` uses the bare foundation keys (``care_harm``);
            ``"meandiff"`` uses the ``{foundation}_meandiff`` keys.
        n_layers: Layer count of the target model.
        foundations: Foundation order, defaults to ``FOUNDATION_ORDER``.
        tol: Singular values below ``tol * s_max`` are treated as zero so a
            rank-deficient direction set yields a thin basis instead of noise.

    Returns:
        ``(basis_by_layer, rank_by_layer, names)`` where ``basis_by_layer[L]`` is
        an ``(r, hidden)`` float32 array of orthonormal rows spanning the moral
        subspace at layer ``L`` (only layers where every foundation is present),
        and ``names`` are the NPZ keys actually used.
    """
    foundations = foundations or FOUNDATION_ORDER
    if kind == "probe":
        names = list(foundations)
    elif kind == "meandiff":
        names = [f"{f}_meandiff" for f in foundations]
    else:
        raise ValueError(f"Unknown direction kind: {kind!r}")

    present = [n for n in names if n in directions]
    if len(present) < len(names):
        missing = [n for n in names if n not in directions]
        logger.warning("directions npz missing %d/%d %s keys: %s",
                       len(missing), len(names), kind, missing)

    basis: dict[int, np.ndarray] = {}
    ranks: dict[int, int] = {}
    for layer in range(n_layers):
        vecs = [directions[n][layer] for n in present if layer in directions[n]]
        if not present or len(vecs) < len(present):
            continue  # require every available foundation at this layer
        mat = np.stack(vecs).astype(np.float64)  # (k, hidden)
        # SVD row basis: Vt rows are an orthonormal basis of span(rows of mat).
        _, s, vt = np.linalg.svd(mat, full_matrices=False)
        r = int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0
        basis[layer] = vt[:r].astype(np.float32)
        ranks[layer] = r
    return basis, ranks, present


# ---------------------------------------------------------------------------
# Ablation hooks
# ---------------------------------------------------------------------------


def _make_projection_hook(basis: torch.Tensor):
    """Forward hook that projects a layer's residual output off ``span(basis)``.

    ``basis`` is ``(r, hidden)`` with orthonormal rows, so the projection away
    from the subspace is ``x - (x @ basis^T) @ basis``. The arithmetic stays
    differentiable (no in-place ops, no ``no_grad`` on the activations) so the
    same hook is reusable by the Sprint 6 ART loss.
    """

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        coeff = tensor @ basis.t()       # (..., r)
        proj = coeff @ basis             # (..., hidden)
        patched = tensor - proj
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    return hook


@contextmanager
def ablate_subspace(model, basis_by_layer: dict[int, np.ndarray], layers: list[int]):
    """Context manager: project the moral subspace out of every listed layer.

    Registers one forward hook per layer for the duration of the ``with`` block,
    mirroring ``heretic_ablation.orthogonalize_weights`` but applied to
    activations (so it is non-destructive and reversible) and to a multi-
    direction subspace rather than a single direction.
    """
    param = next(model.model.parameters())
    handles = []
    try:
        for layer in layers:
            v = basis_by_layer.get(layer)
            if v is None or v.shape[0] == 0:
                continue
            basis = torch.from_numpy(v).to(device=param.device, dtype=param.dtype)
            module = model._get_layer_module(layer)
            handles.append(module.register_forward_hook(_make_projection_hook(basis)))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Cross-entropy / perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def _text_nll(model, text: str) -> tuple[float, int]:
    """Sum next-token NLL (nats) and token count for one text, batch size 1."""
    enc = model.tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"]
    if ids.shape[1] < 2:
        return 0.0, 0
    param = next(model.model.parameters())
    ids = ids.to(param.device)
    attn = enc.get("attention_mask")
    kwargs: dict[str, Any] = {"input_ids": ids}
    if attn is not None:
        kwargs["attention_mask"] = attn.to(param.device)
    logits = model.model(**kwargs).logits[0].float()  # (seq, vocab)
    shift_logits = logits[:-1]
    shift_labels = ids[0, 1:]
    logp = F.log_softmax(shift_logits, dim=-1)
    token_ll = logp[torch.arange(shift_labels.shape[0], device=logp.device), shift_labels]
    return float(-token_ll.sum().item()), int(shift_labels.shape[0])


def corpus_cross_entropy(model, texts: list[str]) -> dict:
    """Token-weighted mean cross-entropy (nats/token) over ``texts``.

    Returns ``ce`` (sum NLL / sum tokens), ``ppl`` (= exp(ce)), ``total_nll``,
    ``total_tokens``, and per-text ``nll``/``n_tokens`` arrays for full
    reproducibility from the JSON alone.
    """
    per_nll: list[float] = []
    per_tok: list[int] = []
    for t in texts:
        nll, ntok = _text_nll(model, t)
        per_nll.append(round(nll, 6))
        per_tok.append(ntok)
    total_nll = float(sum(per_nll))
    total_tok = int(sum(per_tok))
    ce = total_nll / total_tok if total_tok else float("nan")
    return {
        "ce": ce,
        "ppl": math.exp(ce) if ce == ce else float("nan"),
        "total_nll": total_nll,
        "total_tokens": total_tok,
        "per_text_nll": per_nll,
        "per_text_n_tokens": per_tok,
    }


# ---------------------------------------------------------------------------
# Dependency measurement
# ---------------------------------------------------------------------------


def measure_dependency(
    model,
    moral_texts: list[str],
    neutral_texts: list[str],
    basis_by_layer: dict[int, np.ndarray],
    layers: list[int],
    *,
    keep_per_text: bool = True,
) -> dict:
    """Difference-in-differences moral dependency for one loaded model.

    Runs four cross-entropy passes (moral/neutral × clean/ablated) and assembles
    the dependency score plus a perplexity-ratio form of the same comparison.
    """
    clean_moral = corpus_cross_entropy(model, moral_texts)
    clean_neutral = corpus_cross_entropy(model, neutral_texts)
    with ablate_subspace(model, basis_by_layer, layers):
        abl_moral = corpus_cross_entropy(model, moral_texts)
        abl_neutral = corpus_cross_entropy(model, neutral_texts)

    delta_moral = abl_moral["ce"] - clean_moral["ce"]
    delta_neutral = abl_neutral["ce"] - clean_neutral["ce"]
    score = delta_moral - delta_neutral

    # Perplexity-ratio form: how much more does ablation inflate moral PPL than
    # neutral PPL? > 1.0 means moral-specific dependency. Equals exp(score).
    ppl_ratio = (
        (abl_moral["ppl"] / clean_moral["ppl"]) / (abl_neutral["ppl"] / clean_neutral["ppl"])
        if clean_moral["ppl"] and clean_neutral["ppl"]
        else float("nan")
    )

    result = {
        "moral_dependency_score": round(score, 6),
        "moral_dependency_ppl_ratio": round(ppl_ratio, 6) if ppl_ratio == ppl_ratio else None,
        "delta_ce": {"moral": round(delta_moral, 6), "neutral": round(delta_neutral, 6)},
        "ce": {
            "moral": round(clean_moral["ce"], 6),
            "moral_ablated": round(abl_moral["ce"], 6),
            "neutral": round(clean_neutral["ce"], 6),
            "neutral_ablated": round(abl_neutral["ce"], 6),
        },
        "ppl": {
            "moral": round(clean_moral["ppl"], 4),
            "moral_ablated": round(abl_moral["ppl"], 4),
            "neutral": round(clean_neutral["ppl"], 4),
            "neutral_ablated": round(abl_neutral["ppl"], 4),
        },
        "n_tokens": {
            "moral": clean_moral["total_tokens"],
            "neutral": clean_neutral["total_tokens"],
        },
        "n_texts": {"moral": len(moral_texts), "neutral": len(neutral_texts)},
    }
    if keep_per_text:
        result["per_text"] = {
            "moral_nll": clean_moral["per_text_nll"],
            "moral_nll_ablated": abl_moral["per_text_nll"],
            "moral_n_tokens": clean_moral["per_text_n_tokens"],
            "neutral_nll": clean_neutral["per_text_nll"],
            "neutral_nll_ablated": abl_neutral["per_text_nll"],
            "neutral_n_tokens": clean_neutral["per_text_n_tokens"],
        }
    return result


# ---------------------------------------------------------------------------
# Probing texts
# ---------------------------------------------------------------------------


def load_probing_texts(
    target_per_foundation: int = 40,
    dataset_version: str = "v2",
    *,
    split: str = "all",
    max_texts: int | None = None,
) -> tuple[list[str], list[str], dict]:
    """Moral-positive and matched-neutral texts from the probing dataset.

    Args:
        target_per_foundation: Pairs per foundation requested from the builder.
        dataset_version: Probing dataset version.
        split: ``"all"`` (train+test), ``"train"``, or ``"test"``.
        max_texts: Optional cap per class (for fast local smoke tests).

    Returns:
        ``(moral_texts, neutral_texts, meta)``. The two lists stay index-aligned
        so per-text deltas can be paired downstream.
    """
    from deepsteer.datasets.pipeline import build_probing_dataset

    dataset = build_probing_dataset(
        target_per_foundation=target_per_foundation, dataset_version=dataset_version
    )
    if split == "train":
        pairs = list(dataset.train)
    elif split == "test":
        pairs = list(dataset.test)
    elif split == "all":
        pairs = list(dataset.train) + list(dataset.test)
    else:
        raise ValueError(f"Unknown split: {split!r}")

    if max_texts is not None:
        pairs = pairs[:max_texts]

    moral_texts = [p.moral for p in pairs]
    neutral_texts = [p.neutral for p in pairs]
    by_foundation: dict[str, int] = {}
    for p in pairs:
        by_foundation[p.foundation.value] = by_foundation.get(p.foundation.value, 0) + 1
    meta = {
        "version": dataset_version,
        "target_per_foundation": target_per_foundation,
        "split": split,
        "n_pairs": len(pairs),
        "pairs_per_foundation": by_foundation,
    }
    return moral_texts, neutral_texts, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Moral-ablation perplexity / dependency metric.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--directions-npz", required=True,
                    help="exp1_probe_directions.npz-format file with foundation directions.")
    ap.add_argument("--direction-kind", choices=["probe", "meandiff"], default="probe")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default=None, help="State label for metadata/orchestrator.")
    ap.add_argument("--dataset-target", type=int, default=40)
    ap.add_argument("--dataset-version", default="v2")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all")
    ap.add_argument("--max-texts", type=int, default=None,
                    help="Cap texts per class (fast local smoke test).")
    ap.add_argument("--ablate-layers", default=None,
                    help="Comma-separated layer subset to ablate; default = all layers "
                         "where the full moral subspace is present.")
    ap.add_argument("--no-per-text", dest="per_text", action="store_false", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    moral_texts, neutral_texts, ds_meta = load_probing_texts(
        args.dataset_target, args.dataset_version,
        split=args.split, max_texts=args.max_texts,
    )
    directions = du.load_directions(args.directions_npz)

    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device,
                          access_tier=AccessTier.WEIGHTS, revision=args.revision)
    n_layers = model.info.n_layers
    param = next(model.model.parameters())
    print(f"Loaded {args.model}@{args.revision} ({n_layers}L) in {time.time()-t0:.1f}s; "
          f"device={param.device}, dtype={param.dtype}")

    basis_by_layer, ranks, names = build_subspace_basis(
        directions, kind=args.direction_kind, n_layers=n_layers,
    )
    available = sorted(basis_by_layer)
    if args.ablate_layers:
        wanted = {int(x) for x in args.ablate_layers.split(",")}
        layers = [L for L in available if L in wanted]
    else:
        layers = available
    print(f"Moral subspace: {len(names)} {args.direction_kind} directions "
          f"({', '.join(FOUNDATION_SHORT.get(n, n) for n in names)}); "
          f"ablating {len(layers)} layers; "
          f"{len(moral_texts)} moral / {len(neutral_texts)} neutral texts")
    if not layers:
        raise RuntimeError("No layers with a complete moral subspace to ablate; "
                           "check --directions-npz matches the model hidden dim/layers.")

    t1 = time.time()
    metrics = measure_dependency(
        model, moral_texts, neutral_texts, basis_by_layer, layers,
        keep_per_text=args.per_text,
    )
    model.release()

    payload = {
        "analysis": "moral_dependency",
        "model": args.model,
        "revision": args.revision,
        "label": args.label,
        "directions_npz": args.directions_npz,
        "direction_kind": args.direction_kind,
        "direction_names": names,
        "n_layers": n_layers,
        "ablated_layers": layers,
        "subspace_rank_per_layer": {str(L): ranks[L] for L in layers},
        "full_attention_layers": du.OLMO3_FULL_ATTENTION_LAYERS,
        "device": str(param.device),
        "dtype": str(param.dtype),
        "dataset": ds_meta,
        "texts": {"moral": moral_texts, "neutral": neutral_texts},
        "metrics": metrics,
        "elapsed_s": round(time.time() - t1, 1),
    }
    with open(out / "moral_dependency.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    m = metrics
    print(f"\nWrote {out/'moral_dependency.json'} (measured in {payload['elapsed_s']}s)")
    print(f"  CE  moral {m['ce']['moral']:.4f} -> {m['ce']['moral_ablated']:.4f} "
          f"(Δ {m['delta_ce']['moral']:+.4f})")
    print(f"  CE  neutral {m['ce']['neutral']:.4f} -> {m['ce']['neutral_ablated']:.4f} "
          f"(Δ {m['delta_ce']['neutral']:+.4f})")
    print(f"  moral dependency score = {m['moral_dependency_score']:+.4f} nats/token "
          f"(ppl ratio {m['moral_dependency_ppl_ratio']})")


if __name__ == "__main__":
    main()
