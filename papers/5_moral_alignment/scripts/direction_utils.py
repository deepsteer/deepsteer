#!/usr/bin/env python3
"""Shared direction-extraction utilities for the Phase 2 (comprehension-
compliance) study.

Everything here keeps methodology identical to Paper 3's
``exp1_2_3_framework_geometry.py`` so that directions extracted across models,
checkpoints, and input formats are cosine-comparable:

  * mean-pooled hidden states (same pooling as ``LayerWiseMoralProbe``),
  * seeded ``nn.Linear`` BCE probe (50 epochs, lr 1e-2) -> unit weight vector,
  * mean-diff direction = unit(mean(pos) - mean(neg)),
  * NPZ key format ``{name}_layer{idx}`` (reads exp1_probe_directions.npz).

Two activation paths:
  * ``input_format="raw"`` reuses ``model.collect_batch_activations(pooling=
    "mean")`` verbatim -> bit-for-bit the same pooling as the moral probe.
  * ``input_format="chat"`` wraps each text in the model chat template and
    mean-pools ONLY the content tokens (template tokens excluded via the
    tokenizer offset mapping).

numpy/torch only; no sklearn/scipy dependency.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)

# OLMo-3 7B is hybrid attention: every 4th layer is full attention, the rest
# sliding-window (4096). Flag these in layer-wise plots so any 4-layer
# periodicity is attributable to attention type rather than a real signal.
OLMO3_FULL_ATTENTION_LAYERS: list[int] = [3, 7, 11, 15, 19, 23, 27, 31]

PROBE_SEED = 42
PROBE_EPOCHS = 50
PROBE_LR = 1e-2

_NO_CHAT_TEMPLATE_WARNED = False


def load_whitebox(model_id: str, device=None, access_tier=None) -> WhiteBoxModel:
    """WhiteBoxModel loader that dequantizes mxfp4 (GPT-OSS) to bf16 for precision parity.

    GPT-OSS ships mxfp4-quantized MoE weights; Paper 7's resolution is
    ``Mxfp4Config(dequantize=True)`` + bf16 (needs transformers>=4.55, torch>=2.6). OLMo/other
    repos load unchanged, so this is a safe drop-in for the extraction/refusal scripts.
    """
    from deepsteer.core.types import AccessTier
    kw: dict = {}
    if "gpt-oss" in model_id.lower() or "gpt_oss" in model_id.lower():
        kw["torch_dtype"] = torch.bfloat16
        try:
            from transformers import Mxfp4Config
            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Mxfp4Config unavailable (%s); relying on torch_dtype auto-dequant", e)
    return WhiteBoxModel(model_id, device=device,
                         access_tier=access_tier or AccessTier.WEIGHTS, **kw)


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def chat_wrap(tokenizer, text: str) -> tuple[str, tuple[int, int]]:
    """Render *text* as a single user turn and return (full_string, char_span).

    ``char_span`` is the ``(start, end)`` character offset of *text* inside the
    rendered string, used to mask template tokens when pooling.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        # Base / pretraining checkpoints have no chat template. Probe the raw
        # text rather than fabricating <|im_start|> tokens the model never saw
        # (which would distort its activations). Warn once per run.
        global _NO_CHAT_TEMPLATE_WARNED
        if not _NO_CHAT_TEMPLATE_WARNED:
            logger.warning("Tokenizer has no chat_template; probing raw text "
                           "(expected for base/pretraining checkpoints).")
            _NO_CHAT_TEMPLATE_WARNED = True
        return text, (0, len(text))
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    start = full.rfind(text)
    if start < 0:  # template altered the content (rare); fall back to whole string
        logger.warning("Could not locate content span in templated string.")
        return full, (0, len(full))
    return full, (start, start + len(text))


def _content_pooled(
    model: WhiteBoxModel,
    full_text: str,
    span: tuple[int, int],
    layers: list[int],
) -> dict[int, Tensor]:
    """Mean-pool hidden states over the tokens overlapping *span* only."""
    enc = model.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    cs, ce = span
    keep = torch.tensor(
        [1.0 if (b > a and b > cs and a < ce) else 0.0 for a, b in offsets]
    )
    if keep.sum() == 0:  # degenerate; keep everything non-special
        keep = torch.tensor([1.0 if b > a else 0.0 for a, b in offsets])
    acts = model.get_activations(full_text, layers=layers)
    out: dict[int, Tensor] = {}
    w = keep.unsqueeze(-1)
    denom = w.sum().clamp(min=1.0)
    for layer_idx, t in acts.items():
        h = t[0].float()  # (seq, hidden)
        out[layer_idx] = (h * w).sum(dim=0) / denom
    return out


def collect_pair_activations(
    model: WhiteBoxModel,
    pairs: list[tuple[str, str]],
    *,
    input_format: str = "raw",
    layers: list[int] | None = None,
) -> dict[int, tuple[Tensor, Tensor]]:
    """Collect mean-pooled activations for ``(positive, negative)`` pairs.

    Rows are interleaved ``[pos0, neg0, pos1, neg1, ...]`` with labels
    ``[1, 0, 1, 0, ...]`` (matching the moral-probe ordering so pair-aware
    bootstrap resampling stays valid).

    Args:
        model: White-box model.
        pairs: ``(positive_text, negative_text)`` tuples.
        input_format: ``"raw"`` (verbatim text, batched pooling identical to
            the moral probe) or ``"chat"`` (chat-templated, content-token-only
            pooling).
        layers: Layer indices, ``None`` = all layers.

    Returns:
        ``{layer_idx: (X, y)}`` with ``X`` float32 ``(2*n_pairs, hidden)``.
    """
    if layers is None:
        layers = list(range(model.info.n_layers))  # type: ignore[arg-type]

    texts: list[str] = []
    for pos, neg in pairs:
        texts.extend([pos, neg])
    y = torch.tensor([1.0, 0.0] * len(pairs))

    if input_format == "raw":
        pooled = model.collect_batch_activations(texts, layers=layers, pooling="mean")
        return {L: (pooled[L].float(), y) for L in layers}

    if input_format == "chat":
        acc: dict[int, list[Tensor]] = {L: [] for L in layers}
        for i, t in enumerate(texts):
            full, span = chat_wrap(model.tokenizer, t)
            pooled = _content_pooled(model, full, span, layers)
            for L in layers:
                acc[L].append(pooled[L])
            if (i + 1) % 100 == 0 or i + 1 == len(texts):
                logger.info("  chat-pooled %d/%d texts", i + 1, len(texts))
        return {L: (torch.stack(acc[L]).float(), y) for L in layers}

    raise ValueError(f"Unknown input_format: {input_format!r}")


# ---------------------------------------------------------------------------
# Direction extraction
# ---------------------------------------------------------------------------


def mean_diff_direction(X: Tensor, y: Tensor) -> np.ndarray:
    """Unit-norm ``mean(positive) - mean(negative)`` direction."""
    Xn = X.detach().cpu().numpy()
    yn = y.detach().cpu().numpy()
    d = Xn[yn == 1].mean(axis=0) - Xn[yn == 0].mean(axis=0)
    return d / (np.linalg.norm(d) + 1e-12)


def probe_weight_direction(
    train_X: Tensor,
    train_y: Tensor,
    *,
    test_X: Tensor | None = None,
    test_y: Tensor | None = None,
    n_epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
    seed: int = PROBE_SEED,
) -> tuple[np.ndarray, float]:
    """Train a seeded linear probe; return (unit weight vector, accuracy).

    Identical methodology to Paper 3's ``train_probe_with_direction`` so probe
    directions are reproducible and comparable. Accuracy is on the test set if
    provided, else on the training set.
    """
    hidden_dim = train_X.shape[1]
    torch.manual_seed(seed)
    probe = nn.Linear(hidden_dim, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    probe.train()
    for _ in range(n_epochs):
        logits = probe(train_X).squeeze(-1)
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eval_X = test_X if test_X is not None else train_X
    eval_y = test_y if test_y is not None else train_y
    probe.eval()
    with torch.no_grad():
        preds = (probe(eval_X).squeeze(-1) > 0).float()
        accuracy = (preds == eval_y).float().mean().item()

    w = probe.weight.data.squeeze(0).cpu().numpy()
    return w / (np.linalg.norm(w) + 1e-12), accuracy


def extract_pair_directions(
    model: WhiteBoxModel,
    train_pairs: list[tuple[str, str]],
    *,
    test_pairs: list[tuple[str, str]] | None = None,
    input_format: str = "raw",
    layers: list[int] | None = None,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, float]]:
    """Extract per-layer probe-weight and mean-diff directions for one concept.

    Collects activations once, then per layer trains a seeded probe (direction
    + accuracy) and computes the mean-diff direction. Reused for fresh moral
    directions, persona directions, and any binary-pair concept across the
    pipeline.

    Returns:
        ``({"probe": {layer: vec}, "mean_diff": {layer: vec}}, {layer: acc})``.
        ``acc`` is held-out probe accuracy if ``test_pairs`` given, else train.
    """
    train = collect_pair_activations(
        model, train_pairs, input_format=input_format, layers=layers
    )
    test = (
        collect_pair_activations(model, test_pairs, input_format=input_format, layers=layers)
        if test_pairs
        else None
    )
    probe_dirs: dict[int, np.ndarray] = {}
    md_dirs: dict[int, np.ndarray] = {}
    accs: dict[int, float] = {}
    for layer_idx in sorted(train):
        tX, tY = train[layer_idx]
        eX, eY = test[layer_idx] if test else (None, None)
        w, acc = probe_weight_direction(tX, tY, test_X=eX, test_y=eY)
        probe_dirs[layer_idx] = w
        accs[layer_idx] = acc
        md_dirs[layer_idx] = mean_diff_direction(tX, tY)
    return {"probe": probe_dirs, "mean_diff": md_dirs}, accs


# ---------------------------------------------------------------------------
# Transfer evaluation (no retraining)
# ---------------------------------------------------------------------------


def project_scores(X: Tensor, direction: np.ndarray) -> np.ndarray:
    """Project rows of *X* onto a (not necessarily unit) *direction*."""
    Xn = X.detach().cpu().numpy() if isinstance(X, Tensor) else np.asarray(X)
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return Xn @ d


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average-rank tie handling. numpy only."""
    s = np.asarray(scores, dtype=float)
    yn = np.asarray(y)
    n1 = int((yn == 1).sum())
    n0 = int((yn == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[yn == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def transfer_metrics(X: Tensor, y: Tensor, direction: np.ndarray) -> dict:
    """Evaluate a fixed *direction* on (X, y) without retraining.

    Returns:
        ``auc``: threshold-free separability (orientation-aware; <0.5 means
            the direction points the opposite way). ``auc_abs`` = max(auc,
            1-auc).
        ``acc_midpoint``: accuracy thresholding at the midpoint of the two
            class-mean projections (one mild centering param, the standard
            probe-transfer report).
        ``threshold``: that midpoint.
    """
    scores = project_scores(X, direction)
    yn = y.detach().cpu().numpy() if isinstance(y, Tensor) else np.asarray(y)
    auc = roc_auc(scores, yn)
    pos_mean = scores[yn == 1].mean()
    neg_mean = scores[yn == 0].mean()
    thr = (pos_mean + neg_mean) / 2.0
    sign = 1.0 if pos_mean >= neg_mean else -1.0
    preds = (sign * (scores - thr) > 0).astype(float)
    acc = float((preds == yn).mean())
    return {
        "auc": auc,
        "auc_abs": max(auc, 1.0 - auc) if auc == auc else float("nan"),
        "acc_midpoint": acc,
        "threshold": float(thr),
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def cosine_matrix(direction_list: list[np.ndarray]) -> np.ndarray:
    """Pairwise cosine matrix for a list of (unit) direction vectors."""
    mat = np.stack([d / (np.linalg.norm(d) + 1e-12) for d in direction_list])
    return mat @ mat.T


def effective_dimensionality(
    direction_list: list[np.ndarray], variance_threshold: float = 0.9
) -> int:
    """PCs explaining >= threshold of variance in the centered direction set."""
    mat = np.stack(direction_list)
    mat = mat - mat.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(mat, full_matrices=False)
    explained = np.cumsum(s**2) / np.sum(s**2)
    return int(np.searchsorted(explained, variance_threshold)) + 1


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# NPZ persistence (exp1_probe_directions.npz compatible)
# ---------------------------------------------------------------------------


def save_directions(path, directions: dict[str, dict[int, np.ndarray]]) -> None:
    """Save ``{name: {layer: vec}}`` as NPZ with keys ``{name}_layer{idx}``."""
    arrays: dict[str, np.ndarray] = {}
    for name, by_layer in directions.items():
        for layer_idx, vec in by_layer.items():
            arrays[f"{name}_layer{layer_idx}"] = vec
    np.savez(path, **arrays)


_KEY_RE = re.compile(r"^(.+)_layer(\d+)$")


def load_directions(path) -> dict[str, dict[int, np.ndarray]]:
    """Inverse of :func:`save_directions`. Reads exp1_probe_directions.npz too."""
    npz = np.load(path)
    out: dict[str, dict[int, np.ndarray]] = {}
    for key in npz.files:
        m = _KEY_RE.match(key)
        if not m:
            continue
        name, layer_idx = m.group(1), int(m.group(2))
        out.setdefault(name, {})[layer_idx] = npz[key]
    return out
