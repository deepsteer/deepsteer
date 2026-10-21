"""ST's adapters (ST_PLAN §5.1-5.3, T0.6): on-policy traces -> position masks, tau, chat_sigma,
DPO pairs, and the three objectives through ``deepsteer.steering.objective_lora``.

Training runs on the ST stack of record (torch 2.13 in the vLLM venv, Amendment 9), bf16 on the
development model, QLoRA on the 27-30B second family; every adapter is saved with a sidecar
(objective, layer, class, tau, the anchor axis's provenance, sigma's provenance, LoRA config,
training config, per-step records), and evaluated merged at the base's precision.

Anchor axis (author, 2026-10-19): §5.2's ``<h, V_care> / |h|`` needs a signed axis in the rank-2
``V_care``; it is the normalized sum of the three ``V_moral`` source mean-differences (moral pole
positive) with the in-span harm component removed (``anchor_axis``). The protect-minus-rule
orientation inside ``V_care`` is a diagnostic only (``orientation_diagnostic``); anti-alignment is
logged in ANOMALIES.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steering_gate as SG  # noqa: E402

from deepsteer.steering import objective_lora as OL  # noqa: E402

PAIR_FLOOR = 40  # §5.1 / Amendment 1
TAU_QUANTILE = 0.75  # §5.2
DPO_BETA = 0.1  # Amendment 1


def trace_masks(fmt, tok, prompt: list[int], gen: list[int], cls: str,
                punct: np.ndarray) -> tuple[list[bool], list[bool]]:
    """Over prompt + generated ids: the anchoring class's trace positions, and the KL positions
    (trace tokens outside the class). The prompt and the answer carry neither."""
    trk = SG.Tracker(fmt, tok, punct)
    anchor, trace = [False] * len(prompt), [False] * len(prompt)
    for i in range(len(gen)):
        g = gen[: i + 1]
        trk.update(g)
        a = trk.in_class(g, cls)
        t = trk.in_class(g, "constant")
        anchor.append(a)
        trace.append(t)
    kl = [t and not a for a, t in zip(anchor, trace)]
    return anchor, kl


def compute_tau(protect_margins: np.ndarray) -> float:
    """§5.2: tau puts the unadapted model's protect rollouts at the 75th percentile of the margin
    at the anchoring positions (from T1.1's banked residuals; committed before training)."""
    return float(np.quantile(np.asarray(protect_margins, float), TAU_QUANTILE))


def anchor_axis(sources: np.ndarray, harm: np.ndarray) -> np.ndarray:
    """The §5.2 anchor (author, 2026-10-19): the normalized sum of the signed source
    mean-differences ``sources`` [k, H] (moral pole positive), each with the component along the
    in-span harm direction removed (the same removal that makes ``V_care`` rank 2)."""
    S = np.asarray(sources, float)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    B = np.linalg.qr(S.T)[0].T
    hv = B.T @ (B @ (harm / np.linalg.norm(harm)))
    if np.linalg.norm(hv) == 0:
        raise ValueError("harm has no component in the span of the sources")
    hv = hv / np.linalg.norm(hv)
    r = S - np.outer(S @ hv, hv)
    u = r.sum(0)
    return (u / np.linalg.norm(u)).astype(np.float32)


def orientation_diagnostic(axis: np.ndarray, v_care: np.ndarray, protect: np.ndarray,
                           rule: np.ndarray) -> float:
    """cos between the anchor axis and the protect-minus-rule orientation inside ``V_care``
    (rows orthonormal); negative = anti-aligned (author: log it in ANOMALIES)."""
    d = protect.mean(0) - rule.mean(0)
    o = v_care.T @ (v_care @ d)
    return float(o @ axis / (np.linalg.norm(o) * np.linalg.norm(axis)))


def margins(resid: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (resid @ axis) / np.linalg.norm(resid, axis=-1).clip(1e-6)


def chat_sigma(resid: np.ndarray, provenance: dict) -> tuple[np.ndarray, dict]:
    """§5.3 / Amendment 1: per-dimension std of the model's own judgment-frame residuals at the
    pre-decision gate, layer L*, over H1_train; never reused across models or formats."""
    need = {"model", "revision", "format", "position_class", "layer", "split"}
    missing = need - set(provenance)
    if missing:
        raise ValueError(f"chat_sigma provenance missing {sorted(missing)}")
    s = resid.std(0, ddof=1).astype(np.float32)
    return s, {**provenance, "n": int(resid.shape[0]), "min": float(s.min()),
               "median": float(np.median(s))}


def dpo_pairs(rows: list[dict]) -> tuple[list[dict], dict]:
    """§5.1: per scenario, chosen = a ``protect`` rollout, rejected = a ``rule`` rollout, paired in
    rollout order; ``defer`` (and unparsed) rollouts are never in a pair. Counts per scenario."""
    by: dict[str, dict[str, list]] = {}
    for r in rows:
        if r.get("action") in ("protect", "rule"):
            by.setdefault(r["scenario_id"], {"protect": [], "rule": []})[r["action"]].append(r)
    pairs, counts = [], {}
    for sid, d in sorted(by.items()):
        k = min(len(d["protect"]), len(d["rule"]))
        counts[sid] = k
        pairs += [{"scenario_id": sid, "prompt": d["protect"][i]["prompt_ids"],
                   "chosen": d["protect"][i]["gen_ids"], "rejected": d["rule"][i]["gen_ids"]}
                  for i in range(k)]
    return pairs, {"n_pairs": len(pairs), "per_scenario": counts,
                   "scenarios_with_pair": sum(v > 0 for v in counts.values()),
                   "floor": PAIR_FLOOR, "meets_floor": len(pairs) >= PAIR_FLOOR}


def run(model, objective: str, *, layer: int, traces: list[dict] | None = None,
        pairs: list[dict] | None = None, general: list[list[int]] = (), kl_weight: float = 1.0,
        axis: np.ndarray | None = None, tau: float | None = None,
        targets: dict[str, np.ndarray] | None = None, sigma: np.ndarray | None = None,
        max_steps: int = 100, lr: float = 1e-4, grad_accum: int = 1, seed: int = 0,
        eval_fn=None, eval_every: int = 0, lora_rank: int = 16, lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        allow_empty_kl: bool = False) -> dict:
    """Train one adapter. ``traces``: rows with ``ids``, ``anchor_mask``, ``kl_mask`` and
    ``scenario_id`` (anchor / self); ``pairs``: DPO pairs (beh); ``general``: token sequences for
    the general-slice KL term, weighted ``kl_weight`` in every arm.

    For anchor / self, a KL term that is identically zero (no trace has a KL position and there is
    no general slice) is refused unless ``allow_empty_kl`` (author, 2026-10-19: T1.1's QLoRA
    shakedown ran with an empty mask and so never exercised the term)."""
    import torch

    n_kl = sum(sum(bool(x) for x in t["kl_mask"]) for t in traces or [])
    if (objective in ("anchor", "self") and n_kl == 0 and not general and kl_weight
            and not allow_empty_kl):
        raise ValueError("KL-to-base term is identically zero: no trace has a KL position and "
                         "there is no general slice")

    # dropout 0: PEFT's expert-parameter LoRA (GPT-OSS's experts) refuses any other value
    lora = OL.inject_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    dev = model.model.device
    ax = torch.as_tensor(axis, device=dev) if axis is not None else None
    sg = torch.as_tensor(sigma, device=dev) if sigma is not None else None
    tr_it = OL.micro_batches(traces or [], seed) if traces else None
    pr_it = OL.micro_batches(pairs or [], seed) if pairs else None
    gen_it = OL.micro_batches(list(general), seed + 1) if general else None

    def step_loss(_micro: int):
        kl_g = OL.kl_general(model, next(gen_it)) if gen_it is not None else None
        if objective == "beh":
            p = next(pr_it)
            loss, info = OL.dpo_loss(model, p["prompt"], p["chosen"], p["rejected"],
                                     beta=DPO_BETA)
            kl = kl_g if kl_g is not None else loss * 0.0
            return loss + kl_weight * kl, None, float(kl.detach()), info
        t = next(tr_it)
        tgt = (torch.as_tensor(targets[t["scenario_id"]], device=dev)
               if objective == "self" else None)
        rep, kl_t, info = OL.anchor_and_kl(
            model, t["ids"], layer=layer, anchor_mask=t["anchor_mask"], kl_mask=t["kl_mask"],
            kind="anchor" if objective == "anchor" else "self", direction=ax, tau=tau,
            target=tgt, sigma=sg)
        kl = kl_t + (kl_g if kl_g is not None else 0.0)
        return rep + kl_weight * kl, float(rep.detach()), float(kl.detach()), info

    if objective not in ("beh", "anchor", "self"):
        raise ValueError(objective)
    steps, snaps = OL.train(model, step_loss, max_steps=max_steps, lr=lr, grad_accum=grad_accum,
                            seed=seed, eval_fn=eval_fn, eval_every=eval_every)
    return {"objective": objective, "layer": layer, "lora": lora, "kl_weight": kl_weight,
            "tau": tau, "dpo_beta": DPO_BETA if objective == "beh" else None,
            "n_traces": len(traces or []), "n_pairs": len(pairs or []), "n_kl_positions": n_kl,
            "steps": [s.__dict__ for s in steps], "eval": snaps,
            "max_steps": max_steps, "lr": lr, "grad_accum": grad_accum, "seed": seed}


def save_adapter(model, out: Path, record: dict, sidecar: dict) -> None:
    """The adapter (PEFT ``save_pretrained``) and its sidecar: the training record plus the
    provenance fields (anchor axis, sigma, class, tau) the plan requires."""
    out.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(str(out))
    (out / "st_adapter.json").write_text(json.dumps({**record, "sidecar": sidecar}, indent=1,
                                                    default=str))


def action_counts(rows: list[dict]) -> dict:
    return dict(Counter(r.get("action") for r in rows))
