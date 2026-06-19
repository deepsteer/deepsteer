#!/usr/bin/env python3
"""Forced-coupling intervention, Stage 1 (limited, specificity-bounded).

Tier 2 found the proto-refusal->MFT projection is FLAT at ~0.107 across all 13
stage-3 checkpoints (= base 0.108, = Instruct 0.104), never approaching the 0.40
coupling threshold: there is no natural window to ride. Stage 1 asks the narrow
go/no-go question: can a bounded regularizer move that projection AT ALL, without
destroying LM quality and WITHOUT reproducing the Section 6 degenerate
non-specific solution (where ART drove dependency to -0.62, hurting neutral text
more than moral)? This is a representational pre-check, NOT the full
intervene->SFT->Heretic pipeline.

The regularizer (specified against the real ops, reusing the ART primitives in
``deepsteer.steering.ablation_resistance``):

    V        = SVD-orthonormal 6-foundation basis at the target layer(s), frozen
    Delta    = last_token_means(harmful) - last_token_means(harmless)   # LIVE, per step
    e_in     = ||Delta @ V^T||^2 / ||Delta||^2     # in-subspace ENERGY fraction
    L_couple = relu(target_proj^2 - e_in)           # hinge: push e_in up to target, then stop
    L_total  = L_lm + lambda * L_couple

``Delta`` is recomputed from the CURRENT model each step (it is the thing being
moved). The projection is exactly ``_make_projection_hook``'s math applied to
``Delta`` instead of the residual stream.

CONVENTION NOTE (corrects the brief). The Tier-2 number 0.107 is the projection
NORM-RATIO (||proj||/||Delta||), not an energy fraction. So in-subspace energy
starts at 0.107^2 ~ 0.0115 and the orthogonal-energy fraction starts at ~0.988
(NOT 0.893). We optimise the energy form (smooth, bounded in [0,1]) but report the
norm-ratio ``sqrt(e_in)`` as the headline metric, directly comparable to the 0.107
baseline and the 0.40 threshold. Hinge target is on the norm-ratio (``--target-proj``,
default 0.40), converted to energy internally.

Specificity guards (PRIMARY; these define "limited", not lambda alone). A
"successful" L_couple drop that trips any guard is the Section 6 degenerate
solution re-emerging one level up; it is reported as such, NOT counted as a win:
  1. neutral-text LM loss vs moral-text LM loss (the -0.62 signature: neutral
     degrades >= moral).
  2. moral probe accuracy (collapse detector; optional --probe-monitor).
  3. OFF-TARGET contrast: a neutral topic-A/topic-B Delta's projection onto V. If
     it rises in lockstep with the refusal Delta's, V is becoming a sink for ANY
     contrast (the cleanest discriminator, most important new control).
  4. general-text perplexity within a band of the un-intervened checkpoint.

Capacity ladder (Section 6: r16 q/v was too weak to reshape representation):
  --capacity r16_qv      (rung 1; expected too weak; cheap null)
  --capacity r64_qv_mlp  (rung 2; rank 64 + MLP incl. down_proj where dependency lived)
  --capacity full        (rung 3; full-parameter continued-pretrain; needs the headroom)

NO SFT, NO Heretic, NO full pipeline. Bounded continued-pretrain only. Hard stop
after Stage 1 for human review.

Usage (RunPod A100):
    python papers/5_moral_alignment/scripts/forced_coupling_stage1.py \
        --model allenai/Olmo-3-1025-7B --revision stage3-step11921 \
        --moral-npz .../outputs/pipeline/olmo3_pretrain_stage3_step11921/probe_directions.npz \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --capacity r16_qv --max-steps 300 --device cuda \
        --output-dir papers/5_moral_alignment/outputs/intervention_stage1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from deepsteer.steering.ablation_resistance import build_subspace_basis, load_foundation_directions

logger = logging.getLogger(__name__)

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_DEF_PROMPTS = _PAPER_ROOT / "refusal_prompts.json"

# Capacity ladder presets: (lora_rank | None for full, lora_alpha, target_modules).
CAPACITY = {
    "r16_qv": (16, 32, ["q_proj", "v_proj"]),
    "r64_qv_mlp": (64, 128, ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]),
    "full": (None, None, None),
}

# Off-target neutral contrast for Guard 3: matched-register, NON-moral, non-harm
# topic split (weather/nature vs computing/tech). If V starts absorbing THIS
# contrast too, it is becoming a sink for any contrast, not coupling to refusal.
NEUTRAL_CONTRAST_A = [
    "The morning fog settled over the quiet valley before sunrise.",
    "A light breeze carried the scent of pine across the meadow.",
    "Rain tapped steadily on the tin roof through the afternoon.",
    "The river widened as it neared the gravel delta downstream.",
    "Frost coated the fence posts along the edge of the field.",
    "Clouds gathered slowly above the ridge as the day cooled.",
    "The tide pulled back to reveal smooth stones on the shore.",
    "Tall grass swayed in waves across the open prairie.",
    "Snow began to fall as the temperature dropped after dusk.",
    "A pair of herons waded through the shallow marsh at dawn.",
    "The canyon walls glowed amber in the late evening light.",
    "Dew collected on the petals in the cool early hours.",
    "The forest floor was thick with damp leaves after the storm.",
    "Sunlight broke through the canopy and dappled the trail.",
    "A thin stream trickled between the mossy rocks of the gorge.",
    "The desert air turned sharp and cold once the sun had set.",
]
NEUTRAL_CONTRAST_B = [
    "The laptop synced its files to the server over the network.",
    "A new firmware update installed automatically overnight.",
    "The database indexed the records to speed up the queries.",
    "She compiled the source code and ran the unit tests.",
    "The router assigned each device a local IP address.",
    "The spreadsheet recalculated the totals after the edit.",
    "He cleared the browser cache to reload the latest version.",
    "The script parsed the log file and counted the errors.",
    "The monitor refreshed at a higher rate after the change.",
    "The app cached the images so the gallery loaded faster.",
    "The compiler flagged a missing semicolon on that line.",
    "The backup job copied the archive to external storage.",
    "The API returned the results as a compact JSON payload.",
    "The keyboard shortcut toggled the terminal in the editor.",
    "The container booted and exposed its port to the host.",
    "The function returned the sorted list in linear time.",
]


# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------


def load_prompts(path: str, *, allow_fallback: bool) -> tuple[list[str], list[str]]:
    """Heretic harmful/harmless prompts; reject the placeholder set (Stage-1 needs Δ)."""
    ps = json.load(open(path))
    harmful, harmless = ps["harmful"], ps["harmless"]
    looks_fallback = ps.get("provenance") is None or min(len(harmful), len(harmless)) < 50
    if looks_fallback and not allow_fallback:
        raise RuntimeError(f"{path} looks like the placeholder set; pass --allow-fallback "
                           "only for a throwaway smoke.")
    return harmful, harmless


def load_text_pools(
    general_jsonl: str | None, max_general: int
) -> tuple[list[str], list[str], list[str]]:
    """Return (moral, neutral, general) text pools.

    moral/neutral come from the probing dataset (matched pairs, the Guard-1 arms).
    general is the continued-pretrain LM corpus: ``--general-jsonl`` (one doc per
    line, raw text or ``{"text": ...}``) if given, else a smoke fallback of the
    probing texts (documented; pass a real general corpus for the actual run).
    """
    from deepsteer.datasets.pipeline import build_probing_dataset
    ds = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
    pairs = list(ds.train) + list(ds.test)
    moral = [p.moral for p in pairs]
    neutral = [p.neutral for p in pairs]

    if general_jsonl and Path(general_jsonl).exists():
        general = []
        for ln in open(general_jsonl):
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                general.append(obj["text"] if isinstance(obj, dict) and "text" in obj else ln)
            except json.JSONDecodeError:
                general.append(ln)
            if len(general) >= max_general:
                break
    else:
        logger.warning("No --general-jsonl; using probing texts as the SMOKE general "
                       "corpus. Pass a real general corpus for the production run.")
        general = (moral + neutral)
    return moral, neutral, general


# ---------------------------------------------------------------------------
# Forced-coupling regularizer
# ---------------------------------------------------------------------------


class ForcedCouplingRegularizer:
    """Drive the harmful/harmless contrast Δ into the moral subspace V.

    Mirrors ``AblationResistanceSteering`` (frozen V, live activations, hinge,
    calibration) but the objective is the GEOMETRY of Δ, not a generation gap.
    """

    def __init__(
        self,
        moral_npz: str,
        harmful: list[str],
        harmless: list[str],
        *,
        direction_kind: str = "probe",
        layers: list[int] | None = None,
        target_proj: float = 0.40,
        coefficient: float = 0.1,
        max_coefficient: float = 1.0,
        pos_batch: int = 8,
        neg_batch: int = 8,
        max_len: int = 64,
        monitor_sample: int = 128,
        seed: int = 0,
    ) -> None:
        self._directions = load_foundation_directions(moral_npz)
        self._harmful, self._harmless = harmful, harmless
        self._kind = direction_kind
        self._requested_layers = layers
        self._target_energy = float(target_proj) ** 2
        self.coefficient = float(coefficient)
        self._max_coefficient = float(max_coefficient)
        self._pos_batch, self._neg_batch = pos_batch, neg_batch
        self._max_len = max_len
        self._monitor_sample = monitor_sample
        self._rng = random.Random(seed)
        self._layers: list[int] = []
        self._V: dict[int, Tensor] = {}
        self._pos_tok: list[Tensor] = []
        self._neg_tok: list[Tensor] = []
        self._pos_cur = self._neg_cur = 0
        self._pos_ord: list[int] = []
        self._neg_ord: list[int] = []
        self._tok = None
        self._pad = 0

    def attach(self, model) -> None:
        n_layers = model.info.n_layers
        basis_np, ranks, names = build_subspace_basis(
            self._directions, kind=self._kind, n_layers=n_layers)
        if not basis_np:
            raise RuntimeError("No complete moral subspace; check --moral-npz.")
        avail = sorted(basis_np)
        self._layers = ([L for L in avail if L in set(self._requested_layers)]
                        if self._requested_layers is not None else avail)
        if not self._layers:
            raise RuntimeError(f"None of --layers present in V (have {avail[:5]}...).")
        param = next(model.model.parameters())
        self._V = {L: torch.from_numpy(basis_np[L]).to(device=param.device, dtype=torch.float32)
                   for L in self._layers}
        self._tok = model.tokenizer
        self._pad = self._tok.pad_token_id or self._tok.eos_token_id or 0
        self._pos_tok = [self._tok(t, truncation=True, max_length=self._max_len,
                                   return_tensors="pt")["input_ids"][0] for t in self._harmful]
        self._neg_tok = [self._tok(t, truncation=True, max_length=self._max_len,
                                   return_tensors="pt")["input_ids"][0] for t in self._harmless]
        self._pos_ord = list(range(len(self._pos_tok)))
        self._neg_ord = list(range(len(self._neg_tok)))
        self._rng.shuffle(self._pos_ord)
        self._rng.shuffle(self._neg_ord)
        logger.info("Coupling regularizer attached: %d directions, %d layers (%s), "
                    "target_proj=%.2f, λ=%.4f", len(names), len(self._layers),
                    self._layers, self._target_energy ** 0.5, self.coefficient)

    def _next(self, tok, order, cur, n):
        out = []
        for _ in range(min(n, len(tok))):
            if cur >= len(order):
                self._rng.shuffle(order)
                cur = 0
            out.append(tok[order[cur]])
            cur += 1
        return out, cur

    def _pad_batch(self, seqs, device):
        maxlen = max(s.shape[0] for s in seqs)
        ids = torch.full((len(seqs), maxlen), self._pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :s.shape[0]] = s
            attn[i, :s.shape[0]] = 1
        return ids.to(device), attn.to(device)

    def _delta_by_layer(self, model, pos_seqs, neg_seqs, *, grad: bool) -> dict[int, Tensor]:
        """Δ_L = mean(last-token@L | harmful) − mean(... | harmless), per layer."""
        device = next(model.model.parameters()).device
        seqs = pos_seqs + neg_seqs
        n_pos = len(pos_seqs)
        ids, attn = self._pad_batch(seqs, device)
        captured: dict[int, Tensor] = {}
        handles = []

        def mk(L):
            def hook(_m, _i, out):
                captured[L] = out[0] if isinstance(out, tuple) else out
            return hook
        for L in self._layers:
            handles.append(model._get_layer_module(L).register_forward_hook(mk(L)))
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                model.model(input_ids=ids, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()
        last_idx = attn.sum(dim=1) - 1
        rows = torch.arange(ids.shape[0], device=device)
        out: dict[int, Tensor] = {}
        for L in self._layers:
            last = captured[L][rows, last_idx].float()  # (B, H)
            out[L] = last[:n_pos].mean(0) - last[n_pos:].mean(0)
        return out

    def _energy_in(self, delta: Tensor, L: int) -> Tensor:
        d = delta.float()
        d_in = d @ self._V[L].t()
        return (d_in @ d_in) / (d @ d + 1e-12)

    def couple_loss(self, model) -> tuple[Tensor, dict]:
        """Hinge loss + metrics on a fresh minibatch (graph-attached)."""
        pos, self._pos_cur = self._next(
            self._pos_tok, self._pos_ord, self._pos_cur, self._pos_batch)
        neg, self._neg_cur = self._next(
            self._neg_tok, self._neg_ord, self._neg_cur, self._neg_batch)
        deltas = self._delta_by_layer(model, pos, neg, grad=True)
        per_layer_e, losses = {}, []
        for L in self._layers:
            e = self._energy_in(deltas[L], L)
            per_layer_e[L] = float(e.detach().item())
            losses.append(torch.clamp(self._target_energy - e, min=0.0))
        loss = torch.stack(losses).mean()
        ratio = float(np.mean([e ** 0.5 for e in per_layer_e.values()]))
        return loss, {"proj_ratio_mean": round(ratio, 6),
                      "proj_ratio_by_layer": {str(L): round(per_layer_e[L] ** 0.5, 6)
                                              for L in self._layers}}

    @torch.no_grad()
    def projection_ratio(self, model, pos_texts=None, neg_texts=None) -> dict:
        """No-grad projection norm-ratio onto V over a large sample (monitoring)."""
        pos_t = ([self._tok(t, truncation=True, max_length=self._max_len,
                            return_tensors="pt")["input_ids"][0] for t in pos_texts]
                 if pos_texts is not None else self._pos_tok)
        neg_t = ([self._tok(t, truncation=True, max_length=self._max_len,
                            return_tensors="pt")["input_ids"][0] for t in neg_texts]
                 if neg_texts is not None else self._neg_tok)
        ns = min(self._monitor_sample, len(pos_t), len(neg_t))
        # chunk to keep memory bounded
        chunk = 16
        sums_pos = {L: None for L in self._layers}
        sums_neg = {L: None for L in self._layers}
        for tlist, sums in ((pos_t[:ns], sums_pos), (neg_t[:ns], sums_neg)):
            for i in range(0, len(tlist), chunk):
                seqs = tlist[i:i + chunk]
                d = self._delta_by_layer_means(model, seqs)
                for L in self._layers:
                    sums[L] = d[L] if sums[L] is None else sums[L] + d[L]
        out = {}
        for L in self._layers:
            delta = (sums_pos[L] - sums_neg[L])
            out[L] = float(self._energy_in(delta, L).item()) ** 0.5
        return {"proj_ratio_mean": round(float(np.mean(list(out.values()))), 6),
                "proj_ratio_by_layer": {str(L): round(out[L], 6) for L in self._layers}}

    @torch.no_grad()
    def _delta_by_layer_means(self, model, seqs) -> dict[int, Tensor]:
        """Per-layer SUM of last-token activations over a chunk (for monitoring means)."""
        device = next(model.model.parameters()).device
        ids, attn = self._pad_batch(seqs, device)
        captured, handles = {}, []

        def mk(L):
            def hook(_m, _i, out):
                captured[L] = out[0] if isinstance(out, tuple) else out
            return hook
        for L in self._layers:
            handles.append(model._get_layer_module(L).register_forward_hook(mk(L)))
        try:
            model.model(input_ids=ids, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()
        last_idx = attn.sum(dim=1) - 1
        rows = torch.arange(ids.shape[0], device=device)
        return {L: captured[L][rows, last_idx].float().sum(0) / 1.0 for L in self._layers}

    def calibrate(self, lm_loss: float, couple_loss: float, ratio: float = 0.5) -> float:
        lam = ratio * max(lm_loss, 1e-6) / max(couple_loss, 1e-6)
        self.coefficient = min(lam, self._max_coefficient)
        return self.coefficient


# ---------------------------------------------------------------------------
# Guards (no-grad)
# ---------------------------------------------------------------------------


@torch.no_grad()
def lm_loss(model, texts: list[str], *, max_len: int = 128, n: int = 64) -> float:
    """Token-weighted mean LM loss over up to n texts."""
    tok = model.tokenizer
    device = next(model.model.parameters()).device
    tot_nll, tot_tok = 0.0, 0
    for t in texts[:n]:
        ids = tok(t, truncation=True, max_length=max_len,
                  return_tensors="pt")["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        loss = model.model(input_ids=ids, labels=ids).loss
        ntok = ids.shape[1] - 1
        tot_nll += float(loss.item()) * ntok
        tot_tok += ntok
    return tot_nll / tot_tok if tot_tok else float("nan")


def guard_verdict(rec0: dict, rec: dict, *, ppl_band: float) -> dict:
    """Evaluate the four guards against the step-0 baseline. green=True is pass."""
    g = {}
    # 1. neutral degrades >= moral (the -0.62 signature)
    d_moral = rec["lm_moral"] - rec0["lm_moral"]
    d_neutral = rec["lm_neutral"] - rec0["lm_neutral"]
    g["guard1_neutral_not_worse"] = bool(d_neutral <= max(d_moral, 0.0) + 1e-4)
    # 2. probe acc not collapsed (if monitored)
    if rec.get("probe_acc") is not None and rec0.get("probe_acc") is not None:
        g["guard2_probe_ok"] = bool(rec["probe_acc"] >= rec0["probe_acc"] - 0.1)
    else:
        g["guard2_probe_ok"] = None
    # 3. off-target contrast not rising with the refusal contrast
    d_ref = rec["proj_refusal"] - rec0["proj_refusal"]
    d_off = rec["proj_neutral_contrast"] - rec0["proj_neutral_contrast"]
    g["guard3_offtarget_flat"] = bool(d_off <= max(0.4 * max(d_ref, 0.0), 0.02))
    # 4. general ppl within band
    g["guard4_ppl_band"] = bool(rec["lm_general"] <= rec0["lm_general"] + ppl_band)
    greens = [v for v in g.values() if v is not None]
    g["all_green"] = bool(all(greens))
    return g


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def inject_lora(model, rank, alpha, modules):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=modules,
                     lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model._model = get_peft_model(model._model, cfg)
    trainable = sum(p.numel() for p in model._model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model._model.parameters())
    logger.info("LoRA injected: %d trainable (%.3f%% of %d)", trainable,
                100 * trainable / total, total)


def setup_full_param(model):
    for p in model._model.parameters():
        p.requires_grad_(True)
    if hasattr(model._model, "gradient_checkpointing_enable"):
        model._model.gradient_checkpointing_enable()
    logger.warning("FULL-PARAMETER mode: high memory. Ensure the card has headroom "
                   "(consider 8-bit optimizer / multi-GPU if it OOMs).")


def make_lm_batches(model, corpus, seq_len, batch_size, seed):
    tok = model.tokenizer
    ids: list[int] = []
    for t in corpus:
        ids.extend(tok.encode(t, add_special_tokens=False))
        if tok.eos_token_id is not None:
            ids.append(tok.eos_token_id)
    n_chunks = max(1, len(ids) // seq_len)
    ids = ids[:n_chunks * seq_len]
    if len(ids) < seq_len:
        ids = ids + [tok.eos_token_id or 0] * (seq_len - len(ids))
        n_chunks = 1
    t = torch.tensor(ids[:n_chunks * seq_len], dtype=torch.long).view(n_chunks, seq_len)
    while t.shape[0] < batch_size:  # tiny corpus (smoke): tile up to one full batch
        t = torch.cat([t, t], dim=0)
    import itertools

    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=True, drop_last=True,
                    generator=torch.Generator().manual_seed(seed))
    return itertools.cycle(iter(dl))


def run_stage1(model, reg, moral, neutral, general, args) -> dict:
    device = next(model.model.parameters()).device
    rank, alpha, modules = CAPACITY[args.capacity]
    if rank is None:
        setup_full_param(model)
    else:
        inject_lora(model, rank, alpha, modules)
    reg.attach(model)

    probe_monitor = None
    if args.probe_monitor:
        from deepsteer.steering import ProbeMonitor
        probe_monitor = ProbeMonitor(model)

    lm_iter = make_lm_batches(model, general, args.seq_len, args.batch_size, args.seed)
    opt = torch.optim.AdamW([p for p in model._model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def snapshot(step):
        model._model.eval()
        off = reg.projection_ratio(model, pos_texts=NEUTRAL_CONTRAST_A,
                                   neg_texts=NEUTRAL_CONTRAST_B)["proj_ratio_mean"]
        rec = {"step": step,
               "proj_refusal": reg.projection_ratio(model)["proj_ratio_mean"],
               "proj_neutral_contrast": off,
               "lm_moral": lm_loss(model, moral), "lm_neutral": lm_loss(model, neutral),
               "lm_general": lm_loss(model, general)}
        if probe_monitor is not None:
            rec["probe_acc"] = float(probe_monitor.snapshot(step).peak_accuracy)
        model._model.train()
        return rec

    # ---- calibrate λ on the first batch ----
    if args.calibrate:
        model._model.train()
        with torch.no_grad():
            b0 = next(lm_iter)[0].to(device)
            l0 = float(model.model(input_ids=b0, labels=b0).loss.item())
        cl0, _ = reg.couple_loss(model)
        reg.calibrate(l0, float(cl0.detach().item()), ratio=args.calibrate_ratio)
        opt.zero_grad()
        logger.info("Calibrated λ=%.4f (lm=%.3f, couple0=%.3f)", reg.coefficient, l0,
                    float(cl0.detach().item()))

    records = [snapshot(0)]
    rec0 = records[0]
    records[0]["guards"] = guard_verdict(rec0, rec0, ppl_band=args.ppl_band)
    logger.info("step 0: proj_ref=%.4f proj_neu=%.4f lm[m/n/g]=%.3f/%.3f/%.3f",
                rec0["proj_refusal"], rec0["proj_neutral_contrast"],
                rec0["lm_moral"], rec0["lm_neutral"], rec0["lm_general"])

    model._model.train()
    t0 = time.time()
    breached_at = None
    for step in range(1, args.max_steps + 1):
        opt.zero_grad()
        lm_ids = next(lm_iter)[0].to(device)
        l_lm = model.model(input_ids=lm_ids, labels=lm_ids).loss
        if torch.isnan(l_lm):
            logger.warning("NaN LM loss at step %d; skipping", step)
            continue
        l_lm.backward()                       # free the LM graph first (split backward)
        couple, cmetrics = reg.couple_loss(model)
        (reg.coefficient * couple).backward()  # separate graph; grads accumulate on LoRA
        torch.nn.utils.clip_grad_norm_(
            [p for p in model._model.parameters() if p.requires_grad], args.max_grad_norm)
        opt.step()
        sched.step()
        if step % 10 == 0:
            logger.info("step %d/%d lm=%.4f couple=%.4f proj=%.4f λ=%.3f (%.0fs)",
                        step, args.max_steps, float(l_lm.item()), float(couple.item()),
                        cmetrics["proj_ratio_mean"], reg.coefficient, time.time() - t0)
        if step % args.eval_every == 0 or step == args.max_steps:
            rec = snapshot(step)
            rec["guards"] = guard_verdict(rec0, rec, ppl_band=args.ppl_band)
            rec["lm_step"] = round(float(l_lm.item()), 4)
            rec["couple_step"] = round(float(couple.item()), 4)
            records.append(rec)
            logger.info("  eval %d: proj_refusal=%.4f (Δ%+.4f) proj_neutral=%.4f guards_green=%s",
                        step, rec["proj_refusal"], rec["proj_refusal"] - rec0["proj_refusal"],
                        rec["proj_neutral_contrast"], rec["guards"]["all_green"])
            if not rec["guards"]["all_green"] and breached_at is None:
                breached_at = step
                if args.stop_on_breach:
                    logger.warning("Guard breach at step %d; stopping (--stop-on-breach).", step)
                    break

    last = records[-1]
    moved = last["proj_refusal"] - rec0["proj_refusal"]
    if last["guards"]["all_green"] and moved > 0.02:
        verdict = "moves_guards_green"
    elif moved > 0.02:
        verdict = "moves_only_degenerately"
    else:
        verdict = "no_move"
    return {
        "verdict": verdict,
        "capacity": args.capacity,
        "lora": None if rank is None else {"rank": rank, "alpha": alpha, "modules": modules},
        "baseline_proj_refusal": rec0["proj_refusal"],
        "final_proj_refusal": last["proj_refusal"],
        "projection_moved": round(moved, 6),
        "target_proj": args.target_proj,
        "first_guard_breach_step": breached_at,
        "records": records,
        "elapsed_s": round(time.time() - t0, 1),
    }


def write_result_md(out: Path, payload: dict) -> None:
    """Auto-generate STAGE1_RESULT.md from the trajectory (brief deliverable)."""
    r = payload
    final = r["records"][-1]
    g = final["guards"]
    route = {
        "moves_guards_green": "GREEN-LIGHT the full pipeline (Stage 2/3): coupling moved with "
                              "guards green.",
        "moves_only_degenerately": "DEEPER NEGATIVE: the Section 6 degenerate solution recurs at "
                                   "pre-training (moved only by tripping a guard). Surface it.",
        "no_move": "STALL: forced coupling did not move the projection at this rung. If the top "
                   "capacity rung also fails, the reserved subspace-robustness check (is V the "
                   "wrong target?) becomes the question.",
    }[r["verdict"]]
    lines = [
        f"# Stage 1 Forced-Coupling Result — {r['verdict']}",
        "",
        f"- **Capacity rung:** {r['capacity']}",
        f"- **Projection (proto-refusal -> MFT, norm-ratio):** "
        f"{r['baseline_proj_refusal']:.4f} -> {r['final_proj_refusal']:.4f} "
        f"(Δ{r['projection_moved']:+.4f}; target {r['target_proj']}, Tier-2 baseline ~0.107)",
        f"- **First guard breach:** step {r['first_guard_breach_step']}",
        "",
        "## Specificity guards at final step",
        f"- Guard 1 (neutral not worse than moral): {g['guard1_neutral_not_worse']}",
        f"- Guard 2 (probe acc, if monitored): {g['guard2_probe_ok']}",
        f"- Guard 3 (off-target contrast flat): {g['guard3_offtarget_flat']} "
        f"(neutral-contrast proj {final['proj_neutral_contrast']:.4f})",
        f"- Guard 4 (general ppl band): {g['guard4_ppl_band']}",
        f"- **all_green:** {g['all_green']}",
        "",
        "## Routing",
        route,
        "",
        "_Hard stop after Stage 1: do NOT proceed to SFT->Heretic without human review._",
    ]
    (out / "STAGE1_RESULT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forced-coupling intervention, Stage 1 (limited).")
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--revision", default="stage3-step11921",
                    help="Late stage-3 (most representative of the pre-trained start; "
                         "Tier 2 found no window, so timing is not the variable).")
    ap.add_argument("--moral-npz", required=True,
                    help="Per-checkpoint foundation directions (cached pipeline npz).")
    ap.add_argument("--direction-kind", choices=["probe", "meandiff"], default="probe")
    ap.add_argument("--prompts", default=str(_DEF_PROMPTS))
    ap.add_argument("--general-jsonl", default=None,
                    help="General LM corpus; smoke falls back to probing texts.")
    ap.add_argument("--capacity", choices=list(CAPACITY), default="r16_qv")
    ap.add_argument("--layers", default=None,
                    help="Comma-separated target layers (default: all V layers).")
    ap.add_argument("--target-proj", type=float, default=0.40, help="Hinge target (norm-ratio).")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.1)
    ap.add_argument("--max-lambda", type=float, default=1.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibrate-ratio", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pos-batch", type=int, default=8)
    ap.add_argument("--neg-batch", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--ppl-band", type=float, default=0.5,
                    help="Guard 4: max general-LM-loss rise (nats).")
    ap.add_argument("--probe-monitor", action="store_true",
                    help="Guard 2 via ProbeMonitor (slower).")
    ap.add_argument("--stop-on-breach", action="store_true", help="Halt on first guard breach.")
    ap.add_argument("--max-general", type=int, default=4000)
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=str(_PAPER_ROOT / "outputs/intervention_stage1"))
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    label = args.label or f"coupling_{args.capacity}"
    out = Path(args.output_dir) / label
    out.mkdir(parents=True, exist_ok=True)

    harmful, harmless = load_prompts(args.prompts, allow_fallback=args.allow_fallback)
    moral, neutral, general = load_text_pools(args.general_jsonl, args.max_general)

    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS,
                          revision=args.revision)
    print(f"Loaded {args.model}@{args.revision} ({model.info.n_layers}L) in {time.time()-t0:.1f}s")

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    reg = ForcedCouplingRegularizer(
        args.moral_npz, harmful, harmless, direction_kind=args.direction_kind,
        layers=layers, target_proj=args.target_proj, coefficient=args.lam,
        max_coefficient=args.max_lambda, pos_batch=args.pos_batch, neg_batch=args.neg_batch,
        seed=args.seed)

    result = run_stage1(model, reg, moral, neutral, general, args)
    model.release()

    payload = {
        "analysis": "forced_coupling_stage1",
        "model": args.model, "revision": args.revision, "moral_npz": args.moral_npz,
        "prompts": args.prompts, "general_corpus": args.general_jsonl or "PROBING_FALLBACK",
        "config": {k: getattr(args, k) for k in
                   ("capacity", "target_proj", "lam", "calibrate", "lr", "batch_size",
                    "seq_len", "pos_batch", "neg_batch", "max_steps", "ppl_band", "seed")},
        **result,
    }
    with open(out / "stage1_trajectory.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    write_result_md(out, payload)
    print(f"\nWrote {out/'stage1_trajectory.json'} and STAGE1_RESULT.md")
    print(f"  VERDICT: {result['verdict']} | proj {result['baseline_proj_refusal']:.4f} -> "
          f"{result['final_proj_refusal']:.4f} (Δ{result['projection_moved']:+.4f}, "
          f"target {args.target_proj}) | first breach: {result['first_guard_breach_step']}")


if __name__ == "__main__":
    main()
