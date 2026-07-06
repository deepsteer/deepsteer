#!/usr/bin/env python3
"""Paper 1 §4.3 confound check: per-condition activation RMS + RMS-normalized sigma*.

Measures whether "declarative = most fragile" survives scale-matching.
Reuses the exact fragility recipe (seed 42, 50 epochs, lr 1e-2, thr 0.6,
grid [0.1,0.3,1,3,10], 10 noise seeds, cap-at-max, mean over layers) and the
exact LoRATrainer so the fine-tunes reproduce the paper conditions.

Outputs JSON to papers/1_accuracy_vs_fragility/outputs/phase_c_tier2/c3/confound_rms/.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import AccessTier
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
from deepsteer.steering.lora_trainer import LoRATrainer

REPO = "allenai/OLMo-2-0425-1B-early-training"
REV = "stage1-step1000-tokens3B"
STEP = 1000
NOISE = [0.1, 0.3, 1.0, 3.0, 10.0]
N_SEEDS = 10
THR = 0.6
SEED = 42
N_EPOCHS = 50
LR = 1e-2
OUT = Path("papers/1_accuracy_vs_fragility/outputs/phase_c_tier2/c3/confound_rms")


def _train_probe(X, y):
    torch.manual_seed(SEED)
    p = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(p.parameters(), lr=LR)
    lf = nn.BCEWithLogitsLoss()
    p.train()
    for _ in range(N_EPOCHS):
        loss = lf(p(X).squeeze(-1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    p.eval()
    return p


def _acc(p, X, y):
    with torch.no_grad():
        return ((p(X).squeeze(-1) > 0).float() == y).float().mean().item()


def _layer_sigma(trX, trY, teX, teY, rms_norm):
    if rms_norm:
        rms = trX.pow(2).mean().sqrt().clamp_min(1e-8)
        trX = trX / rms; teX = teX / rms
    p = _train_probe(trX, trY)
    cap = max(NOISE)
    sigma = cap
    for nl in NOISE:
        accs = []
        for s in range(N_SEEDS):
            torch.manual_seed(SEED + s)
            accs.append(_acc(p, teX + torch.randn_like(teX) * nl, teY))
        if float(np.mean(accs)) < THR:
            sigma = nl; break
    return sigma


def _condition_metrics(train_acts, test_acts, n_layers):
    per_rms, sig_raw, sig_rms = [], [], []
    for L in range(n_layers):
        trX, trY = train_acts[L]; teX, teY = test_acts[L]
        per_rms.append(float(teX.pow(2).mean().sqrt()))
        sig_raw.append(_layer_sigma(trX, trY, teX, teY, False))
        sig_rms.append(_layer_sigma(trX, trY, teX, teY, True))
    return {
        "per_layer_rms": [round(r, 4) for r in per_rms],
        "mean_rms": round(float(np.mean(per_rms)), 4),
        "per_layer_sigma_raw": sig_raw,
        "per_layer_sigma_rms": sig_rms,
        "mean_sigma_raw": round(float(np.mean(sig_raw)), 4),
        "mean_sigma_rms": round(float(np.mean(sig_rms)), 4),
        "n_fragile_raw": int(sum(1 for s in sig_raw if s < max(NOISE))),
    }


def _collect(model, dataset):
    tr = LayerWiseMoralProbe._collect_all_activations(model, dataset.train)
    te = LayerWiseMoralProbe._collect_all_activations(model, dataset.test)
    return tr, te


def _load():
    return WhiteBoxModel(REPO, revision=REV, access_tier=AccessTier.CHECKPOINTS,
                         checkpoint_step=STEP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--conditions", default="declarative,narrative,general")
    ap.add_argument("--max-tokens", type=int, default=500_000)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = build_probing_dataset(target_per_foundation=40)
    print(f"dataset: {len(dataset.train)} train pairs, {len(dataset.test)} test pairs")

    results = {"config": {"steps": args.steps, "max_tokens": args.max_tokens,
                          "noise_grid": NOISE, "repo": REPO, "rev": REV}}
    outp = OUT / f"confound_rms_{args.tag}.json"

    def _save():
        with open(outp, "w") as f:
            json.dump(results, f, indent=2)

    # BASE (no fine-tune)
    t0 = time.time()
    model = _load()
    n_layers = model.info.n_layers
    tr, te = _collect(model, dataset)
    results["base"] = _condition_metrics(tr, te, n_layers)
    _save()
    print(f"BASE mean_rms={results['base']['mean_rms']} "
          f"sigma_raw={results['base']['mean_sigma_raw']} "
          f"sigma_rms={results['base']['mean_sigma_rms']} ({time.time()-t0:.0f}s)")
    del model; import gc; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    from deepsteer.datasets.corpora import (load_declarative_corpus,
        load_narrative_corpus, load_general_corpus)
    loaders = {"declarative": load_declarative_corpus,
               "narrative": load_narrative_corpus,
               "general": load_general_corpus}

    for cond in args.conditions.split(","):
        cond = cond.strip()
        if not cond: continue
        print(f"\n=== {cond} ===")
        t0 = time.time()
        try:
            corpus = loaders[cond](max_tokens=args.max_tokens)
        except Exception as e:
            print(f"  CORPUS FAILED ({cond}): {type(e).__name__}: {str(e)[:150]}")
            results[cond] = {"error": f"corpus_load: {type(e).__name__}: {str(e)[:150]}"}
            _save()
            continue
        print(f"  corpus: {len(corpus)} chunks ({time.time()-t0:.0f}s)")
        model = _load()
        trainer = LoRATrainer(model, corpus, dataset, lora_rank=16, lora_alpha=32,
                              max_steps=args.steps, eval_every=args.steps + 10,
                              run_fragility=False)
        tt = time.time()
        res = trainer.train(experiment_id=f"confound_{cond}", corpus_name=cond)
        train_s = time.time() - tt
        losses = [s.loss for s in res.training_steps]
        # model is now merged fine-tuned
        tr, te = _collect(model, dataset)
        m = _condition_metrics(tr, te, n_layers)
        m["loss_first"] = round(losses[0], 3) if losses else None
        m["loss_last"] = round(float(np.mean(losses[-20:])), 3) if losses else None
        m["train_time_s"] = round(train_s, 1)
        m["per_step_s"] = round(train_s / max(args.steps, 1), 3)
        results[cond] = m
        _save()
        print(f"  {cond}: loss {m['loss_first']}->{m['loss_last']} "
              f"mean_rms={m['mean_rms']} sigma_raw={m['mean_sigma_raw']} "
              f"sigma_rms={m['mean_sigma_rms']} n_fragile={m['n_fragile_raw']} "
              f"({m['per_step_s']}s/step, {train_s:.0f}s)")
        del model, trainer; gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    outp = OUT / f"confound_rms_{args.tag}.json"
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
