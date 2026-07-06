#!/usr/bin/env python3
"""Paper 1 §4.3 DEFINITIVE confound check: analytic (censoring-free) sigma* + paired Delta-CI.

Pre-registered in papers/ANOMALIES.md A7. Resolves the coarse-grid censoring artifact that
manufactured a spurious sign-flip in the naive RMS-normalized arm.

Analytic sigma*: for a frozen linear probe f(x)=w.x+b, adding isotropic Gaussian noise
eps~N(0,sigma^2 I) to a test point gives logit ~ N(margin_i, sigma^2 ||w||^2), so
  acc(sigma) = mean_i Phi(margin_i / (sigma*||w||)),  margin_i = y_i*(w.x_i+b) signed (>0 correct).
This is continuous, monotone decreasing from clean-acc to 0.5, so sigma*(acc=thr) is always
finite (NO cap, NO censoring, NO noise seeds, NO grid). Two conventions:
  raw:  sigma*_raw(L)                       (absolute activation units)
  snr:  sigma*_raw(L) / rms(L)              (per-layer SNR, the scale-matched quantity)
mean over layers. Paired bootstrap over TEST examples for Delta = sigma*(narrative) - sigma*(declarative).

Branch A (flip confirmed: declarative LEAST fragile under a censoring-free estimator) -> P1v2 erratum.
Branch B (ordering preserved: declarative MOST fragile) -> section 4.3 survives + scale-sensitivity note.
"""
from __future__ import annotations
import argparse, json, time, math
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
THR = 0.6
SEED = 42
N_EPOCHS = 50
LR = 1e-2
N_BOOT = 2000
OUT = Path("papers/1_accuracy_vs_fragility/outputs/phase_c_tier2/c3/confound_analytic")
SQRT2 = math.sqrt(2.0)


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


def _phi(z):  # standard normal CDF, vectorized numpy
    from scipy.special import ndtr
    return ndtr(z)


def _acc_at(margins, wnorm, sigma):
    if sigma <= 0:
        return float((margins > 0).mean())
    return float(_phi(margins / (sigma * wnorm + 1e-12)).mean())


def _analytic_sigma(margins, wnorm, thr=THR):
    """Continuous sigma* where acc(sigma)=thr. Always finite if clean-acc>thr (acc->0.5<thr)."""
    clean = float((margins > 0).mean())
    if clean < thr:
        return 0.0
    lo, hi = 1e-4, 1.0
    it = 0
    while _acc_at(margins, wnorm, hi) > thr and hi < 1e8 and it < 60:
        hi *= 2; it += 1
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _acc_at(margins, wnorm, mid) > thr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _collect(model, dataset):
    tr = LayerWiseMoralProbe._collect_all_activations(model, dataset.train)
    te = LayerWiseMoralProbe._collect_all_activations(model, dataset.test)
    return tr, te


def _condition_margins(train_acts, test_acts, n_layers):
    """Return per-layer: test margins (signed), ||w||, rms(test), clean acc, analytic sigma*_raw + snr."""
    out = {"per_layer_margins": [], "per_layer_wnorm": [], "per_layer_rms": [],
           "per_layer_clean_acc": [], "per_layer_sigma_raw": [], "per_layer_sigma_snr": []}
    for L in range(n_layers):
        trX, trY = train_acts[L]; teX, teY = test_acts[L]
        p = _train_probe(trX, trY)
        with torch.no_grad():
            logits = p(teX).squeeze(-1)
        s = (teY * 2 - 1)
        margins = (s * logits).numpy().astype(np.float64)
        wnorm = float(p.weight.detach().norm())
        rms = float(teX.pow(2).mean().sqrt())
        sr = _analytic_sigma(margins, wnorm)
        out["per_layer_margins"].append(margins.tolist())
        out["per_layer_wnorm"].append(wnorm)
        out["per_layer_rms"].append(round(rms, 5))
        out["per_layer_clean_acc"].append(round(float((margins > 0).mean()), 4))
        out["per_layer_sigma_raw"].append(round(sr, 5))
        out["per_layer_sigma_snr"].append(round(sr / max(rms, 1e-8), 5))
    out["mean_sigma_raw"] = round(float(np.mean(out["per_layer_sigma_raw"])), 5)
    out["mean_sigma_snr"] = round(float(np.mean(out["per_layer_sigma_snr"])), 5)
    out["mean_rms"] = round(float(np.mean(out["per_layer_rms"])), 5)
    return out


def _paired_delta_ci(res, a="narrative", b="declarative", conv="raw", n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over TEST examples of sigma*(a)-sigma*(b), mean over layers, given convention."""
    rng = np.random.default_rng(seed)
    Ma = [np.asarray(m) for m in res[a]["per_layer_margins"]]
    Mb = [np.asarray(m) for m in res[b]["per_layer_margins"]]
    wa = res[a]["per_layer_wnorm"]; wb = res[b]["per_layer_wnorm"]
    ra = res[a]["per_layer_rms"]; rb = res[b]["per_layer_rms"]
    nL = len(Ma); n = len(Ma[0])
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # shared test indices (paired)
        sa = []; sb = []
        for L in range(nL):
            va = _analytic_sigma(Ma[L][idx], wa[L]); vb = _analytic_sigma(Mb[L][idx], wb[L])
            if conv == "snr":
                va /= max(ra[L], 1e-8); vb /= max(rb[L], 1e-8)
            sa.append(va); sb.append(vb)
        deltas.append(float(np.mean(sa) - np.mean(sb)))
    deltas = np.array(deltas)
    return {"mean": round(float(deltas.mean()), 5),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 5),
                     round(float(np.percentile(deltas, 97.5)), 5)],
            "frac_le0": round(float((deltas <= 0).mean()), 5)}


def _load():
    return WhiteBoxModel(REPO, revision=REV, access_tier=AccessTier.CHECKPOINTS,
                         checkpoint_step=STEP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--conditions", default="declarative,narrative,general")
    ap.add_argument("--max-tokens", type=int, default=500_000)
    ap.add_argument("--tag", default="analytic1000")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f"confound_analytic_{args.tag}.json"

    dataset = build_probing_dataset(target_per_foundation=40)
    print(f"dataset: {len(dataset.train)} train, {len(dataset.test)} test", flush=True)

    results = {"config": {"steps": args.steps, "estimator": "analytic_phi", "thr": THR,
                          "n_boot": args.boot, "repo": REPO, "rev": REV}}

    def _save():
        # margins are large; keep a slim copy without per-example margins for the summary json
        slim = {"config": results["config"]}
        for k, v in results.items():
            if k == "config":
                continue
            slim[k] = {kk: vv for kk, vv in v.items() if kk != "per_layer_margins"} if isinstance(v, dict) else v
        with open(outp, "w") as f:
            json.dump(slim, f, indent=2)

    t0 = time.time()
    model = _load()
    n_layers = model.info.n_layers
    tr, te = _collect(model, dataset)
    results["base"] = _condition_margins(tr, te, n_layers)
    _save()
    print(f"BASE sigma_raw={results['base']['mean_sigma_raw']} "
          f"sigma_snr={results['base']['mean_sigma_snr']} ({time.time()-t0:.0f}s)", flush=True)
    del model; import gc; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    from deepsteer.datasets.corpora import (load_declarative_corpus,
        load_narrative_corpus, load_general_corpus)
    loaders = {"declarative": load_declarative_corpus, "narrative": load_narrative_corpus,
               "general": load_general_corpus}

    for cond in args.conditions.split(","):
        cond = cond.strip()
        if not cond:
            continue
        print(f"\n=== {cond} ===", flush=True)
        t0 = time.time()
        corpus = loaders[cond](max_tokens=args.max_tokens)
        print(f"  corpus: {len(corpus)} chunks", flush=True)
        model = _load()
        trainer = LoRATrainer(model, corpus, dataset, lora_rank=16, lora_alpha=32,
                              max_steps=args.steps, eval_every=args.steps + 10, run_fragility=False)
        res = trainer.train(experiment_id=f"analytic_{cond}", corpus_name=cond)
        losses = [s.loss for s in res.training_steps]
        tr, te = _collect(model, dataset)
        m = _condition_margins(tr, te, n_layers)
        m["loss_last"] = round(float(np.mean(losses[-20:])), 3) if losses else None
        results[cond] = m
        _save()
        print(f"  {cond}: loss_end={m['loss_last']} mean_rms={m['mean_rms']} "
              f"sigma_raw={m['mean_sigma_raw']} sigma_snr={m['mean_sigma_snr']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del model, trainer; gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # save full margins npz for zero-GPU CI re-computation
    npz = {}
    for c in [k for k in results if isinstance(results[k], dict) and "per_layer_margins" in results[k]]:
        for L, mm in enumerate(results[c]["per_layer_margins"]):
            npz[f"{c}_L{L}"] = np.asarray(mm)
    np.savez(OUT / f"margins_{args.tag}.npz", **npz)

    # paired Delta-CIs under BOTH conventions (pre-registered)
    conds = [k for k in results if isinstance(results[k], dict) and "per_layer_margins" in results[k]]
    if "narrative" in conds and "declarative" in conds:
        results["delta_ci"] = {
            "raw": _paired_delta_ci(results, conv="raw", n_boot=args.boot),
            "snr": _paired_delta_ci(results, conv="snr", n_boot=args.boot),
            "note": "Delta = sigma*(narrative) - sigma*(declarative); >0 => declarative more fragile (Branch B)"}
    # ordering summary
    order = {}
    for conv in ["raw", "snr"]:
        key = "mean_sigma_raw" if conv == "raw" else "mean_sigma_snr"
        vals = {c: results[c][key] for c in conds}
        order[conv] = {"values": vals, "most_fragile": min(vals, key=vals.get),
                       "declarative_most_fragile": min(vals, key=vals.get) == "declarative"}
    results["ordering"] = order
    _save()
    print("\n=== VERDICT ===", flush=True)
    print("ordering:", json.dumps(order), flush=True)
    print("delta_ci:", json.dumps(results.get("delta_ci", {})), flush=True)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
