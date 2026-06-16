#!/usr/bin/env python3
"""Sprint 1.1/1.2: probe transfer test (raw + chat-formatted).

Loads base-model-trained foundation directions and evaluates how well they
separate moral/neutral pairs on a TARGET model, without retraining. Also
trains fresh directions on the target and reports the cosine between
base-trained and target-fresh directions.

Two input formats:
  * ``--input-format raw``  : probing texts fed verbatim.
  * ``--input-format chat`` : texts wrapped in the model chat template; only
    content tokens are pooled (template tokens excluded).

Per foundation, per layer, reports:
  * ``transfer_auc``  : threshold-free separability of base direction on the
    target test pairs (headline metric; no fitting).
  * ``transfer_acc``  : accuracy at the class-mean midpoint (mild centering).
  * ``cos_base_vs_fresh_probe`` / ``cos_base_vs_fresh_meandiff`` : direction
    agreement between base and target.
  * ``fresh_probe_acc`` : held-out accuracy of the target's own fresh probe.

Usage:
    python papers/5_moral_alignment/scripts/probe_transfer.py \
        --model allenai/Olmo-3-7B-Instruct \
        --probe-dir papers/5_moral_alignment/outputs/olmo3_base \
        --input-format raw \
        --output-dir papers/5_moral_alignment/outputs/olmo3_instruct_raw \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_utils as du  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402

logger = logging.getLogger(__name__)


def group_by_foundation(pairs) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for p in pairs:
        out[p.foundation.value].append((p.moral, p.neutral))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe transfer test (raw + chat).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--probe-dir", required=True,
                    help="Dir with base exp1_probe_directions.npz.")
    ap.add_argument("--moral-npz", default=None,
                    help="Override path to base directions npz.")
    ap.add_argument("--input-format", choices=["raw", "chat"], default="raw")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset-target", type=int, default=40)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.pipeline import build_probing_dataset

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    moral_npz = args.moral_npz or str(Path(args.probe_dir) / "exp1_probe_directions.npz")
    base_dirs = du.load_directions(moral_npz)  # {foundation: {layer: vec}}
    foundations = [f for f in FOUNDATION_ORDER if f in base_dirs]
    print(f"Loaded base directions for {len(foundations)} foundations from {moral_npz}")

    dataset = build_probing_dataset(
        target_per_foundation=args.dataset_target, dataset_version="v2"
    )
    train_by_f = group_by_foundation(dataset.train)
    test_by_f = group_by_foundation(dataset.test)

    t0 = time.time()
    model = WhiteBoxModel(
        args.model, device=args.device, access_tier=AccessTier.WEIGHTS,
        revision=args.revision,
    )
    n_layers = model.info.n_layers
    print(f"Loaded {args.model} ({n_layers} layers) in {time.time()-t0:.1f}s; "
          f"input_format={args.input_format}")

    results: dict[str, dict] = {}
    fresh_probe_npz: dict[str, dict[int, np.ndarray]] = {}
    fresh_md_npz: dict[str, dict[int, np.ndarray]] = {}

    for f in foundations:
        test_pairs = test_by_f.get(f, [])
        train_pairs = train_by_f.get(f, [])
        if not test_pairs or not train_pairs:
            continue
        print(f"  {FOUNDATION_SHORT[f]}: transfer on {len(test_pairs)} test pairs, "
              f"fresh on {len(train_pairs)} train pairs")

        # Target activations of TEST pairs -> transfer eval against base dirs.
        test_acts = du.collect_pair_activations(
            model, test_pairs, input_format=args.input_format
        )
        # Fresh target directions from TRAIN pairs.
        fresh, fresh_acc = du.extract_pair_directions(
            model, train_pairs, input_format=args.input_format
        )
        fresh_probe_npz[f] = fresh["probe"]
        fresh_md_npz[f] = fresh["mean_diff"]

        per_layer: dict[str, dict] = {}
        for L in range(n_layers):
            if L not in base_dirs[f] or L not in test_acts:
                continue
            X, y = test_acts[L]
            tm = du.transfer_metrics(X, y, base_dirs[f][L])
            per_layer[str(L)] = {
                "transfer_auc": round(tm["auc"], 4),
                "transfer_auc_abs": round(tm["auc_abs"], 4),
                "transfer_acc": round(tm["acc_midpoint"], 4),
                "fresh_probe_acc": round(fresh_acc[L], 4),
                "cos_base_vs_fresh_probe": round(du.cosine(base_dirs[f][L], fresh["probe"][L]), 4),
                "cos_base_vs_fresh_meandiff": round(du.cosine(base_dirs[f][L], fresh["mean_diff"][L]), 4),
            }
        results[f] = per_layer

    model.release()

    # ---- summary: best transfer AUC layer per foundation ----
    summary = {}
    for f, pl in results.items():
        if not pl:
            continue
        best_L = max(pl, key=lambda L: pl[L]["transfer_auc_abs"])
        summary[f] = {
            "best_layer": int(best_L),
            "transfer_auc_abs": pl[best_L]["transfer_auc_abs"],
            "transfer_acc": pl[best_L]["transfer_acc"],
            "cos_base_vs_fresh_probe": pl[best_L]["cos_base_vs_fresh_probe"],
        }

    payload = {
        "analysis": "probe_transfer",
        "model": args.model,
        "revision": args.revision,
        "input_format": args.input_format,
        "base_directions": moral_npz,
        "n_layers": n_layers,
        "full_attention_layers": du.OLMO3_FULL_ATTENTION_LAYERS,
        "foundations": foundations,
        "per_foundation": results,
        "summary": summary,
    }
    with open(out / "probe_transfer.json", "w") as f:
        json.dump(payload, f, indent=2)
    du.save_directions(out / "fresh_probe_directions.npz", fresh_probe_npz)
    du.save_directions(out / "fresh_meandiff_directions.npz", fresh_md_npz)

    print(f"\nWrote {out/'probe_transfer.json'}")
    print(f"  {'foundation':12s} bestL  AUC   acc   cos(base,fresh)")
    for f in foundations:
        if f in summary:
            s = summary[f]
            print(f"  {FOUNDATION_SHORT[f]:12s} {s['best_layer']:3d}  "
                  f"{s['transfer_auc_abs']:.3f} {s['transfer_acc']:.3f}  "
                  f"{s['cos_base_vs_fresh_probe']:+.3f}")


if __name__ == "__main__":
    main()
