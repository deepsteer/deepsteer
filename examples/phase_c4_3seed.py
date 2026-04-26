#!/usr/bin/env python3
"""Phase C4 3-seed compositional fragility replication.

Settles the post-step-7K compositional fragility decline question
flagged in `outputs/phase_c4_compositional/RESULTS.md` and locked as a
gating decision for `paper/PAPER_PLAN.md` §4.3.

The original C4 trajectory (split seed 42) reported compositional mean
critical noise rising 0.10 → 5.69 (peak step 5K) and then drifting
back down to 2.66 by step 36K — *opposite* to the standard moral
probe's behavior in Phase C1, where mean critical noise rises through
step 1K and stays at the maximum across mid- and late-layers
throughout training. With only one seed, we cannot tell if the
post-step-7K decline is real (compositional encoding becomes more
brittle as training continues on text that does not specifically
reinforce compositional moral integration) or seed-side noise (a
single train / test split at probe accuracy ≈ 0.75 is not enough
resolution to claim the 5.7 → 2.7 drop is signal).

This script runs the same `CompositionalMoralProbe` +
`MoralFragilityTest` pipeline at all 37 OLMo-2 1B early-training
checkpoints with three additional split seeds (43, 44, 45). For
efficiency it loads each model checkpoint once and runs all three
seeds against the loaded weights. The original seed-42 results are
read from `outputs/phase_c4_compositional/step_NNNNNNN/c4_step.json`
during aggregation; this script only generates the 3 new seeds.

Decision rule (per `paper/PAPER_PLAN.md` §4.3):

* If 4-seed mean critical noise drops by ≥ 1.0 between step 7K and
  step 30K, **and** seed-to-seed std at both endpoints is smaller
  than the gap → the post-step-7K decline is real and earns a paper
  subsection.
* Otherwise → the apparent decline is within seed noise and gets a
  one-line mention.

Per-seed runtime estimate: ~20 min for the full 37 × (probe +
fragility), matching the original. Batched across seeds with shared
model loading: ~50 min total for all three new seeds.

Outputs (`outputs/phase_c4_compositional/3seed/`):

* `step_NNNNNNN/seed_NN.json` — per-seed per-step probe + fragility
* `aggregate_per_checkpoint.json` — 4-seed combined per-step JSON
* `decision.json` — decision rule application + verdict
* `4seed_fragility_evolution.png` — mean ± shaded std plot (vs. C1
  standard moral as baseline)

Usage:
    # Run all 3 new seeds across all 37 checkpoints
    python examples/phase_c4_3seed.py

    # Aggregate-and-plot only (assumes per-seed JSONs already exist)
    python examples/phase_c4_3seed.py --aggregate-only

    # Resume from a specific step
    python examples/phase_c4_3seed.py --resume-from step18000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"
SEEDS = [43, 44, 45]
ORIGINAL_DIR = Path("outputs/phase_c4_compositional")
OUTPUT_DIR = Path("outputs/phase_c4_compositional/3seed")


def _clear_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _parse_step(rev: str) -> int | None:
    m = re.search(r"step(\d+)", rev)
    return int(m.group(1)) if m else None


def _get_revisions() -> list[tuple[int, str]]:
    from deepsteer.benchmarks.representational.trajectory import list_available_revisions

    revs = list_available_revisions(REPO_ID)
    out: list[tuple[int, str]] = []
    for r in revs:
        s = _parse_step(r)
        if s is not None:
            out.append((s, r))
    out.sort(key=lambda x: x[0])
    return out


def run_one_seed(model, seed: int) -> dict:
    """Run probe + fragility for one seed on a loaded model."""
    import torch

    from deepsteer.benchmarks.representational.compositional_moral_probe import (
        CompositionalMoralProbe,
        _build_compositional_probing_dataset,
    )
    from deepsteer.benchmarks.representational.fragility import MoralFragilityTest

    dataset = _build_compositional_probing_dataset(seed=seed)

    torch.manual_seed(seed)
    probe = CompositionalMoralProbe(dataset=dataset)
    t0 = time.time()
    probe_result = probe.run(model)
    probe_elapsed = time.time() - t0
    mean_acc = float(np.mean([s.accuracy for s in probe_result.layer_scores]))

    torch.manual_seed(seed)
    frag_test = MoralFragilityTest(dataset=dataset)
    t0 = time.time()
    frag_result = frag_test.run(model)
    frag_elapsed = time.time() - t0

    return {
        "seed": seed,
        "probe": {
            "peak_accuracy": float(probe_result.peak_accuracy),
            "peak_layer": int(probe_result.peak_layer),
            "onset_layer": probe_result.onset_layer,
            "mean_accuracy": mean_acc,
            "encoding_depth": float(probe_result.moral_encoding_depth),
            "encoding_breadth": float(probe_result.moral_encoding_breadth),
            "elapsed_s": round(probe_elapsed, 1),
            "layer_scores": [
                {
                    "layer": s.layer,
                    "accuracy": float(s.accuracy),
                    "loss": float(s.loss),
                }
                for s in probe_result.layer_scores
            ],
        },
        "fragility": {
            "mean_critical_noise": frag_result.mean_critical_noise,
            "most_fragile_layer": frag_result.most_fragile_layer,
            "most_robust_layer": frag_result.most_robust_layer,
            "noise_levels": frag_result.noise_levels,
            "elapsed_s": round(frag_elapsed, 1),
            "layer_scores": [
                {
                    "layer": s.layer,
                    "baseline_accuracy": float(s.baseline_accuracy),
                    "critical_noise": s.critical_noise,
                    "accuracy_by_noise": {
                        str(k): float(v)
                        for k, v in s.accuracy_by_noise.items()
                    },
                }
                for s in frag_result.layer_scores
            ],
        },
    }


def _load_seed42(step: int) -> dict | None:
    """Load the original seed-42 result for a checkpoint, if it exists."""
    p = ORIGINAL_DIR / f"step_{step:07d}" / "c4_step.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return {"seed": 42, "probe": d["probe"], "fragility": d.get("fragility", {})}


def aggregate_and_decide(output_dir: Path) -> dict:
    """Aggregate seeds 42 + 43 + 44 + 45 and apply the decision rule."""
    revs = _get_revisions()
    sorted_steps = [s for s, _ in revs]

    per_step: dict[int, list[dict]] = {}
    for step in sorted_steps:
        per_step[step] = []

        # Original seed 42
        s42 = _load_seed42(step)
        if s42 is not None:
            per_step[step].append(s42)

        # New seeds
        step_dir = output_dir / f"step_{step:07d}"
        for seed in SEEDS:
            p = step_dir / f"seed_{seed:02d}.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                per_step[step].append(
                    {"seed": d["seed"], "probe": d["probe"], "fragility": d["fragility"]},
                )

    # Compute per-step mean ± std for the headline metrics
    aggregate: dict[str, dict] = {}
    for step in sorted_steps:
        runs = per_step[step]
        if not runs:
            continue
        mean_accs = [r["probe"]["mean_accuracy"] for r in runs]
        peak_accs = [r["probe"]["peak_accuracy"] for r in runs]
        mcns = [
            r["fragility"]["mean_critical_noise"]
            for r in runs
            if r["fragility"].get("mean_critical_noise") is not None
        ]
        aggregate[str(step)] = {
            "step": step,
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "mean_accuracy": {
                "mean": float(np.mean(mean_accs)),
                "std": float(np.std(mean_accs, ddof=1)) if len(mean_accs) > 1 else 0.0,
                "values": mean_accs,
            },
            "peak_accuracy": {
                "mean": float(np.mean(peak_accs)),
                "std": float(np.std(peak_accs, ddof=1)) if len(peak_accs) > 1 else 0.0,
                "values": peak_accs,
            },
            "mean_critical_noise": {
                "mean": float(np.mean(mcns)) if mcns else None,
                "std": float(np.std(mcns, ddof=1)) if len(mcns) > 1 else 0.0,
                "values": mcns,
                "n_with_data": len(mcns),
            },
        }

    out = {
        "model": REPO_ID,
        "experiment": "C4_3seed",
        "checkpoints": sorted_steps,
        "per_checkpoint": aggregate,
        "seeds_per_checkpoint": {
            str(s): per_step[s] for s in sorted_steps
        },
    }
    with open(output_dir / "aggregate_per_checkpoint.json", "w") as f:
        # Drop the large per-seed dump from this top-level file
        json.dump(
            {k: v for k, v in out.items() if k != "seeds_per_checkpoint"},
            f, indent=2,
        )

    # Apply decision rule
    steps_present = [int(k) for k in aggregate]
    step_7k = 7000 if 7000 in steps_present else None
    step_30k = 30000 if 30000 in steps_present else None

    if step_7k is None or step_30k is None:
        verdict = {
            "verdict": "incomplete",
            "reason": f"Need step 7K and 30K data; have 7K={step_7k}, 30K={step_30k}",
        }
    else:
        m7 = aggregate[str(step_7k)]["mean_critical_noise"]
        m30 = aggregate[str(step_30k)]["mean_critical_noise"]
        gap = (m7["mean"] or 0.0) - (m30["mean"] or 0.0)
        std_max = max(m7["std"], m30["std"])
        decline_replicates = gap >= 1.0
        std_smaller_than_gap = std_max < gap
        if decline_replicates and std_smaller_than_gap:
            verdict_str = "decline_real"
            interpretation = (
                "Post-step-7K decline replicates across all seeds. "
                "Compositional fragility evolution earns its own paper subsection."
            )
        elif decline_replicates and not std_smaller_than_gap:
            verdict_str = "decline_within_noise"
            interpretation = (
                "Mean drops by ≥ 1.0 but seed-to-seed std exceeds the gap. "
                "Apparent decline is within seed noise; one-line mention."
            )
        else:
            verdict_str = "no_decline"
            interpretation = (
                "Mean does not drop by ≥ 1.0 between step 7K and step 30K. "
                "No post-step-7K decline; one-line mention."
            )
        verdict = {
            "verdict": verdict_str,
            "interpretation": interpretation,
            "step_7k_mean_critical_noise": m7,
            "step_30k_mean_critical_noise": m30,
            "gap": float(gap),
            "max_std": float(std_max),
            "decline_replicates": bool(decline_replicates),
            "std_smaller_than_gap": bool(std_smaller_than_gap),
            "rule": "decline_real if (mean[7K] - mean[30K] >= 1.0) AND (max(std[7K], std[30K]) < gap)",
        }

    with open(output_dir / "decision.json", "w") as f:
        json.dump(verdict, f, indent=2)

    return verdict


def generate_4seed_plot(output_dir: Path) -> None:
    """Plot 4-seed mean critical noise across all 37 checkpoints with std bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(output_dir / "aggregate_per_checkpoint.json") as f:
        agg = json.load(f)

    sorted_steps = sorted(int(k) for k in agg["per_checkpoint"])
    means = []
    stds = []
    for s in sorted_steps:
        d = agg["per_checkpoint"][str(s)]["mean_critical_noise"]
        means.append(d["mean"] if d["mean"] is not None else 0.0)
        stds.append(d["std"])

    means = np.array(means)
    stds = np.array(stds)

    # Pull C1 standard moral mean critical noise for comparison.
    c1_steps, c1_mcn = [], []
    for s in sorted_steps:
        c1_path = Path("outputs/phase_c1") / f"step_{s:07d}" / "moral_fragility_test_allenai_OLMo-2-0425-1B-early-training.json"
        if c1_path.exists():
            with open(c1_path) as f:
                d = json.load(f)
            c1_steps.append(s)
            # Compute mean critical noise from per-layer scores
            cns = [ls["critical_noise"] for ls in d["layer_scores"] if ls.get("critical_noise") is not None]
            c1_mcn.append(float(np.mean(cns)) if cns else 0.0)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Compositional 4-seed: mean ± std band
    ax.fill_between(sorted_steps, means - stds, means + stds, color="#9C27B0", alpha=0.20,
                    label="Compositional moral (4 seeds: 42/43/44/45) ± std")
    ax.plot(sorted_steps, means, "D-", color="#9C27B0", linewidth=2, markersize=4,
            label="Compositional moral (4-seed mean)")

    # Standard moral baseline for comparison
    if c1_steps:
        ax.plot(c1_steps, c1_mcn, "o-", color="#F44336", linewidth=2, markersize=4,
                label="Standard moral (C1, 1 seed)", alpha=0.85)

    ax.axhline(y=10.0, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.5,
               label="max sweep σ = 10")

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Mean critical noise (σ)", fontsize=11)
    ax.set_title(
        f"Phase C4 4-seed fragility evolution — {REPO_ID}\n"
        f"Compositional (mean ± std across split seeds 42/43/44/45) vs. standard moral (C1)",
        fontsize=11,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.05, 15)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    out = output_dir / "4seed_fragility_evolution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("4-seed fragility plot: %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default=None,
                        help="Device override (cuda, mps, cpu).")
    parser.add_argument("--resume-from", default=None,
                        help="Resume trajectory from this step (e.g. step18000).")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip the trajectory; just aggregate + decide + plot.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        from deepsteer.core.model_interface import WhiteBoxModel
        from deepsteer.core.types import AccessTier

        revs = _get_revisions()
        sorted_steps = [s for s, _ in revs]
        rev_map = dict(revs)

        if args.resume_from:
            rs = _parse_step(args.resume_from)
            if rs is not None:
                sorted_steps = [s for s in sorted_steps if s >= rs]

        print(f"Found {len(_get_revisions())} step revisions; running {len(sorted_steps)} from "
              f"step {sorted_steps[0]} to step {sorted_steps[-1]}")

        total_t0 = time.time()
        for i, step in enumerate(sorted_steps):
            rev = rev_map[step]
            step_dir = OUTPUT_DIR / f"step_{step:07d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            seeds_to_run = [s for s in SEEDS if not (step_dir / f"seed_{s:02d}.json").exists()]
            if not seeds_to_run:
                print(f"[{i+1}/{len(sorted_steps)}] step {step}: all 3 seeds done, skipping")
                continue

            print(f"\n=== [{i+1}/{len(sorted_steps)}] step {step} ({rev}) — seeds {seeds_to_run} ===")

            t0 = time.time()
            model = WhiteBoxModel(
                REPO_ID,
                revision=rev,
                device=args.device,
                access_tier=AccessTier.CHECKPOINTS,
                checkpoint_step=step,
            )
            print(f"  loaded in {time.time()-t0:.1f}s")

            for seed in seeds_to_run:
                print(f"  seed {seed}:")
                result = run_one_seed(model, seed)
                with open(step_dir / f"seed_{seed:02d}.json", "w") as f:
                    json.dump({"step": step, "revision": rev, **result}, f, indent=2)
                mean_acc = result["probe"]["mean_accuracy"]
                mcn = result["fragility"]["mean_critical_noise"]
                print(f"    mean_acc={mean_acc:.3f}, mean_critical_noise="
                      f"{mcn if mcn is not None else 'None':>5}")

            del model
            _clear_memory()

            elapsed = time.time() - total_t0
            remaining = len(sorted_steps) - (i + 1)
            avg = elapsed / (i + 1)
            print(f"  elapsed: {elapsed/60:.1f}min, ETA: {avg*remaining/60:.1f}min")

        print(f"\n3-seed trajectory complete in {(time.time()-total_t0)/60:.1f} min")

    # Aggregate + decide + plot.
    print(f"\n{'='*60}\nAggregating 4-seed data and applying decision rule\n{'='*60}")
    verdict = aggregate_and_decide(OUTPUT_DIR)
    print(f"VERDICT: {verdict.get('verdict')}")
    print(f"  Reason: {verdict.get('interpretation', verdict.get('reason'))}")
    if "step_7k_mean_critical_noise" in verdict:
        m7 = verdict["step_7k_mean_critical_noise"]
        m30 = verdict["step_30k_mean_critical_noise"]
        print(f"  step 7K:  mean={m7['mean']:.3f}, std={m7['std']:.3f}, n={m7['n_with_data']}")
        print(f"  step 30K: mean={m30['mean']:.3f}, std={m30['std']:.3f}, n={m30['n_with_data']}")
        print(f"  gap (7K → 30K): {verdict['gap']:.3f}; max std: {verdict['max_std']:.3f}")
        print(f"  decline replicates (gap ≥ 1.0): {verdict['decline_replicates']}")
        print(f"  std smaller than gap: {verdict['std_smaller_than_gap']}")

    generate_4seed_plot(OUTPUT_DIR)
    print(f"\nDone. See {OUTPUT_DIR}/decision.json and {OUTPUT_DIR}/4seed_fragility_evolution.png")


if __name__ == "__main__":
    main()
