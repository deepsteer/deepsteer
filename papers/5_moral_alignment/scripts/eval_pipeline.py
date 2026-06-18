#!/usr/bin/env python3
"""Sprint 7: post-ART evaluation battery + Heretic ablation comparison.

Runs the full Phase 2 measurement battery on the ART-SFT and control-SFT models,
applies Arditi/Heretic uniform refusal ablation to both, re-runs the battery on
the ablated models, and assembles the headline comparison across the four cells:

    control-SFT | control-SFT+Heretic | ART-SFT | ART-SFT+Heretic

The prediction: ART-SFT+Heretic degrades on moral probe accuracy / moral
judgment / moral-text dependency (ablating compliance can no longer leave
comprehension intact), while control-SFT+Heretic does not.

The battery reuses the existing Phase 2 / Sprint 5 CLIs as subprocesses
(fail-soft: one failed measurement does not abort the rest):
  * pipeline_study.py    -> probing (fresh acc), framework geometry (eff-dim), persona
  * coupling_measurement.py -> comprehension / compliance / coupling phi
  * behavioral_baseline.py  -> moral-foundations judgment + persona shift
  * moral_dependency.py     -> moral-ablation dependency score
  * heretic_ablation.py     -> ablated model + refusal-morality projection fraction (7.4)

DURABLE ARTIFACTS: the inputs can be the small LoRA *adapters* (~100-200 MB) plus
``--base-model``, not the 14 GB merged models. Given adapters, this script
reconstructs each merged model on the pod's (ephemeral) disk before the battery,
so nothing 14 GB ever has to survive a pod shutdown — see the runpod README.

Usage (adapters — recommended, durable):
    python papers/5_moral_alignment/scripts/eval_pipeline.py \
        --art-adapter   papers/5_moral_alignment/outputs/art_sft/adapter \
        --control-adapter papers/5_moral_alignment/outputs/control_sft/adapter \
        --base-model allenai/Olmo-3-1025-7B \
        --base-dir papers/5_moral_alignment/outputs/olmo3_base \
        --device cuda

Usage (already-merged dirs — same session):
    python … --art-model …/art_sft/merged_model --control-model …/control_sft/merged_model …
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_P5 = "papers/5_moral_alignment/scripts"
_P6 = "papers/5_moral_alignment/scripts"
_DEFAULT_BASE_DIR = "papers/5_moral_alignment/outputs/olmo3_base"
_DEFAULT_REFUSAL_PROMPTS = "papers/5_moral_alignment/refusal_prompts.json"

STEPS: list[dict] = []  # run log


def run_cmd(name: str, args: list[str], *, dry_run: bool) -> bool:
    """Run a subprocess step (fail-soft). Records to STEPS; returns ok."""
    printable = " ".join(args)
    logger.info("STEP %s\n    %s", name, printable)
    if dry_run:
        STEPS.append({"name": name, "cmd": printable, "ok": None, "dry_run": True})
        return True
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    ok = r.returncode == 0
    if not ok:
        logger.error("  FAILED %s (rc=%d)\n%s", name, r.returncode,
                     "\n".join(r.stderr.splitlines()[-12:]))
    STEPS.append({"name": name, "cmd": printable, "ok": ok,
                  "elapsed_s": round(time.time() - t0, 1)})
    return ok


# ---------------------------------------------------------------------------
# Materialize merged model from base + adapter (so only the small adapter needs
# to persist across pod shutdowns)
# ---------------------------------------------------------------------------


def materialize_merged(adapter: str, base_model: str, dest: Path, *, dry_run: bool) -> str:
    """Load base + LoRA adapter, merge, and save a standalone model to *dest*."""
    if dry_run:
        logger.info("  [dry-run] would merge %s + adapter %s -> %s", base_model, adapter, dest)
        return str(dest)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Merging base %s + adapter %s -> %s", base_model, adapter, dest)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(dest)
    # Tokenizer (with the chat template art_sft saved) lives in the adapter dir.
    AutoTokenizer.from_pretrained(adapter).save_pretrained(dest)
    del base, merged
    return str(dest)


# ---------------------------------------------------------------------------
# Battery + ablation
# ---------------------------------------------------------------------------


def battery(state: str, model_path: str, args, *, dry_run: bool) -> None:
    """Run the full measurement battery for one model state into <out>/<state>/."""
    out = Path(args.output_dir)
    state_dir = out / state
    state_dir.mkdir(parents=True, exist_ok=True)

    # pipeline_study needs a 1-entry grid; probing/geometry use raw (Phase 2 choice).
    grid_dir = out / "_grids"; grid_dir.mkdir(parents=True, exist_ok=True)
    grid = grid_dir / f"{state}.json"
    if not dry_run:
        grid.write_text(json.dumps([{"label": state, "repo": model_path, "revision": None}]))

    run_cmd(f"{state}:probing+geometry", [
        f"{_P5}/pipeline_study.py", "--grid", str(grid), "--base-dir", args.base_dir,
        "--input-format", "raw", "--output-dir", args.output_dir, "--device", args.device,
    ], dry_run=dry_run)
    run_cmd(f"{state}:coupling", [
        f"{_P5}/coupling_measurement.py", "--model", model_path,
        "--probe-dir", args.base_dir, "--layer", str(args.layer),
        "--input-format", "chat", "--output-dir", str(state_dir), "--device", args.device,
    ], dry_run=dry_run)
    run_cmd(f"{state}:behavioral", [
        f"{_P5}/behavioral_baseline.py", "--model", model_path, "--benchmark", "both",
        "--input-format", "chat", "--output-dir", str(state_dir), "--device", args.device,
    ], dry_run=dry_run)
    run_cmd(f"{state}:dependency", [
        f"{_P6}/moral_dependency.py", "--model", model_path,
        "--directions-npz", args.base_directions, "--no-per-text",
        "--label", state, "--output-dir", str(state_dir), "--device", args.device,
    ], dry_run=dry_run)


def heretic(cond: str, model_path: str, args, *, dry_run: bool) -> str:
    """Apply Heretic ablation; return the ablated-model dir."""
    her_dir = Path(args.output_dir) / f"heretic_{cond}"
    cmd = [
        f"{_P5}/heretic_ablation.py", "--model", model_path,
        "--moral-npz", args.base_directions, "--refusal-layer", str(args.layer),
        "--input-format", "chat", "--output-dir", str(her_dir), "--device", args.device,
    ]
    if Path(args.refusal_prompts).exists():
        cmd += ["--prompts", args.refusal_prompts]
    run_cmd(f"{cond}:heretic_ablation", cmd, dry_run=dry_run)
    return str(her_dir / "ablated_model")


# ---------------------------------------------------------------------------
# Comparison assembly (robust to missing files)
# ---------------------------------------------------------------------------


def _load(path: Path):
    return json.load(open(path)) if Path(path).exists() else None


def _moral_probe_acc(state_dir: Path) -> float | None:
    mp = _load(state_dir / "moral_probing.json")
    if not mp:
        return None
    accs = [max((v.get("fresh_probe_acc", 0) for v in pl.values()), default=0)
            for pl in mp.get("per_foundation", {}).values() if pl]
    return round(float(sum(accs) / len(accs)), 4) if accs else None


def _eff_dim(state_dir: Path, layer: int):
    g = _load(state_dir / "geometry.json")
    pl = (g or {}).get("per_layer", {})
    at_layer = pl.get(str(layer), {}).get("effective_dimensionality") if pl else None
    vals = [v.get("effective_dimensionality") for v in pl.values()
            if v.get("effective_dimensionality") is not None]
    return at_layer, (round(sum(vals) / len(vals), 2) if vals else None)


def _state_metrics(state_dir: Path, layer: int) -> dict:
    bb = _load(state_dir / "behavioral_baseline.json") or {}
    mf = bb.get("results", {}).get("moral_foundations", {})
    cp = _load(state_dir / "coupling.json") or {}
    dep = _load(state_dir / "moral_dependency.json") or {}
    eff_at, eff_mean = _eff_dim(state_dir, layer)
    return {
        "moral_probe_acc": _moral_probe_acc(state_dir),
        "eff_dim_at_layer": eff_at,
        "eff_dim_mean": eff_mean,
        "moral_judgment_acc": mf.get("overall_accuracy"),
        "depth_gradient": mf.get("depth_gradient"),
        "compliance_rate": cp.get("compliance_rate"),
        "comprehension_rate": cp.get("comprehension_rate"),
        "coupling_phi": cp.get("coupling_phi"),
        "moral_dependency_score": dep.get("metrics", {}).get("moral_dependency_score"),
    }


def _delta(a, b):
    return round(a - b, 4) if (a is not None and b is not None) else None


def assemble(args) -> dict:
    out = Path(args.output_dir)
    states = {s: _state_metrics(out / s, args.layer)
              for s in ("control", "control_ablated", "art", "art_ablated")}

    refusal = {}
    for cond in ("control", "art"):
        g = _load(out / f"heretic_{cond}" / "refusal_morality_geometry.json")
        if g:
            refusal[cond] = {
                "moral_subspace_projection_fraction": g.get("moral_subspace_projection_fraction"),
                "mean_abs_cosine": g.get("mean_abs_cosine"),
            }

    c, ca, a, aa = (states["control"], states["control_ablated"],
                    states["art"], states["art_ablated"])
    key = {
        # ART should raise dependency + the refusal->moral projection fraction.
        "dependency_control_vs_art": [c["moral_dependency_score"], a["moral_dependency_score"]],
        "proj_fraction_control_vs_art": [
            refusal.get("control", {}).get("moral_subspace_projection_fraction"),
            refusal.get("art", {}).get("moral_subspace_projection_fraction"),
        ],
        # Heretic should damage ART more than control (the headline).
        "moral_judgment_drop_control": _delta(c["moral_judgment_acc"], ca["moral_judgment_acc"]),
        "moral_judgment_drop_art": _delta(a["moral_judgment_acc"], aa["moral_judgment_acc"]),
        "moral_probe_acc_drop_control": _delta(c["moral_probe_acc"], ca["moral_probe_acc"]),
        "moral_probe_acc_drop_art": _delta(a["moral_probe_acc"], aa["moral_probe_acc"]),
        "dependency_drop_control": _delta(c["moral_dependency_score"], ca["moral_dependency_score"]),
        "dependency_drop_art": _delta(a["moral_dependency_score"], aa["moral_dependency_score"]),
    }
    return {"analysis": "art_eval_comparison", "layer": args.layer,
            "states": states, "refusal_morality": refusal, "key_comparison": key}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Sprint 7 post-ART evaluation + Heretic comparison.")
    ap.add_argument("--art-model", default=None, help="Merged ART-SFT model dir.")
    ap.add_argument("--control-model", default=None, help="Merged control-SFT model dir.")
    ap.add_argument("--art-adapter", default=None, help="ART-SFT LoRA adapter dir (durable).")
    ap.add_argument("--control-adapter", default=None, help="control-SFT LoRA adapter dir.")
    ap.add_argument("--base-model", default="allenai/Olmo-3-1025-7B",
                    help="Base model to merge adapters onto (if adapters given).")
    ap.add_argument("--base-dir", default=_DEFAULT_BASE_DIR,
                    help="olmo3_base dir (holds exp1_probe_directions.npz).")
    ap.add_argument("--base-directions", default=None,
                    help="Override base directions npz; default <base-dir>/exp1_probe_directions.npz.")
    ap.add_argument("--refusal-prompts", default=_DEFAULT_REFUSAL_PROMPTS)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--output-dir", default="papers/5_moral_alignment/outputs/eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-heretic", action="store_true")
    ap.add_argument("--keep-merged", action="store_true",
                    help="Keep merged models materialized from adapters (default: leave in _merged/).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the full command sequence without running anything.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.base_directions is None:
        args.base_directions = str(Path(args.base_dir) / "exp1_probe_directions.npz")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve each condition to a usable model dir (merge adapters if needed).
    def resolve(cond: str, model: str | None, adapter: str | None) -> str:
        if model:
            return model
        if adapter:
            return materialize_merged(adapter, args.base_model,
                                      out / "_merged" / cond, dry_run=args.dry_run)
        raise SystemExit(f"Provide --{cond}-model or --{cond}-adapter.")

    control_model = resolve("control", args.control_model, args.control_adapter)
    art_model = resolve("art", args.art_model, args.art_adapter)

    for cond, mpath in (("control", control_model), ("art", art_model)):
        battery(cond, mpath, args, dry_run=args.dry_run)
        if not args.skip_heretic:
            ablated = heretic(cond, mpath, args, dry_run=args.dry_run)
            battery(f"{cond}_ablated", ablated, args, dry_run=args.dry_run)

    comparison = assemble(args) if not args.dry_run else {"dry_run": True}
    with open(out / "comparison.json", "w") as fh:
        json.dump({**comparison, "steps": STEPS}, fh, indent=2)

    n_fail = sum(1 for s in STEPS if s.get("ok") is False)
    print(f"\nWrote {out/'comparison.json'} ({len(STEPS)} steps, {n_fail} failed)")
    if not args.dry_run:
        k = comparison["key_comparison"]
        print("  moral dependency  control -> art :", k["dependency_control_vs_art"])
        print("  refusal->moral proj control -> art:", k["proj_fraction_control_vs_art"])
        print("  Heretic moral-judgment drop  control / art:",
              k["moral_judgment_drop_control"], "/", k["moral_judgment_drop_art"])
        print("  Heretic dependency drop      control / art:",
              k["dependency_drop_control"], "/", k["dependency_drop_art"])


if __name__ == "__main__":
    main()
