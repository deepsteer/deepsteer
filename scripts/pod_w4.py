#!/usr/bin/env python3
"""W4 venue-quality pod driver (D3 PREREGISTRATION Amendments 14 + 15, 2026-09-10).

Runs Tier A then Tier B, batched by loaded model in the pre-registered order
OLMo-3-Instruct -> OLMo-3-Think -> OLMo-3 base -> Llama-3.1-8B-Instruct -> GPT-OSS-20B ->
Qwen2.5-7B-Instruct, with the zero-GPU arm of 14.1 first. Per-unit arrays are the mandatory
outputs; every saved file's sha256 and every load's resolved HF commit hash land in
``papers/d3_decision_anatomy/outputs/w4/manifest_w4.json``.

    python3 scripts/pod_w4.py --dry-run                 # every code path on random tensors, no model
    python3 scripts/pod_w4.py --zero-gpu-only           # 14.1 (a-c) from Paper 5 caches, no model
    python3 scripts/pod_w4.py --closure-map             # MISSING_ARTIFACTS.md -> unit map
    python3 scripts/pod_w4.py --verify-manifest         # recompute every sha256
    python3 scripts/pod_w4.py --models olmo3_instruct,llama31 --units 14.6,14.6b   # real, subset
    python3 scripts/pod_w4.py --merge-from papers/d3_decision_anatomy/outputs/w4_rerun1 --reason "..."  # fold a rerun in

Session W4-2 launches the real run through scripts/remote_w4.sh (RunPod). Verdict rules are applied
in Session W4-3 from the saved arrays; this driver computes and saves, it does not judge.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from w4 import zero_gpu  # noqa: E402
from w4.common import (  # noqa: E402
    MISSING_ARTIFACTS_CLOSURE,
    PANEL,
    REPO,
    W4_OUT,
    Ctx,
    Manifest,
    load_record,
    merge_manifest,
    verify_manifest,
)
from w4.extractors import RealExtractor, StubExtractor  # noqa: E402
from w4.units_tier_a import UNITS as UNITS_A  # noqa: E402
from w4.units_tier_b import UNITS as UNITS_B  # noqa: E402

UNITS = {**UNITS_A, **UNITS_B}
TIER_A = tuple(k for k in UNITS_A)
TIER_B = tuple(k for k in UNITS_B)


def _load_model(spec, dry: bool):
    if dry:
        return None
    from deepsteer.directions import extraction as du
    if spec.revision:
        from deepsteer.core.model_interface import WhiteBoxModel
        from deepsteer.core.types import AccessTier
        return WhiteBoxModel(spec.repo, access_tier=AccessTier.WEIGHTS, revision=spec.revision)
    return du.load_whitebox(spec.repo)


def _extractor(spec, model, dry: bool, rng):
    if dry:
        return StubExtractor(spec.hidden, spec.n_layers, rng)
    cot = None
    if spec.kind == "reasoning_moe":
        from deepsteer.reasoning.think_io import CoTFormat
        cot = CoTFormat.HARMONY_ANALYSIS
    return RealExtractor(model, cot_format=cot)


def run(out_root: Path, dry: bool, models: list[str] | None, units: list[str] | None,
        tiers: tuple[str, ...] = ("A", "B")) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out_root, dry)
    # zero-GPU first (compute-ordering rule 1)
    zg = zero_gpu.proto_refusal_trajectory(out_root, dry)
    manifest.status("14.1_zero_gpu", "ok" if zg.get("available") else "missing_inputs", str(zg.get("missing", "")))
    for spec in PANEL:
        if models and spec.key not in models:
            continue
        wanted = [u for u in spec.units if (not units or u in units)
                  and (("A" in tiers and u in TIER_A) or ("B" in tiers and u in TIER_B))]
        if not wanted:
            continue
        print(f"==== {spec.key} ({spec.repo}) units={wanted}", flush=True)
        model = _load_model(spec, dry)
        try:
            manifest.add_load(load_record(model, spec, dry))
            ctx = Ctx(spec, _extractor(spec, model, dry, np.random.default_rng(0)), out_root, dry, manifest)
            # Tier A units first within a model, then Tier B (15.1 is the OLMo keystone: pilot inside)
            for u in sorted(wanted, key=lambda k: (k in TIER_B, wanted.index(k))):
                t0 = time.time()
                try:
                    UNITS[u](ctx)
                    manifest.status(f"{spec.key}/{u}", "ok", f"{time.time() - t0:.1f}s")
                except Exception as e:  # one failed unit never kills the pod; it is recorded
                    manifest.status(f"{spec.key}/{u}", "failed", f"{type(e).__name__}: {e}")
                    traceback.print_exc()
                manifest.write()   # checkpoint after every unit
        finally:
            if model is not None:
                model.release()
    # post-run zero-GPU combination (14.1 disattenuation needs both split-half saves)
    dis = zero_gpu.disattenuation_14_1(out_root)
    manifest.status("14.1_disattenuation", "ok" if dis else "pending_split_half_saves")
    return manifest.write()


def main() -> int:
    ap = argparse.ArgumentParser(description="W4 pod driver (Amendments 14/15).")
    ap.add_argument("--dry-run", action="store_true", help="random tensors, no model; exercises every path")
    ap.add_argument("--zero-gpu-only", action="store_true", help="only the 14.1 zero-GPU arm")
    ap.add_argument("--closure-map", action="store_true", help="print the MISSING_ARTIFACTS -> unit map")
    ap.add_argument("--verify-manifest", action="store_true")
    ap.add_argument("--merge-from", default=None, metavar="DIR",
                    help="fold a partial rerun's output dir (own manifest) into --out's manifest of record")
    ap.add_argument("--reason", default="", help="with --merge-from: why the rerun happened")
    ap.add_argument("--models", default=None, help="comma list of panel keys (default: all, in order)")
    ap.add_argument("--units", default=None, help="comma list of unit ids (default: all for the model)")
    ap.add_argument("--tier", default="AB", help="A, B, or AB")
    ap.add_argument("--out", default=None,
                    help=f"output root (default: {W4_OUT}; a --dry-run without --out goes to _dry/)")
    args = ap.parse_args()
    # A dry run must never write random-tensor artifacts into the real output tree.
    out_root = Path(args.out) if args.out else (W4_OUT / "_dry" if args.dry_run else W4_OUT)

    if args.closure_map:
        for k, v in MISSING_ARTIFACTS_CLOSURE.items():
            print(f"{k:75s} -> {v}")
        return 0
    if args.merge_from:
        p = merge_manifest(out_root, Path(args.merge_from), args.reason)
        print(f"merged -> {p}; manifest OK")
        return 0
    if args.verify_manifest:
        bad = verify_manifest(out_root / "manifest_w4.json")
        print("manifest OK" if not bad else "\n".join(bad))
        return 0 if not bad else 1
    if args.zero_gpu_only:
        res = zero_gpu.proto_refusal_trajectory(out_root, args.dry_run)
        import json
        print(json.dumps({k: v for k, v in res.items() if k not in ("self_trajectory_vs_final",)}, indent=2))
        return 0
    models = args.models.split(",") if args.models else None
    units = args.units.split(",") if args.units else None
    tiers = tuple(t for t in "AB" if t in args.tier.upper())
    p = run(out_root, args.dry_run, models, units, tiers)
    print(f"manifest -> {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
