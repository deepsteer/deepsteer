#!/usr/bin/env python3
"""Local pre-flight for the Tier-2 malleability code path (Task 2.1 + 2.2).

Run BEFORE any RunPod spend. Exercises the production analysis path on the REAL
cached base directions with synthetic proto-refusal contrasts (no model), then
optionally the real per-checkpoint extraction on the cached OLMo-3 base.

  1. subspace_projection_fraction sanity: an in-span vector projects to ~1, an
     orthogonal vector to ~0.
  2. end-to-end analysis on the real cached base MFT subspace
     (outputs/pipeline/olmo3_base/probe_directions.npz) with two synthetic
     proto-refusal contrasts: a foundation direction (projection ~1) and a
     random vector (projection small). Confirms load + basis + projection +
     malleability_analysis.analyse_state on real 4096-dim data.
  3. (--real) extract_proto_refusal on the cached base (few prompts) then
     malleability_analysis, the actual production path at small scale.

Exits non-zero on any failure, so it can gate the launcher.

Usage:
    python papers/5_moral_alignment/scripts/local_test_malleability.py
    python papers/5_moral_alignment/scripts/local_test_malleability.py --real --device mps
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepsteer.directions import extraction as du  # noqa: E402
import malleability_analysis as ma  # noqa: E402
from heretic_ablation import subspace_projection_fraction  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER  # noqa: E402

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_BASE_PIPE = _PAPER_ROOT / "outputs/pipeline/olmo3_base/probe_directions.npz"


def part1_projection_sanity() -> bool:
    print("\n[1] subspace_projection_fraction sanity (synthetic)")
    rng = np.random.default_rng(0)
    basis = [rng.standard_normal(64).astype(np.float32) for _ in range(6)]
    in_span = sum(c * b for c, b in zip(rng.standard_normal(6), basis))
    M = np.stack(basis, axis=1)
    # A vector outside the span: random then project OFF the span.
    r = rng.standard_normal(64)
    coef, *_ = np.linalg.lstsq(M, r, rcond=None)
    orth = r - M @ coef
    f_in = subspace_projection_fraction(in_span, basis)
    f_out = subspace_projection_fraction(orth, basis)
    print(f"    in-span frac {f_in:.4f} (expect ~1), orthogonal frac {f_out:.4f} (expect ~0)")
    ok = f_in > 0.999 and f_out < 1e-3
    print("    OK" if ok else "    FAIL")
    return ok


def part2_real_basis_synth_proto() -> bool:
    print("\n[2] analyse_state on real cached base MFT + synthetic proto-refusal")
    if not _BASE_PIPE.exists():
        print(f"    SKIP: {_BASE_PIPE} not found")
        return True
    pipe = du.load_directions(_BASE_PIPE)
    foundations = [f for f in FOUNDATION_ORDER if f in pipe]
    n_layers = max(pipe[foundations[0]]) + 1

    with tempfile.TemporaryDirectory() as td:
        stage3 = Path(td) / "stage3"
        # synthetic proto-refusal == the care_harm foundation direction (in-span)
        aligned = {L: pipe["care_harm"][L].astype(np.float32) for L in range(n_layers)}
        (stage3 / "olmo3_base").mkdir(parents=True)
        du.save_directions(stage3 / "olmo3_base" / "proto_refusal_directions.npz",
                           {"proto_refusal": aligned})
        r = ma.analyse_state("olmo3_base", stage3, _PAPER_ROOT / "outputs/pipeline",
                             None, list(range(15, 32)), 16)
        proj_aligned = r["proj_norm_headline"]

        rng = np.random.default_rng(1)
        randd = {L: rng.standard_normal(4096).astype(np.float32) for L in range(n_layers)}
        du.save_directions(stage3 / "olmo3_base" / "proto_refusal_directions.npz",
                           {"proto_refusal": randd})
        r2 = ma.analyse_state("olmo3_base", stage3, _PAPER_ROOT / "outputs/pipeline",
                              None, list(range(15, 32)), 16)
        proj_rand = r2["proj_norm_headline"]

    print(f"    aligned-proto projection @L16 {proj_aligned:.4f} (expect ~1)")
    print(f"    random-proto  projection @L16 {proj_rand:.4f} (expect small, 6/4096 dims)")
    ok = proj_aligned > 0.99 and proj_rand < 0.2
    print("    OK" if ok else "    FAIL")
    return ok


def part3_real(device: str) -> bool:
    print("\n[3] real extraction smoke on cached OLMo-3 base (few prompts) + analysis")
    import subprocess
    scripts = _PAPER_ROOT / "scripts"
    with tempfile.TemporaryDirectory() as td:
        s3 = Path(td) / "stage3"
        e = subprocess.run([sys.executable, str(scripts / "extract_proto_refusal.py"),
                            "--only", "olmo3_base", "--max-prompts", "4",
                            "--allow-fallback", "--device", device,
                            "--output-dir", str(s3)], capture_output=True, text=True)
        print("    [extract]", e.stdout.strip().splitlines()[-1] if e.stdout else e.stderr[-300:])
        if e.returncode != 0:
            print("    FAIL: extraction errored\n", e.stderr[-800:])
            return False
        a = subprocess.run([sys.executable, str(scripts / "malleability_analysis.py"),
                            "--stage3-dir", str(s3), "--no-figure",
                            "--output-dir", str(Path(td) / "out")],
                           capture_output=True, text=True)
        print("    [analysis]", a.stdout.strip().splitlines()[-1] if a.stdout else a.stderr[-300:])
        ok = a.returncode == 0
    print("    OK" if ok else "    FAIL")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Local pre-flight for Tier-2 malleability code.")
    ap.add_argument("--real", action="store_true",
                    help="Also run the real base extraction smoke (loads 7B; needs it cached).")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    results = {
        "projection_sanity": part1_projection_sanity(),
        "real_basis_synth_proto": part2_real_basis_synth_proto(),
    }
    if args.real:
        results["real_extraction"] = part3_real(args.device)

    print("\n" + "=" * 60)
    failed = [k for k, v in results.items() if not v]
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if failed:
        print(f"\nLOCAL TEST FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("\nALL LOCAL CHECKS PASSED — safe to run the RunPod dry-run (VALIDATE=1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
