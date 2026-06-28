#!/usr/bin/env python3
"""Local (no-GPU) structural test of the Phase 2 pipeline.

Fabricates the stage-0 artifacts with synthetic vectors, then runs the pure-numpy stages
(G-AXIS -> assemble -> null -> G3) to verify the two structural constraints hold, WITHOUT a
model. This is the test-gates-before-GPU local gate; a VALIDATE=1 remote smoke follows.

Asserts:
  1. G-AXIS branches: high cross-source cos -> two-source; low cos -> single-source fallback.
  2. Hard null sequence: phase2_g3 aborts if null_artifact.json is absent; succeeds after
     phase2_null writes it. (The predates-the-result property is structural.)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
sys.path.insert(0, str(S.parents[1] / "5_moral_alignment" / "scripts"))
import direction_utils as du  # noqa: E402

HIDDEN, LAYER = 256, 16
RNG = np.random.default_rng(0)


def fabricate(art: Path, cross_source_cos: float) -> None:
    """Write synthetic stage-0 artifacts with a controllable MORABLES vs MS axis angle."""
    art.mkdir(parents=True, exist_ok=True)
    base = RNG.standard_normal(HIDDEN)
    base /= np.linalg.norm(base)
    other = RNG.standard_normal(HIDDEN)
    other -= other @ base * base
    other /= np.linalg.norm(other)
    d_ms = base
    d_mb = cross_source_cos * base + np.sqrt(1 - cross_source_cos**2) * other

    du.save_directions(art / "moral_directions.npz",
                       {"moral_stories": {LAYER: d_ms}, "morables": {LAYER: d_mb}})
    # diff matrices: pairs concentrated along each source axis + noise
    for name, axis in (("moral_stories", d_ms), ("morables", d_mb)):
        M = np.outer(RNG.uniform(0.5, 1.5, 120), axis) + 0.1 * RNG.standard_normal((120, HIDDEN))
        np.savez(art / f"diffs_{name}.npz", **{f"layer{LAYER}": M})
    np.savez(art / "act_sample.npz", X=RNG.standard_normal((400, HIDDEN)), layer=LAYER)
    persona = RNG.standard_normal(HIDDEN)
    persona /= np.linalg.norm(persona)
    du.save_directions(art / "persona_direction.npz", {"persona": {LAYER: persona}})
    json.dump({"model": "synthetic", "n_layers": 32, "band": [LAYER],
               "match_layer": LAYER, "validate": True, "sources_extracted":
               ["moral_stories", "morables"]}, open(art / "extract_meta.json", "w"))


def run(stage: str, art: Path):
    return subprocess.run([sys.executable, str(S / stage), "--artifacts", str(art)],
                          capture_output=True, text=True)


def main() -> None:
    ok = True
    # --- Constraint 2: G-AXIS branches ---
    for cos, want in ((0.90, "pass"), (0.30, "fail")):
        with tempfile.TemporaryDirectory() as d:
            art = Path(d)
            fabricate(art, cos)
            run("phase2_gaxis.py", art)
            dec = json.load(open(art / "g_axis_decision.json"))
            got = dec["decision"]
            srcs = dec["v_moral_sources"]
            branch_ok = (got == want and
                         (srcs == ["moral_stories", "morables"] if want == "pass"
                          else srcs == ["moral_stories"]))
            ok &= branch_ok
            print(f"[G-AXIS cos~{cos}] decision={got} sources={srcs} -> "
                  f"{'OK' if branch_ok else 'FAIL'}")
            if want == "pass":
                run("phase2_assemble_vmoral.py", art)
                vm = np.load(art / "v_moral.npz", allow_pickle=True)
                print(f"  V_moral eff_dim={int(vm['eff_dim'])} basis={vm['basis'].shape}")

                # --- Constraint 1: hard null sequence (g3 takes two same-model artifact dirs) ---
                g3_no_null = subprocess.run(
                    [sys.executable, str(S / "phase2_g3.py"),
                     "--base-artifacts", str(art), "--instruct-artifacts", str(art)],
                    capture_output=True, text=True)
                aborted = g3_no_null.returncode != 0 and "null_artifact" in g3_no_null.stderr
                ok &= aborted
                print(f"  G3 WITHOUT null -> {'ABORTS (OK)' if aborted else 'NO ABORT (FAIL)'}")

                null_run = run("phase2_null.py", art)
                has_artifact = (art / "null_artifact.json").exists()
                ok &= has_artifact and null_run.returncode == 0
                if has_artifact:
                    na = json.load(open(art / "null_artifact.json"))
                    ok &= na.get("predates_refusal_projection") is True
                    print(f"  null artifact: q95={na['q95']} "
                          f"c={na['control_c_persona_projection']} frozen={na['frozen']} "
                          f"predates_refusal={na['predates_refusal_projection']}")

    print("\n" + ("ALL STRUCTURAL CHECKS PASSED" if ok else "STRUCTURAL CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
