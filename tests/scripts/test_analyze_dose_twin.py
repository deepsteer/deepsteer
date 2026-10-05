"""P1-A12 analysis (analyze_dose_twin.py): sign convention and the pre-registered branch rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "papers" / "kdg_panel" / "scripts"))
import analyze_dose_twin as T  # noqa: E402


def _cells(rng, n, drop_prim, drop_twin, noise=0.01):
    ids = [f"S{i}" for i in range(n)]
    f = rng.uniform(0.3, 0.7, n)
    P = {"F": dict(zip(ids, f)), "TF": dict(zip(ids, f))}
    P["D2"] = dict(zip(ids, f - drop_prim + rng.normal(0, noise, n)))
    tw = rng.uniform(0.2, 0.5, n)
    Q = {"F": dict(zip(ids, tw)), "TF": dict(zip(ids, tw))}
    Q["D2"] = dict(zip(ids, tw - drop_twin + rng.normal(0, noise, n)))
    return P, Q, ids


def test_reasoning_lowering_primary_more_than_twin_is_branch_a():
    # most probable failure: ΔE_delib computed as Δ_T − Δ_P (sign flipped), so a pressure-specific
    # reduction reads as unresolved and an at-rest-only drop reads as branch (a)
    P, Q, ids = _cells(np.random.default_rng(0), 120, drop_prim=0.10, drop_twin=0.0)
    r = T.contrast(P, Q, ids)
    assert r["vs_F"]["delta_E_delib"]["mean"] < 0
    assert r["verdict"] == "reduces_pressure_part"


def test_equal_drop_on_primary_and_twin_is_branch_b():
    # twins identical to primaries: ΔE_delib exactly 0 per scenario, Δ_T exactly −0.10
    P, _, ids = _cells(np.random.default_rng(1), 120, drop_prim=0.10, drop_twin=0.10, noise=0.0)
    r = T.contrast(P, {k: dict(v) for k, v in P.items()}, ids)
    assert r["verdict"] == "removes_at_rest_asymmetry"


def test_no_drop_anywhere_is_unresolved():
    P, Q, ids = _cells(np.random.default_rng(2), 120, drop_prim=0.0, drop_twin=0.0)
    assert T.contrast(P, Q, ids)["verdict"] == "unresolved"
