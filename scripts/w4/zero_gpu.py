"""Zero-GPU stages of the W4 pod (Amendment 14.1 a–c, and the post-run disattenuation).

Runs before the pod (compute-ordering rule 1) from Paper 5's per-checkpoint proto-refusal caches,
which use the same construction as D1's ``refusal_base.npz`` (raw last-token diff-of-means over the
Heretic 400/400 set). Everything here reads local files only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deepsteer.geometry.reliability import adjacent_self_cosine, disattenuate_bootstrap, spearman_brown

from .common import REPO, _jsonable

P5_STAGE3 = REPO / "papers" / "5_moral_alignment" / "outputs" / "measurement" / "stage3"
D1_P2 = REPO / "papers" / "d1_moral_subspace" / "outputs" / "phase2"
STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 11900, 11921]


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _proto(label: str, layer: int) -> np.ndarray | None:
    p = P5_STAGE3 / label / "proto_refusal_directions.npz"
    if not p.exists():
        return None
    z = np.load(p)
    k = f"proto_refusal_layer{layer}"
    return _unit(z[k]) if k in z.files else None


def proto_refusal_trajectory(out_root: Path, dry: bool, layer: int = 16) -> dict:
    """(a) adjacent-checkpoint self-cosine + trajectory, (b) cache-consistency control, (c) proto→gate."""
    out_dir = out_root / "zero_gpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    if dry:
        rng = np.random.default_rng(0)
        base = rng.standard_normal(64)
        protos = {s: _unit(base + 0.1 * rng.standard_normal(64) * (12000 - s) / 12000) for s in STEPS}
        protos["main"] = _unit(base + 0.01 * rng.standard_normal(64))
        ref_base = _unit(base + 0.01 * rng.standard_normal(64))
        gate = _unit(rng.standard_normal(64))
        available = True
    else:
        protos = {s: _proto(f"olmo3_pretrain_stage3_step{s}", layer) for s in STEPS}
        protos["main"] = _proto("olmo3_base", layer)
        rb, ri = D1_P2 / "refusal_base.npz", D1_P2 / "refusal_instruct.npz"
        ref_base = _unit(np.load(rb)["refusal"]) if rb.exists() else None
        gate = _unit(np.load(ri)["refusal"]) if ri.exists() else None
        available = all(v is not None for v in protos.values()) and ref_base is not None and gate is not None
    res = {"stage": "14.1 zero-GPU arm", "layer": layer, "available": bool(available),
           "source": "papers/5_moral_alignment/outputs/measurement/stage3 (proto_refusal_directions.npz) + "
                     "papers/d1_moral_subspace/outputs/phase2/refusal_{base,instruct}.npz",
           "construction": "raw last-token diff-of-means, Heretic 400/400, identical to refusal_base.npz"}
    if not available:
        res["missing"] = [k for k, v in protos.items() if v is None] + \
                         [n for n, v in (("refusal_base", ref_base), ("refusal_instruct", gate)) if v is None]
        (out_dir / "proto_refusal_trajectory.json").write_text(json.dumps(res, indent=2))
        return res
    final = protos[11921]
    # (a) reliability under checkpoint drift: adjacent (21 steps) and the full self-trajectory
    res["rel_adj_11900_vs_11921"] = float(protos[11900] @ final)
    res["self_trajectory_vs_final"] = adjacent_self_cosine({s: protos[s] for s in STEPS}, 11921)
    res["rel_11000_vs_11921"] = float(protos[11000] @ final)
    # (b) cache-consistency positive control: Paper 5 main vs D1 refusal_base (same model, same construction)
    res["cache_consistency_cos_main_vs_refusal_base"] = float(protos["main"] @ ref_base)
    res["cache_consistency_pass"] = bool(res["cache_consistency_cos_main_vs_refusal_base"] >= 0.99)
    res["cos_step11921_vs_main"] = float(final @ protos["main"])
    # (c) proto-refusal -> instruct gate trajectory (traj(final) is the 0.155 of record)
    res["proto_to_gate_trajectory"] = {str(s): float(protos[s] @ gate) for s in STEPS}
    res["proto_to_gate_main"] = float(protos["main"] @ gate)
    res["proto_to_gate_refusal_base_of_record"] = float(ref_base @ gate)
    res["spearman_brown_of_rel_adj"] = spearman_brown(res["rel_adj_11900_vs_11921"])
    # (d) DESCRIPTIVE ladder rung (not in the frozen 14.1 branch rule; a Session-3 amendment may promote it):
    # chance |cos(u, gate)| for covariance-matched random directions u drawn from the BASE L16 content
    # act-sample (D1 phase-2 base/act_sample.npz). The 0.155 of record was judged against a 0.50
    # threshold only; this gives it a matched-null rung (instrument-calibration ladder).
    asp = D1_P2 / "base" / "act_sample.npz"
    if asp.exists() and not dry:                       # real 4096-d sample only (dry gate is synthetic)
        A = np.load(asp, allow_pickle=True)["X"].astype(np.float64)
        Ac = A - A.mean(0)
        rng = np.random.default_rng(0)
        draws = np.abs(np.array([_unit(Ac.T @ rng.standard_normal(Ac.shape[0])) @ gate for _ in range(2000)]))
        iso = np.abs(np.array([_unit(rng.standard_normal(gate.shape[0])) @ gate for _ in range(2000)]))
        res["gate_cosine_null_rung"] = {
            "act_sample": str(asp.relative_to(REPO)), "n_sample": int(A.shape[0]), "n_draws": 2000,
            "cov_matched_abs_cos_q50": float(np.median(draws)), "cov_matched_abs_cos_q95": float(np.percentile(draws, 95)),
            "isotropic_abs_cos_q95": float(np.percentile(iso, 95)),
            "isotropic_expectation_sqrt_2_over_pi_d": float(np.sqrt(2 / (np.pi * gate.shape[0]))),
            "note": "descriptive rung for the 0.155 (proto->gate) and for the per-checkpoint trajectory; "
                    "not a pre-registered verdict input"}
    res["reading_rule"] = ("Amendment 14.1: rel >= 0.9 -> branch A; <= 0.3 -> branch B; between -> "
                           "disattenuated cosine carries. The split-half arm (pod) supplies rel_proto; "
                           "this arm supplies the checkpoint-drift ceiling.")
    (out_dir / "proto_refusal_trajectory.json").write_text(json.dumps(_jsonable(res), indent=2))
    np.savez(out_dir / "proto_refusal_checkpoint_directions.npz",
             **{f"step{s}": protos[s] for s in STEPS}, main=protos["main"], refusal_base=ref_base, gate=gate)
    return res


def disattenuation_14_1(out_root: Path) -> dict | None:
    """Post-run: combine the proto (base) and gate (instruct) split-half records into cos_corr + CI."""
    pb = out_root / "olmo3_base" / "proto_refusal_split_half_directions.npz"
    pg = out_root / "olmo3_instruct" / "gate_split_half_directions.npz"
    tj = out_root / "zero_gpu" / "proto_refusal_trajectory.json"
    if not (pb.exists() and pg.exists()):
        return None
    a = np.load(pb, allow_pickle=True)["per_split"]
    b = np.load(pg, allow_pickle=True)["per_split"]
    cos_obs = 0.155
    if tj.exists():
        t = json.loads(tj.read_text())
        cos_obs = float(t.get("proto_to_gate_refusal_base_of_record", cos_obs))
    res = {"stage": "14.1 disattenuation", "cos_observed": cos_obs,
           "rel_proto_full": spearman_brown(float(np.mean(a))), "rel_gate_full": spearman_brown(float(np.mean(b))),
           **disattenuate_bootstrap(cos_obs, a, b, rng=np.random.default_rng(0))}
    (out_root / "zero_gpu" / "disattenuation_14_1.json").write_text(json.dumps(_jsonable(res), indent=2))
    return res
