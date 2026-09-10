"""Tests for the W4 pod driver (scripts/pod_w4.py + scripts/w4/*): dry-run end-to-end, manifest
integrity, the MISSING_ARTIFACTS closure map, the rank-2/4 harm-capture curve, and the pooled-sweep
analysis. No model is loaded anywhere in this file."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "papers" / "d3_decision_anatomy" / "scripts"))

from w4.common import MISSING_ARTIFACTS_CLOSURE, PANEL, verify_manifest  # noqa: E402
from w4.units_tier_b import pilot_gate, pooled_sweep_analysis  # noqa: E402

_spec = importlib.util.spec_from_file_location("pod_w4", REPO / "scripts" / "pod_w4.py")
pod_w4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pod_w4)


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("w4_dry")
    manifest = pod_w4.run(out, dry=True, models=None, units=None)
    return out, json.loads(Path(manifest).read_text())


class TestDryRunEndToEnd:
    def test_every_unit_of_every_model_ran_ok(self, dry_run):
        # assert no unit path raised on random tensors (the Session-2 "known to run end-to-end" gate)
        _, m = dry_run
        failed = {k: v for k, v in m["unit_status"].items() if v["status"] == "failed"}
        assert not failed, failed
        for spec in PANEL:
            for u in spec.units:
                assert m["unit_status"][f"{spec.key}/{u}"]["status"] == "ok", (spec.key, u)

    def test_order_is_preregistered(self, dry_run):
        # assert model batching follows Amendment 14: Instruct, Think, base, Llama, GPT-OSS, Qwen
        _, m = dry_run
        assert [l["key"] for l in m["loads"]] == ["olmo3_instruct", "olmo3_think", "olmo3_base",
                                                 "llama31", "gpt_oss_20b", "qwen25"]

    def test_manifest_hashes_verify(self, dry_run):
        # assert every saved artifact has a sha256 that recomputes (the supplement-style contract)
        out, m = dry_run
        assert m["artifacts"], "no artifacts saved"
        assert all(re.fullmatch(r"[0-9a-f]{64}", a["sha256"]) for a in m["artifacts"])
        assert verify_manifest(out / "manifest_w4.json") == []

    def test_pilot_gates_recorded(self, dry_run):
        _, m = dry_run
        assert "15.1_pilot_olmo3" in m["pilot_gates"] and "15.2_pilot_qwen25" in m["pilot_gates"]

    def test_loads_record_revision_hash_field(self, dry_run):
        # assert the FL App E.4 gap is closed structurally: every load carries commit_hash
        _, m = dry_run
        assert all("commit_hash" in l and "revision_requested" in l for l in m["loads"])

    def test_mandatory_per_unit_saves_exist(self, dry_run):
        # assert the Amendment 14/15 artifact lists are produced (names are the save contract)
        out, _ = dry_run
        must = ["olmo3_base/proto_refusal_samples.npz", "olmo3_base/proto_refusal_split_half_directions.npz",
                "olmo3_instruct/gate_samples.npz", "olmo3_instruct/axis_diffs_fables.npz",
                "olmo3_instruct/mean_content_slices.npz", "olmo3_instruct/cross_ablation_outcomes.npz",
                "olmo3_instruct/pooled_sweep_olmo3_w4.json", "olmo3_think/p0p3_rollouts.npz",
                "olmo3_think/mft_directions.npz", "olmo3_think/refusal_P0.npz", "llama31/severity_contrasts_L12.npz",
                "llama31/harm_capture_L12.json", "llama31/reply_inversion_margins.npz",
                "gpt_oss_20b/decision_token_sample.npz", "gpt_oss_20b/graded_reads.npz",
                "gpt_oss_20b/decision_token_reread.json", "gpt_oss_20b/p0p3_rollouts.npz",
                "qwen25/position_samples.npz", "qwen25/qwen25_read_cell.json",
                "zero_gpu/proto_refusal_trajectory.json", "zero_gpu/disattenuation_14_1.json"]
        missing = [p for p in must if not (out / p).exists()]
        assert not missing, missing

    def test_cli_dry_run_subprocess(self, tmp_path):
        # assert the CLI entry point itself runs (what remote_w4.sh calls before any model load)
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "pod_w4.py"), "--dry-run",
                            "--models", "olmo3_base", "--out", str(tmp_path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert (tmp_path / "manifest_w4.json").exists()


class TestClosureMap:
    def test_every_missing_artifact_entry_has_a_closing_unit(self):
        # assert the ledger is fully covered: one closure key per unique bullet in MISSING_ARTIFACTS.md
        md = (REPO / "papers" / "MISSING_ARTIFACTS.md").read_text()
        bullets = [l for l in md.splitlines() if l.startswith("- ")]
        unique = {b.split("->")[0].strip() for b in bullets}
        assert len(unique) == len(MISSING_ARTIFACTS_CLOSURE), (len(unique), len(MISSING_ARTIFACTS_CLOSURE))
        units = {u for s in PANEL for u in s.units}
        for v in MISSING_ARTIFACTS_CLOSURE.values():
            unit = v.split("/")[1].split(" ")[0]
            assert unit in units, v


class TestHarmCaptureRank24:
    def test_capture_rank_2_and_4_recover_planted_overlap(self):
        # assert a harm basis spanning moral PCs 1-2 captures them at rank 2 and nothing else (Amendment 14.2)
        import sweep as sw
        rng = np.random.default_rng(0)
        d = 48
        pcs = np.linalg.qr(rng.standard_normal((d, 16)))[0]
        harm_rows = np.concatenate([pcs[:, :2] @ rng.standard_normal((2, 30)) * 5.0,
                                    pcs[:, 2:4] @ rng.standard_normal((2, 30)) * 0.5], 1).T  # rank ~2 then 4
        H = sw.nested_pca_basis(harm_rows, [1, 2, 4])
        w = {1: 0.5, 2: 0.5, 3: 0.0, 4: 0.0}
        cap = sw.harm_capture_curve(H, pcs, w)
        assert cap[2]["engage_weighted_capture"] > 0.9
        assert cap[4]["engage_weighted_capture"] >= cap[2]["engage_weighted_capture"] - 1e-9
        assert all(c < 0.05 for c in cap[2]["per_pc_capture"][4:])
        # weight on PCs the harm basis never touches -> low capture (the "beyond harm" branch)
        cap_off = sw.harm_capture_curve(H, pcs, {5: 1.0, 6: 1.0})
        assert cap_off[4]["engage_weighted_capture"] < 0.05

    def test_nested_bases_nest(self):
        import sweep as sw
        X = np.random.default_rng(1).standard_normal((20, 32))
        H = sw.nested_pca_basis(X, [1, 2, 4])
        assert np.allclose(H[1], H[2][:, :1]) and np.allclose(H[2], H[4][:, :2])


class TestPooledSweep:
    def _npz(self, tmp_path, n_orig=23, n_w4=18):
        rng = np.random.default_rng(0)
        from deepsteer.datasets import get_request_twins_union, w4_set_tags
        foll = [a for _f, a, _b in get_request_twins_union()]
        tags = w4_set_tags()
        oi = [i for i, t in enumerate(tags) if t == "original"][:n_orig]
        wi = [i for i, t in enumerate(tags) if t == "w4"][:n_w4]
        idx = oi + wi
        full = -np.abs(rng.standard_normal(len(idx))) - 0.2
        p = tmp_path / "c1_inputs_x.npz"
        np.savez(p, cell_full_deltas=full, cell_restricted_deltas=full * 0.3, cell_harm_deltas=full * 0.3,
                 full_judgment_deltas=np.abs(rng.standard_normal(50)) + 0.2,
                 rt_following=np.array([foll[i] for i in idx], dtype=object), sweep_ks=np.array([1, 3, 8, 16]),
                 sweep_refusal=np.stack([full * f for f in (0.02, 0.31, 0.29, 0.28)]),
                 sweep_judgment=np.stack([(np.abs(rng.standard_normal(50)) + 0.2) * f for f in (0.05, 0.46, 0.59, 0.66)]),
                 sweep_random=np.stack([full * 0.01 for _ in range(4)]))
        return p, {a: t for a, t in zip(foll, tags)}

    def test_set_split_and_pooled_rule(self, tmp_path):
        # assert the per-twin set tags split into original/w4/pooled with the pooled-primary sign rule
        p, set_of = self._npz(tmp_path)
        res = pooled_sweep_analysis(p, set_of, np.random.default_rng(0), n_boot=100)
        assert res["replication_original"]["n"] == 23 and res["alone_w4"]["n"] == 18 and res["pooled"]["n"] == 41
        assert res["pooled"]["shape_verdict"]["verdict"] == "harm_saturating"
        assert res["pooled_is_primary"] is True
        lo, hi = res["pooled"]["R_refusal_ci95"][16]
        assert lo <= res["pooled"]["R_refusal_k"][16] <= hi

    def test_pilot_gate_sign_rule(self, tmp_path):
        # assert the pilot gate fails on sign-incoherent deltas and passes on coherent ones
        good = tmp_path / "good.npz"; bad = tmp_path / "bad.npz"
        np.savez(good, cell_full_deltas=np.array([-1, -2, -1, -3, -2.0]), cell_restricted_deltas=np.zeros(5))
        np.savez(bad, cell_full_deltas=np.array([-1, 2, -1, 3, 2.0]), cell_restricted_deltas=np.zeros(5))
        assert pilot_gate(good)["passed"] is True
        assert pilot_gate(bad)["passed"] is False
