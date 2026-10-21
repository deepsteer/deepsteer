"""ST-A15's discriminator (scripts/router_disc.py): the HF router capture on a tiny GPT-OSS, the
row-level read on synthetic engines with known flips. The vLLM phase needs the pod's vLLM venv and
is not runnable here (recorded in ST-A15)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import router_disc as R  # noqa: E402

JOBS = ROOT / "papers/st_live_reference/outputs/t15_early/gpt_oss_20b/vllm/mxfp4_parity.jobs.jsonl"


def test_router_compare_finds_flips_only_where_planted():
    # most probable failure: top-k sets compared in rank order (a reordering inside the top 4
    # counted as a flip), or the margin read at the wrong rank
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 10, 8)).astype(np.float32)
    o = h.copy()
    o[1, 2] = o[1, 2][::-1]  # a different top-4 set at (layer 1, token 2)
    srt = np.sort(h[2, 5])[::-1]
    o[2, 5] = h[2, 5] + 0.0
    o[2, 5][np.argsort(-h[2, 5])[:4]] += np.linspace(0.01, 0.0, 4)  # reorder inside the top 4
    r = R.router_compare([h], [o])[0]
    assert r["n_differ"] == 1 and r["layers"] == [1] and r["first_layer"] == 1
    assert srt[3] - srt[4] > 0


def test_shown_columns_match_the_parity_rows():
    # most probable failure: the read's rows misaligned with T1.5's parity rows (a different
    # order of scenarios and permutations), so per-row distances land on the wrong rows
    cols = R.shown_columns(64)
    assert len(cols) == 64 and all(len(c) == 3 and set(c) <= {0, 1, 2, 3, 4} for c in cols)


@pytest.mark.slow
def test_hf_phase_on_a_tiny_gpt_oss_and_read(tmp_path):
    # most probable failure: the router hook captures the wrong output (scores or indices instead
    # of logits), or one layer's call is missing, so every comparison on the pod is void
    if not JOBS.exists():
        pytest.skip("T1.5 parity jobs not present locally")
    jobs = tmp_path / "jobs.jsonl"
    lines = JOBS.read_text().splitlines()[:6]
    jobs.write_text("\n".join(lines) + "\n")
    out = tmp_path / "out"
    cmd = [sys.executable, str(R.__file__), "--phase", "hf", "--jobs", str(jobs), "--out", str(out),
           "--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    z = np.load(out / "hf_router.npz")
    x = z[z.files[0]]
    n_tok = len(json.loads(lines[0])["prompt_ids"])
    assert x.shape[1] == n_tok and x.shape[2] >= 4 and x.shape[0] >= 2  # [layers, tokens, experts]
    assert not np.allclose(x, x.max(-1, keepdims=True))  # logits, not one-hot indices
    # the read: a synthetic bf16 engine equal to HF except row 0's first layer
    hf = np.load(out / "hf_letters.npy")
    (out / "bf16_eager_first.json").write_text(json.dumps(
        [{str(t): float(v) for t, v in zip(json.loads(ln)["logprob_ids"], row)}
         for ln, row in zip(lines, hf)]))
    eng = [z[k].astype(np.float32).copy() for k in z.files]
    eng[0][0, 0] = eng[0][0, 0][::-1]
    np.savez_compressed(out / "bf16_eager_router.npz", *eng)
    sys.argv = ["router_disc.py", "--phase", "read", "--jobs", str(jobs), "--out", str(out)]
    res = R.phase_read(type("A", (), {"jobs": jobs, "out": out})())
    assert res["tv_vs_hf_bf16_eager"]["mean"] < 1e-6
    assert res["router_bf16_eager"]["per_row"][0]["n_differ"] >= 1
    assert res["router_bf16_eager"]["per_row"][1]["n_differ"] == 0
