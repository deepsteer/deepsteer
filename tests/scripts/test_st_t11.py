"""ST T1.1 driver (scripts/t11_driver.py): dry phases end to end, the registered class positions,
and the single-row bank forward on a tiny checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t11_driver as T  # noqa: E402


def _phase(out: Path, phase: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / "t11_driver.py"), "--phase", phase,
                           "--out", str(out), "--dry-run", "--date-pin", "2026-10-19"],
                          capture_output=True, text=True, cwd=ROOT)


def test_dry_phases_tag_selection_and_rerun_and_compute_the_floor(tmp_path):
    # most probable failure (Amendment 1): the re-run reuses the selecting rows (or is untagged),
    # so the floor rung P_none^re - P_none^sel is computed against itself
    pytest.importorskip("transformers")
    for ph in ("v1", "h"):
        r = _phase(tmp_path, ph)
        assert r.returncode == 0, r.stderr[-2000:]
    d = tmp_path / "gpt_oss_20b"
    rows = [json.loads(x) for x in (d / "gen" / "h1_rollouts.jsonl").read_text().splitlines()]
    assert rows and all(x["sample"] == "selection" for x in rows)
    sel = json.loads((d / "select" / "engaged.json").read_text())
    held = sorted({x["scenario_id"] for x in rows if x["split"] == "held"})[:2]
    sel["held"]["engaged_ids"] = held  # force a subset (the stub engages none)
    (d / "select" / "engaged.json").write_text(json.dumps(sel))
    r = _phase(tmp_path, "v2")
    assert r.returncode == 0, r.stderr[-2000:]
    re = [json.loads(x) for x in (d / "rerun" / "h1_held_engaged_rerun.jsonl").read_text()
          .splitlines()]
    assert {x["sample"] for x in re} == {"rerun"} and {x["scenario_id"] for x in re} == set(held)
    rec = json.loads((d / "t11_record.json").read_text())
    assert rec["phases"]["v2"]["rerun"]["floor_P_re_minus_P_sel"]["n_scenarios"] == 2
    # seeds of the re-run are disjoint from the selecting sample's
    assert not ({x["vllm_seed"] for x in re} & {x["vllm_seed"] for x in rows})
    h = rec["phases"]["h"]
    assert h["layers_hidden_index"] == [b + 1 for b in T.BANK_BLOCKS] and h["single_row"]
    assert (d / "bank" / "index.jsonl").exists() and list((d / "bank").glob("bank_*.npz"))
    assert set(h["chat_sigma"]) == {str(b + 1) for b in T.BANK_BLOCKS}


def test_class_positions_on_a_harmony_trace():
    # the registered classes (§1 + T1.1): onset = first 48 trace tokens, stride32 at trace
    # offsets 0, 32, 64..., nothing before the analysis header or after <|end|>
    pytest.importorskip("transformers")
    import kdg_harmony as kh
    import pod_st_screen as P
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    spec = P.registry()["gpt_oss_20b"]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = build_format("gpt_oss_20b", tok, spec)
    body = [1000 + i for i in range(80)]
    gen = list(kh.ANALYSIS_OPEN) + body + [kh.END] + list(kh.CANON_HEADER) + [32]
    cp = T.class_positions(fmt, tok, gen, np.zeros(len(tok), bool))
    assert min(cp) == 3 and max(cp) < 3 + 80  # nothing at or after <|end|> is classed
    assert [i for i, c in cp.items() if "stride32" in c] == [3, 35, 67]
    assert sum("onset" in c for c in cp.values()) == 48


def test_forward_bank_matches_the_readout_of_record(tmp_path):
    # most probable failure: the bank's letter log-probs come from a different position than the
    # readout of record (read_ids on the prefix ending before the letter), or the trace log-prob
    # sums the wrong tokens
    torch = pytest.importorskip("torch")
    import screen_lib as L
    import st_tiny
    from kdg_pod_lib import ModelWrapper

    path = st_tiny.build("olmo3_think", tmp_path / "ck")
    m = ModelWrapper(str(path), None, torch_dtype=torch.float32)
    sc = types.SimpleNamespace(model=m, dry=False, layers=lambda: [2, 3],
                               letters={32 + i: "ABCDE"[i] for i in range(5)})
    sc_real = L.Screen("olmo3_think", {"n_layers": 4}, m, None, tmp_path, "2026-10-19",
                       read_batch=1)
    sc_real.layers_override = [2, 3]
    ids = m.tok.encode("Pick one of the options and explain the reason", add_special_tokens=False)
    pos = [3, len(ids) - 2]
    resid, lp, tlp = T.forward_bank(sc_real, ids, pos, len(ids) - 2, logprob_from=4)
    assert resid.shape == (2, 2, m.model.config.hidden_size)
    ref, rres = L.read_ids(sc_real, [ids[: len(ids) - 1]], 1)
    assert np.allclose(lp, ref[0], atol=1e-4)
    assert np.allclose(resid[1].astype(np.float32), rres[0].astype(np.float32), atol=2e-2)
    with torch.no_grad():
        lg = torch.log_softmax(m.model(input_ids=torch.tensor([ids], device=m.model.device))
                               .logits[0].float(), -1)
    want = float(sum(lg[t - 1, ids[t]] for t in range(4, len(ids))))
    assert tlp == pytest.approx(want, abs=1e-3)
    del sc
