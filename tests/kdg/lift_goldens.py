"""Golden outputs of the KDG pieces lifted into deepsteer.choice / deepsteer.stats (MB_PLAN §6).

Each function computes, through the names KDG callers import today, a value the lift must leave
byte-identical. ``python tests/kdg/lift_goldens.py`` writes ``goldens/kdg_lift_v1.json`` from the
current code; the goldens are committed before the lift and the tests stay unchanged through it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
KDG_SCRIPTS = REPO / "papers" / "kdg_panel" / "scripts"
DATA = REPO / "papers" / "kdg_panel" / "data"
P1E = REPO / "papers" / "kdg_panel" / "outputs" / "p1e"
KDG2 = REPO / "papers" / "kdg_panel" / "outputs" / "kdg2"
KDG3 = REPO / "papers" / "kdg_panel" / "outputs" / "kdg3"
GAP_JSON = DATA / "analysis_gap_twoway.json"
GOLDEN = Path(__file__).with_name("goldens") / "kdg_lift_v1.json"
CAL_SETS = ("v1", "v2_real", "v3_real", "v4_real")
sys.path.insert(0, str(KDG_SCRIPTS))
sys.path.insert(0, str(REPO))


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _cal_items(v: str) -> list[dict]:
    from deepsteer.kdg.calibration import load_calibration_set
    return load_calibration_set(DATA / f"calibration_set_{v}.json")


# ---- parser ---------------------------------------------------------------------------------


def calibration_parse() -> dict:
    """kdg.harness.parse_response on the four tracked calibration sets (800 replies)."""
    from deepsteer.kdg.harness import parse_response
    from deepsteer.kdg.schema import Option, Scenario
    out = {}
    for v in CAL_SETS:
        rows = []
        for it in _cal_items(v):
            p = parse_response(it["text"], Scenario.from_dict(it["scenario"]),
                               [(L, Option(**o)) for L, o in it["order"]])
            rows.append([it["id"], p.option_id, p.letter, p.method, p.norm_status])
        out[v] = {"n": len(rows), "sha256": _sha(rows),
                  "methods": dict(sorted(Counter(r[3] for r in rows).items()))}
    return out


def p1e_parse() -> dict | None:
    """parse_response on every rollout row of the banked pod p1e (local only)."""
    from deepsteer.kdg.harness import parse_response
    from deepsteer.kdg.schema import Option
    files = sorted(P1E.rglob("*.jsonl"))
    if not files:
        return None
    rows, skipped = [], 0
    for f in files:
        for i, line in enumerate(f.read_text().splitlines()):
            r = json.loads(line)
            if "text" not in r or "order" not in r:  # forced-readout rows carry no reply text
                skipped += 1
                continue
            order = [(L, Option(option_id=oid, text="", norm_status="neutral"))
                     for L, oid in r["order"].items()]
            p = parse_response(r["text"], types.SimpleNamespace(family=r["family"]), order)
            rows.append([f.relative_to(P1E).as_posix(), i, p.option_id, p.letter, p.method])
    return {"n": len(rows), "files": len(files), "skipped": skipped, "sha256": _sha(rows)}


def mb_parse() -> dict:
    """deepsteer.choice.parse (MB's R0 and R1) on the same 800 calibration replies."""
    from deepsteer.choice.parse import first_char, ladder
    rows = []
    for v in CAL_SETS:
        for it in _cal_items(v):
            t = it["text"]
            r1 = ladder(t, "ABCD")
            rows.append([it["id"], r1.label, r1.method, first_char(t, "AB").label,
                         first_char(t, "AB", strip=False).label])
    return {"n": len(rows), "sha256": _sha(rows)}


# ---- agreement / kappa ----------------------------------------------------------------------


def agreement_values() -> dict:
    """kdg.calibration agreement and Cohen's kappa: harness vs rater 1, rater 1 vs each rater 2."""
    from deepsteer.kdg.calibration import agreement, cohen_kappa, run_harness_on_set
    out = {}
    for v in CAL_SETS:
        items = _cal_items(v)
        gold, pred = [it["gold"] for it in items], run_harness_on_set(items)
        out[v] = {"harness_vs_r1": agreement(pred, gold),
                  "kappa_harness_vs_r1": cohen_kappa(pred, gold)}
        for f in sorted(DATA.glob(f"calibration_set_{v}_rater2_*.json")):
            r2 = json.loads(f.read_text())["labels"]
            if len(r2) == len(gold):
                key = f.stem.split("rater2_")[1]
                out[v][f"r1_vs_{key}"] = agreement(gold, r2)
                out[v][f"kappa_r1_vs_{key}"] = cohen_kappa(gold, r2)
    return out


# ---- stats ----------------------------------------------------------------------------------


def _readouts(seed: int, n: int, prefix: str = "S") -> list:
    from deepsteer.kdg.stats import ScenarioReadout
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        pv = rng.uniform(0.05, 0.95)
        d = [None if rng.uniform() < 0.1 else ("violating" if rng.uniform() < pv else "consistent")
             for _ in range(32)]
        stable = rng.uniform() < 0.8
        js = ["o1"] * 6 + ["o2"] * 2 if stable else ["o1"] * 4 + ["o2"] * 4
        out.append(ScenarioReadout(
            scenario_id=f"{prefix}{i:03d}", family=("F1", "F3", "F5")[i % 3], d_statuses=d,
            j_greedy="violating" if rng.uniform() < 0.15 else "consistent", j_sampled=[None] * 8,
            j_greedy_option="o1", j_sampled_options=js))
    return out


def stats_values() -> dict:
    """kdg.stats kdg_rate, difference_ci (paired, unpaired) and mde_rate_difference on fixed
    synthetic readouts (pure numpy: the same on every platform)."""
    from deepsteer.kdg.stats import difference_ci, kdg_rate, mde_rate_difference
    a, b = _readouts(11, 60), _readouts(22, 60)
    keep = ("rate", "ci95", "n_defined", "n_undefined", "diff", "n", "excludes_zero")
    pick = lambda d: {k: d[k] for k in keep if k in d}  # noqa: E731
    return {
        "kdg_rate_nested": pick(kdg_rate(a, n_boot=400, seed=0, nested=True)),
        "kdg_rate_flat": pick(kdg_rate(a, n_boot=400, seed=3, nested=False)),
        "diff_paired": pick(difference_ci(a, b, n_boot=400, seed=7, paired=True)),
        "diff_unpaired": pick(difference_ci(a, b, n_boot=400, seed=9, paired=False)),
        "mde": [mde_rate_difference(30, 40), mde_rate_difference(64, 64, 0.3, power=0.9)],
    }


# ---- readout --------------------------------------------------------------------------------


def mask_positions_value() -> list:
    import kdg_pod_lib as lib
    import torch
    m = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1]])
    return lib.mask_positions(m).tolist()


def _platform() -> dict:
    import torch
    import transformers
    return {"machine": platform.machine(), "system": platform.system(),
            "torch": torch.__version__, "transformers": transformers.__version__}


def readout_value() -> dict | None:
    """kdg_pod_lib.ModelWrapper.raw_next_logprobs on the tiny bf16 fixture of
    test_kdg_readout_v2 (local only: the Qwen2.5 tokenizer and this platform's bf16 kernels)."""
    import kdg_pod_lib as lib
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001
        return None
    tok.padding_side = "left"
    cfg = LlamaConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tok),
                      max_position_embeddings=4096)
    torch.manual_seed(0)
    fx = types.SimpleNamespace(tok=tok, model=LlamaForCausalLM(cfg).to(torch.bfloat16).eval(),
                               device=torch.device("cpu"), torch=torch)
    words = "the agent weighs the request against the posted rule and answers with one letter"
    prompts = [" ".join(words.split()[: 4 + 2 * k]) * (1 + k) for k in range(8)]
    read = lib.ModelWrapper.raw_next_logprobs
    out = {"platform": _platform()}
    for name, kw in {"v2_b8": {"batch_size": 8}, "v2_b1": {"batch_size": 1},
                     "v1_b8": {"batch_size": 8, "readout_version": 1}}.items():
        arr = read(fx, prompts, add_special_tokens=False, **kw)
        out[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape),
                     "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    return out


# ---- analysis of record ---------------------------------------------------------------------


def gap_twoway_sha() -> str | None:
    """Re-run analyze_gap_twoway.py (kdg_rate and difference_ci on the banked kdg2/kdg3 pods; it
    asserts the registered intervals reproduce) and hash its output; the committed file is
    restored afterwards whatever happens (local only)."""
    if not (KDG2.is_dir() and KDG3.is_dir()):
        return None
    backup = GAP_JSON.read_bytes()
    try:
        subprocess.run([sys.executable, str(KDG_SCRIPTS / "analyze_gap_twoway.py")], cwd=REPO,
                       check=True, capture_output=True, text=True)
        return hashlib.sha256(GAP_JSON.read_bytes()).hexdigest()
    finally:
        GAP_JSON.write_bytes(backup)


def compute_all() -> dict:
    return {
        "calibration_parse": calibration_parse(), "p1e_parse": p1e_parse(), "mb_parse": mb_parse(),
        "agreement": agreement_values(), "stats": stats_values(),
        "mask_positions": mask_positions_value(), "readout": readout_value(),
        "gap_twoway_sha256": gap_twoway_sha(),
        "gap_twoway_committed_sha256": hashlib.sha256(GAP_JSON.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    data = compute_all()
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN.relative_to(REPO)}")
