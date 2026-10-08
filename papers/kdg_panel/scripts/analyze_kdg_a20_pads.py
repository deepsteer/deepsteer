#!/usr/bin/env python3
"""ANOMALIES KDG-A20 pad-count check (zero GPU; rule pre-registered and pushed in 3db23c8).

    python3 papers/kdg_panel/scripts/analyze_kdg_a20_pads.py

For each panel model's four C3 letter cells of record, rebuild the batches (16 consecutive rows in
file order), re-render every row's prompt with the model's pinned tokenizer and template, require
the row's ``prompt_sha256`` to match, and count each row's left padding (longest prompt in its batch
minus its own length). Reports pads for primary vs pressure-removed twin cells, the per-batch
distribution, the pads of the saved VALIDATE batches (rebuilt by replaying
``cell_forward_matches_generate``), and the E-aligned contrast
Δpad_s = mean over permutations of (pad_D − pad_J) − (pad_Dtwin − pad_Jtwin) with corr(E_s, Δpad_s).
"""

from __future__ import annotations

import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402
import kdg_pod_lib as lib  # noqa: E402
from pod_kdg_phase1 import registry  # noqa: E402

from deepsteer.kdg.schema import assign_letters, letter_map, load_scenario_dir  # noqa: E402

BATCH = 16  # raw_next_logprobs default since 8b115d6 (2026-09-26), before every cell of record
CELLS = {  # cell -> (frame, variant); prefix neutral throughout
    "dl_chat_neutral": ("agent", "primary"),
    "jl_chat_neutral": ("eval", "primary"),
    "dl_chat_neutral_pressure_removed": ("agent", "pressure_removed"),
    "jl_chat_neutral_pressure_removed": ("eval", "pressure_removed"),
}
MODELS = {  # analysis key -> registry key
    "olmo3_final": "olmo3_instruct",
    "llama31_instruct_meta": "llama31_instruct_meta",
    "tulu3_final": "tulu3_final",
    "qwen25_instruct": "qwen25_instruct_p1",
}
VALIDATE = {  # analysis key -> [(run, scenario-id file or None, n recorded, dir)]
    "olmo3_final": [
        ("p1a", None, 8, "p1a/final_validate/olmo3_instruct"),
        ("p2b", "screened_ids_a17_union.json", 10, "p2b/olmo3_instruct/olmo3_instruct"),
    ],
    "llama31_instruct_meta": [
        ("p2b", "screened_ids_llama31_meta.json", 10,
         "p2b/llama31_instruct_meta/llama31_instruct_meta"),
    ],
}


def tokenizer(reg_key: str):
    from transformers import AutoTokenizer

    spec = registry()[1][reg_key]
    return AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])


def render(tok, msgs) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def ntok(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def cell_pads(tok, d: Path, cell: str, scen: dict) -> list[dict]:
    frame, variant = CELLS[cell]
    rows = A._rows(d / f"{cell}.jsonl")
    lens, bad = [], 0
    for r in rows:
        s = scen[r["scenario_id"]]
        order = assign_letters(s, r["seed"])
        assert letter_map(order) == r["order"], (cell, s.id, "order drift")
        p = render(tok, lib.letter_chat_messages(s, order, frame, "neutral", variant))
        bad += lib.sha256_text(p) != r["prompt_sha256"]
        lens.append(ntok(tok, p))
    if bad:
        raise SystemExit(f"{d.name}/{cell}: {bad} of {len(rows)} rows do not re-render to their "
                         "prompt_sha256; reconstruction invalid for this model (rule item 1)")
    out = []
    for i in range(0, len(rows), BATCH):
        L = lens[i : i + BATCH]
        for k, r in enumerate(rows[i : i + BATCH]):
            out.append({"scenario_id": r["scenario_id"], "seed": r["seed"], "batch": i // BATCH,
                        "len": L[k], "pad": max(L) - L[k]})
    return out


def validate_pads(tok, files, run: str, ids_file: str | None, n: int) -> list[int]:
    """Replay cell_forward_matches_generate's prompt list with a recording stand-in model."""
    scen, _ = load_scenario_dir(files)
    if ids_file:
        keep = set(json.loads((A.DATA / ids_file).read_text())["ids"])
        scen = [s for s in scen if s.id in keep]
    seen: list[str] = []

    def raw(prompts, **_):
        seen.extend(prompts)
        return np.zeros((len(prompts), 8), np.float16)

    def gen(prompts, **_):
        return [lib.GenOut("", [0], np.zeros(8, np.float16), None, 0) for _ in prompts]

    m = types.SimpleNamespace(render_chat=lambda msgs: render(tok, msgs), raw_next_logprobs=raw,
                              generate=gen, token_id=lambda s: 0,
                              encode=lambda x: tok.encode(x, add_special_tokens=False))
    ctx = types.SimpleNamespace(model=m, out=Path("/dev/null/never"), dry=True)
    try:
        lib.cell_forward_matches_generate(ctx, scen)
    except (OSError, NotADirectoryError):
        pass  # the record write at the end; the prompts are already captured
    # the first raw call is the G6 batch; p1a's code had no TSN extras, so its first n prompts are
    # the batch of record (the pad-ladder batch added 2026-10-07 comes after and is dropped)
    seen = seen[:n]
    L = [ntok(tok, p) for p in seen]
    return [max(L) - x for x in L]


def q(x) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}


def boot_ci(f, n: int, rng) -> list[float]:
    vals = [f(rng.integers(0, n, n)) for _ in range(A.N_BOOT)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def main() -> int:
    files = sorted(A.DATA.glob("*_scenarios_*.json"))
    S, _ = load_scenario_dir(files)
    scen = {s.id: s for s in S}
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in S
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    rep: dict = {"rule": "ANOMALIES KDG-A20 pre-registration (3db23c8)", "batch": BATCH,
                 "n_boot": A.N_BOOT, "seed": A.SEED, "models": {}}
    for key, reg_key in MODELS.items():
        d = SR.MODELS[key]
        tok = tokenizer(reg_key)
        pads = {c: cell_pads(tok, d, c, scen) for c in CELLS}
        prim = [r["pad"] for c in CELLS if "pressure_removed" not in c for r in pads[c]]
        twin = [r["pad"] for c in CELLS if "pressure_removed" in c for r in pads[c]]
        batch_max = [max(r["pad"] for r in pads[c] if r["batch"] == b)
                     for c in CELLS for b in {r["batch"] for r in pads[c]}]
        allp = prim + twin
        per = {c: defaultdict(list) for c in CELLS}
        for c in CELLS:
            for r in pads[c]:
                per[c][r["scenario_id"]].append(r["pad"])
        T = A.four_cells([d], tuple(CELLS), status)
        free = sorted(s for s in T if T[s]["mass_min"] >= A.FLOOR)
        m = {c: {s: float(np.mean(v)) for s, v in per[c].items()} for c in CELLS}
        dl, jl, dlt, jlt = (m[c] for c in CELLS)
        dpad = np.array([(dl[s] - jl[s]) - (dlt[s] - jlt[s]) for s in free])
        E = np.array([A.scales(T[s])["E_prob"] for s in free])
        rng = np.random.default_rng(A.SEED)
        mean_ci = boot_ci(lambda i: dpad[i].mean(), len(free), rng)
        corr = float(np.corrcoef(E, dpad)[0, 1]) if dpad.std() > 0 else 0.0
        rng = np.random.default_rng(A.SEED)
        corr_ci = (boot_ci(lambda i: np.corrcoef(E[i], dpad[i])[0, 1], len(free), rng)
                   if dpad.std() > 0 else [0.0, 0.0])
        val = []
        for run, ids_file, n, vdir in VALIDATE.get(key, []):
            vp = validate_pads(tok, files, run, ids_file, n)
            rec = json.loads((A.OUT / vdir / "forward_matches_generate.json").read_text())
            val.append({"run": run, "n": n, "max_pad": max(vp), "pads": vp,
                        "max_abs_nats": rec["max_abs_nats"], "n_recorded": rec["n"]})
        vmax = max((v["max_pad"] for v in val), default=None)
        if vmax is None or max(allp) > vmax:
            verdict = "bound_does_not_transfer"
        elif mean_ci[0] > 0 or mean_ci[1] < 0 or corr_ci[0] > 0 or corr_ci[1] < 0:
            verdict = "aligned_channel"
        else:
            verdict = "bounded_no_aligned_channel"
        rep["models"][key] = {
            "rows": len(allp), "sha_verified_rows": len(allp),
            "pad_primary": q(prim), "pad_twin": q(twin),
            "share_rows_pad0": float(np.mean(np.array(allp) == 0)),
            "batch_max_pad": {**q(batch_max), "q25": float(np.percentile(batch_max, 25)),
                              "q75": float(np.percentile(batch_max, 75)),
                              "n_batches": len(batch_max)},
            "validate_batches": val, "validate_max_pad": vmax,
            "rows_above_validate_max_pad": None if vmax is None else int(sum(
                p > vmax for p in allp)),
            "n_model_free": len(free),
            "dpad_mean": float(dpad.mean()), "dpad_ci95": mean_ci,
            "corr_E_dpad": corr, "corr_ci95": corr_ci,
            "verdict": verdict,
        }
    (A.DATA / "analysis_kdg_a20_pads.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep["models"].items():
        print(f"{k:22s} prim {v['pad_primary']['mean']:5.2f}/{v['pad_primary']['max']:4.0f}  "
              f"twin {v['pad_twin']['mean']:5.2f}/{v['pad_twin']['max']:4.0f}  "
              f"pad0 {v['share_rows_pad0']:.2f}  batchmax med {v['batch_max_pad']['median']:.0f} "
              f"p90 {v['batch_max_pad']['p90']:.0f} max {v['batch_max_pad']['max']:.0f}  "
              f"VAL {v['validate_max_pad']}  dpad {v['dpad_mean']:+.3f} {v['dpad_ci95']}  "
              f"r {v['corr_E_dpad']:+.3f} {v['corr_ci95']}  -> {v['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
