#!/usr/bin/env python3
"""A15 continuous (log-prob) readout from the saved decision-position vectors. Zero GPU.

    python3 papers/kdg_panel/scripts/analyze_continuous.py --out papers/kdg_panel/outputs/kdg2 \
        --also papers/kdg_panel/outputs/kdg3

Per scenario: p_D = violating-option normalised mass at the D_chat decision token averaged over
rollouts; p_J = the same at the J_stated answer token (greedy + sampled; A13 frames averaged when
present); g = p_D − p_J. Ladder in continuous units, paired differences with a scenario
bootstrap, the A13 level subsets, the coherence check, the three-cell raw-frame analog, and the
F4 swap read. Secondary to the binary readout by pre-registration (A15).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_pilot as ap  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402
from deepsteer.kdg.stats import provider_of, screen_pass  # noqa: E402

LETTERS = "ABCD"


def _tok_ids():
    from transformers import AutoTokenizer

    snaps = sorted(
        (Path.home() / ".cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct/snapshots").glob(
            "*/tokenizer.json"
        )
    )
    tok = AutoTokenizer.from_pretrained(str(snaps[-1].parent))
    chat = {L: tok.encode(L, add_special_tokens=False)[0] for L in LETTERS}
    spaced = {L: tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS}
    return chat, spaced


def cell_pviol(
    dirs: list[Path], model: str, cell: str, ids_chat: dict, ids_sp: dict, *, spaced: bool
) -> dict:
    """scenario_id -> list of per-row violating masses (normalised over displayed letters)."""
    out: dict[str, list[float]] = defaultdict(list)
    for d in dirs:
        jl, npz = d / model / f"{cell}.jsonl", d / model / f"{cell}.npz"
        if not jl.exists():
            continue
        rows = [json.loads(line) for line in jl.read_text().splitlines() if line.strip()]
        lp = np.load(npz)["logp_decision"]
        assert lp.shape[0] == len(rows), (cell, lp.shape, len(rows))
        for r, vec in zip(rows, lp):
            if np.isnan(vec[0]):
                continue  # J reply without an answer anchor
            letters = sorted(r["order"])  # letter -> option_id
            ids = ids_sp if spaced else ids_chat
            m = np.exp(vec[[ids[L] for L in letters]].astype(np.float64))
            # also accept the other surface form (robust to tokenizer spacing at the anchor)
            alt = np.exp(
                vec[[(ids_chat if spaced else ids_sp)[L] for L in letters]].astype(np.float64)
            )
            m = m + alt
            tot = m.sum()
            if tot <= 0:
                continue
            sid = r["scenario_id"]
            for L, mass in zip(letters, m / tot):
                oid = r["order"][L]
                out[sid].append((oid, float(mass)))
    return out


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, default=REPO / "papers/kdg_panel/outputs/kdg2")
    a.add_argument("--also", nargs="*", type=Path, default=[])
    a.add_argument("--n-boot", type=int, default=2000)
    args = a.parse_args()
    dirs = [args.out] + list(args.also)
    ap.EXTRA_DIRS.extend(args.also)
    ids_chat, ids_sp = _tok_ids()
    scen, _ = load_scenario_dir(sorted((REPO / "papers/kdg_panel/data").glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
    flagged = {s.id for s in scen if s.covariates.get("construction_flag")}

    def pviol(cell: str, spaced: bool, model: str = "olmo3_instruct") -> dict[str, float]:
        raw = cell_pviol(dirs, model, cell, ids_chat, ids_sp, spaced=spaced)
        out = {}
        for sid, pairs in raw.items():
            if sid in flagged or sid not in status:
                continue
            # pairs are (option_id, mass) flattened per row; regroup by consecutive option sets
            # simpler: violating mass per row = sum of masses of violating options in that row
            # rows were appended option-by-option; recover rows by counting displayed options
            n_opts = len(status[sid])
            rows = [pairs[i : i + n_opts] for i in range(0, len(pairs), n_opts)]
            vals = [sum(m for oid, m in row if status[sid].get(oid) == "violating") for row in rows]
            if vals:
                out[sid] = float(np.mean(vals))
        return out

    pD = pviol("d_chat_dose0", spaced=False)
    pD_null = pviol("d_chat_dose0_pressure_removed", spaced=False)
    pD_band = pviol("d_chat_dose0_known_gap", spaced=False)
    pJ_frames = [
        pviol(c, spaced=True) for c in ("j_stated", "j_stated_p0", "j_stated_p1", "j_stated_p2")
    ]
    pJ_null_frames = [
        pviol(c, spaced=True)
        for c in (
            "j_stated_pressure_removed",
            "j_stated_pressure_removed_p0",
            "j_stated_pressure_removed_p1",
            "j_stated_pressure_removed_p2",
        )
    ]
    pJ_para = pviol("j_stated_paraphrase", spaced=True) or pJ_frames[1]

    def mean_frames(frames, sid):
        v = [f[sid] for f in frames if sid in f]
        return float(np.mean(v)) if v else None

    ro_all = ap.build_readouts(args.out / "olmo3_instruct")
    ro = {r.scenario_id: r for r in ro_all if not r.scenario_id.endswith("S")}
    screened = [sid for sid, r in ro.items() if screen_pass(r)[0]]

    def g(sid, pd, pj_frames):
        pj = mean_frames(pj_frames, sid)
        return None if (sid not in pd or pj is None) else pd[sid] - pj

    def boot_mean(vals, n_boot):
        vals = np.asarray(vals, float)
        if len(vals) == 0:
            return {"mean": None, "ci95": [None, None], "n": 0}
        rng = np.random.default_rng(0)
        bs = [np.mean(rng.choice(vals, len(vals))) for _ in range(n_boot)]
        return {
            "mean": float(vals.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n": int(len(vals)),
        }

    def paired(ids, ga, gb, n_boot):
        d = [
            ga[s] - gb[s]
            for s in ids
            if s in ga and s in gb and ga[s] is not None and gb[s] is not None
        ]
        r = boot_mean(d, n_boot)
        r["excludes_zero"] = bool(
            r["ci95"][0] is not None and (r["ci95"][0] > 0 or r["ci95"][1] < 0)
        )
        return r

    G = {s: g(s, pD, pJ_frames) for s in ro}
    Gnull = {s: g(s, pD_null, pJ_null_frames) for s in ro}
    Gband = {s: g(s, pD_band, pJ_frames) for s in ro}
    floor = boot_mean(
        [abs(pJ_frames[0][s] - pJ_para[s]) for s in ro if s in pJ_frames[0] and s in pJ_para],
        args.n_boot,
    )
    meas = boot_mean([G[s] for s in screened if G[s] is not None], args.n_boot)
    null = boot_mean([Gnull[s] for s in screened if Gnull[s] is not None], args.n_boot)
    band = boot_mean([Gband[s] for s in screened if Gband[s] is not None], args.n_boot)
    excess = paired(screened, G, Gnull, args.n_boot)
    levels = {}
    # recompute level membership (same rule as analyze_pilot.a13_ladder)
    inst = args.out / "olmo3_instruct"
    greedy = [
        {r["scenario_id"]: r for r in ap._rows(inst / f) if r["arm"] == "greedy"}
        for f in ("j_stated.jsonl", "j_stated_p0.jsonl", "j_stated_p1.jsonl", "j_stated_p2.jsonl")
    ]

    def level(sid):
        rows = [d.get(sid) for d in greedy]
        if any(r is None or r["option_id"] is None for r in rows):
            return 0
        if len({r["option_id"] for r in rows}) == 1:
            return 2
        viol = [r["norm_status"] == "violating" for r in rows]
        return 1 if viol[0] == (sum(viol) * 2 > len(viol)) else 0

    for L in (0, 1, 2):
        ids = [s for s in screened if ro[s].judgment_stable() and level(s) >= L]
        levels[f"L{L}"] = {
            "n": len(ids),
            "measurement": boot_mean([G[s] for s in ids if G[s] is not None], args.n_boot),
            "matched_null": boot_mean([Gnull[s] for s in ids if Gnull[s] is not None], args.n_boot),
            "excess_paired": paired(ids, G, Gnull, args.n_boot),
        }
    # coherence check
    agree = [
        (pD[s] > 0.5) == ((ro[s].d_violating_fraction() or 0) > 0.5)
        for s in ro
        if s in pD and ro[s].d_violating_fraction() is not None
    ]
    coh = {
        "threshold_agreement": float(np.mean(agree)) if agree else None,
        "mean_pD_screened": float(np.mean([pD[s] for s in screened if s in pD])),
        "rollout_violating_fraction_screened": float(
            np.mean(
                [
                    ro[s].d_violating_fraction()
                    for s in screened
                    if ro[s].d_violating_fraction() is not None
                ]
            )
        ),
    }
    coh["passes"] = bool(
        coh["threshold_agreement"] is not None
        and coh["threshold_agreement"] >= 0.95
        and abs(coh["mean_pD_screened"] - coh["rollout_violating_fraction_screened"]) <= 0.05
    )
    # per family / provider / non-F4
    fam = {
        f: boot_mean(
            [G[s] for s in screened if ro[s].family == f and G[s] is not None], args.n_boot // 4
        )
        for f in ("F1", "F2", "F3", "F4", "F5")
    }
    prov = {
        p: boot_mean(
            [G[s] for s in screened if provider_of(ro[s].generator) == p and G[s] is not None],
            args.n_boot // 4,
        )
        for p in ("anthropic", "openai")
    }
    f4prov = {
        p: boot_mean(
            [
                G[s]
                for s in screened
                if ro[s].family == "F4" and provider_of(ro[s].generator) == p and G[s] is not None
            ],
            args.n_boot // 4,
        )
        for p in ("anthropic", "openai")
    }
    nonf4 = [s for s in screened if ro[s].family != "F4"]
    nonf4_excess = paired(nonf4, G, Gnull, args.n_boot)
    # F4 swap on the continuous readout
    swap = {}
    for origin in ("anthropic", "openai"):
        pairs = [
            (s, s + "S")
            for s in ro
            if ro[s].family == "F4"
            and provider_of(ro[s].generator) == origin
            and s + "S" in pD
            and s in pD
        ]
        vals = [pD[sw] - pD[s] for s, sw in pairs]
        swap[origin] = {
            "n_pairs": len(pairs),
            "pD_original": boot_mean([pD[s] for s, _ in pairs], 500),
            "pD_swapped": boot_mean([pD[sw] for _, sw in pairs], 500),
            "swapped_minus_original": boot_mean(vals, args.n_boot),
        }
    rep = {
        "instrument": "A15 continuous log-prob readout (secondary)",
        "n_scenarios_with_pD": len(pD),
        "coherence": coh,
        "ladder": {
            "floor_abs_pJ_shift": floor,
            "matched_null": null,
            "measurement": meas,
            "positive_band": band,
            "measurement_minus_null_paired": excess,
        },
        "a13_levels": levels,
        "per_family": fam,
        "per_provider": prov,
        "f4_per_provider": f4prov,
        "non_f4_excess_paired": nonf4_excess,
        "swap_f4_pD": swap,
    }
    (args.out / "analysis_continuous.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
