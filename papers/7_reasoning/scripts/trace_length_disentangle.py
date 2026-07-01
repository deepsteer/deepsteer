#!/usr/bin/env python3
"""Phase 2a analysis: is GPT-OSS's distributed trace morality real, or a length artifact?

Reads the position-resolved ``trace_profile.npz`` for each model (+ its moral
subspace) and asks whether GPT-OSS's higher trace-level moral content survives
when length and position are controlled. LOCAL, no GPU.

Two controlled views of the harmful-harmless trace-mean-diff moral fraction:

  * **matched-length** — recompute the moral fraction from the mean over the first
    N trace tokens, for a grid of N, using only prompts whose trace is at least N
    long. Comparing models at the SAME N removes the absolute-length confound.
  * **fractional-position** — moral fraction per fractional-position bin (0=start,
    1=end), independent of absolute length, showing WHERE in the trace moral
    content sits and whether it is spread or concentrated.

Verdict (per the user mandate): at the largest N with adequate n in all models,
does GPT-OSS's matched-length moral fraction still exceed the distills'? If yes,
the trace-distribution axis survives and Phase 2b (causal load-bearing) is
warranted; if no, the axis is a length artifact and the Phase-1 negative stands.

Usage:
    python papers/7_reasoning/scripts/trace_length_disentangle.py \
        --outputs-dir papers/7_reasoning/outputs --keys gpt_oss_20b,ds_r1_llama8b,ds_r1_qwen14b \
        --output papers/7_reasoning/outputs/trace_length_disentangle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepsteer.directions import extraction as du  # noqa: E402
from moral_dependency import build_subspace_basis  # noqa: E402

_MIN_N = 8  # minimum prompts/class to report a matched-length or bin estimate


_unit = du.unit_vector  # shared: deepsteer.directions.extraction.unit_vector


def _mft(direction, basis):
    c = basis @ _unit(direction)
    return float(c @ c)


def _meandiff_frac(Hrows, Srows, basis):
    """Moral fraction of the harmful-harmless mean-diff over valid (finite) rows."""
    Hv = Hrows[np.isfinite(Hrows).all(1)]
    Sv = Srows[np.isfinite(Srows).all(1)]
    if len(Hv) < _MIN_N or len(Sv) < _MIN_N:
        return None, len(Hv), len(Sv)
    d = Hv.mean(0) - Sv.mean(0)
    return round(_mft(d, basis), 6), len(Hv), len(Sv)


def analyze_model(prof_npz: Path, moral_npz: Path) -> dict:
    z = np.load(prof_npz, allow_pickle=True)
    headline = int(z["headline"])
    grid = [int(x) for x in z["prefix_grid"]]
    K = int(z["k_bins"])
    moral = du.load_directions(str(moral_npz))
    n_layers = 1 + max(L for d in moral.values() for L in d)
    basis = build_subspace_basis(moral, kind="probe", n_layers=n_layers)[0][headline]

    Hc, Sc = z["H_coherent"], z["S_coherent"]
    # closed flags may be absent in older profiles; default to coherent.
    Hcl = z["H_closed"] if "H_closed" in z.files else Hc
    Scl = z["S_closed"] if "S_closed" in z.files else Sc
    Htl, Stl = z["H_trace_len"], z["S_trace_len"]
    dtypes = [str(x) for x in z["float_dtypes"]] if "float_dtypes" in z.files else None

    # --- matched-length prefix (SECONDARY: truncation-phase CONFOUNDED) ---------
    # First-N tokens of a long trace is its OPENING, not the same cognitive phase
    # as a complete short trace. Reported for diagnosis, NOT the verdict.
    matched = {}
    for i, N in enumerate(grid):
        Hrows = z["H_prefix"][Hc & (Htl >= N), i, :]
        Srows = z["S_prefix"][Sc & (Stl >= N), i, :]
        frac, nh, ns = _meandiff_frac(Hrows, Srows, basis)
        matched[str(N)] = {"moral_frac": frac, "n_harmful": nh, "n_harmless": ns}

    # --- fractional-position bins on CLOSED traces (PRIMARY: matched phase) ------
    # Bin j is "fraction j/K through THIS trace's complete deliberation", so it
    # compares mid-deliberation to mid-deliberation regardless of absolute length.
    # Restricted to closed traces so each binned trace is a full start->decision
    # deliberation (an unclosed trace's bins are a truncated prefix).
    Hmask, Smask = Hc & Hcl, Sc & Scl
    position, frac_vals = {}, []
    for j in range(K):
        Hrows = z["H_bins"][Hmask, j, :]
        Srows = z["S_bins"][Smask, j, :]
        frac, nh, ns = _meandiff_frac(Hrows, Srows, basis)
        position[str(j)] = {"frac_pos": round((j + 0.5) / K, 2), "moral_frac": frac,
                            "n_harmful": nh, "n_harmless": ns}
        if frac is not None:
            frac_vals.append(frac)
    # Length-normalized "distributed moral content": equal weight per fractional
    # slice (the matched-phase analog of Phase 1's token-weighted CoT-mean).
    frac_mean = round(float(np.mean(frac_vals)), 6) if frac_vals else None

    return {"headline_layer": headline, "float_dtypes": dtypes,
            "harmful_trace_len_median": int(np.median(Htl)),
            "harmless_trace_len_median": int(np.median(Stl)),
            "n_closed_harmful": int(Hmask.sum()), "n_closed_harmless": int(Smask.sum()),
            "closed_frac_harmful": round(float(Hcl[Hc].mean()), 3) if Hc.any() else None,
            "closed_frac_harmless": round(float(Scl[Sc].mean()), 3) if Sc.any() else None,
            "matched_length_CONFOUNDED": matched,
            "fractional_position": position, "fractional_mean_moral": frac_mean}


def compute_verdict(results: dict, gpt: str = "gpt_oss_20b") -> dict:
    """The go/no-go, keyed to the FRACTIONAL-position comparison (matched phase).

    The decision (`survives`) reads the fractional-position-mean moral fraction on
    CLOSED traces — which compares GPT-OSS's mid-deliberation to the distills'
    mid-deliberation regardless of absolute length. The matched-length prefix
    comparison is reported as a SECONDARY diagnostic only and is flagged
    CONFOUNDED: at GPT-OSS's short trace length it pits GPT-OSS's complete
    deliberation against the distills' opening tokens (different cognitive phase),
    so it is NOT allowed to drive the decision.
    """
    distills = [k for k in results if k != gpt]
    dts = {k: results[k].get("float_dtypes") for k in results}
    precision_ok = (bool(results) and all(v is not None for v in dts.values())
                    and len({tuple(v) for v in dts.values()}) == 1)

    def _cmp(getter, label):
        if gpt not in results or not distills:
            return {"survives": None, "note": "needs gpt_oss_20b + >=1 distill"}
        gf = getter(results[gpt])
        dfs = {k: getter(results[k]) for k in distills}
        if gf is None or any(v is None for v in dfs.values()):
            return {"survives": None, "note": f"insufficient data for {label}"}
        return {"gpt_oss": gf, "distills": dfs, "survives": bool(all(gf > dfs[k] for k in distills))}

    frac_v = _cmp(lambda r: r.get("fractional_mean_moral"), "fractional")

    Ns = sorted({int(n) for r in results.values() for n in r.get("matched_length_CONFOUNDED", {})})
    shared = [N for N in Ns if all(
        results[k]["matched_length_CONFOUNDED"].get(str(N), {}).get("moral_frac") is not None
        for k in results)]
    ml_v = {"survives": None}
    if gpt in results and distills and shared:
        Nstar = max(shared)
        ml_v = _cmp(lambda r, N=Nstar: r["matched_length_CONFOUNDED"][str(N)]["moral_frac"], "matched")
        ml_v["matched_N"] = Nstar

    sf, sm = frac_v.get("survives"), ml_v.get("survives")
    reading = ("trace-distribution axis SURVIVES at matched cognitive phase "
               "-> Phase 2b (causal load-bearing) warranted." if sf else
               "GPT-OSS does NOT exceed the distills at matched fractional position "
               "-> trace-distribution signal is a length artifact; drop the axis, the "
               "Phase-1 negative stands.") if sf is not None else \
              "INCONCLUSIVE: too few closed traces for the fractional verdict (raise budget)."
    verdict = {
        "decision_metric": "fractional_position_closed_traces (matched cognitive phase)",
        "survives": sf, "reading": reading,
        "fractional_position_verdict": frac_v,
        "matched_length_verdict_CONFOUNDED": ml_v,
        "both_methods_agree": (sf is not None and sm is not None and sf == sm),
        "precision_parity_ok": precision_ok,
        "float_dtypes_per_model": dts,
    }
    if not precision_ok:
        verdict["precision_warning"] = (
            "models profiled at DIFFERENT float precisions; a moral-fraction difference "
            "may be precision, not length — fix before trusting the verdict.")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2a trace-length disentanglement (local)")
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--keys", default="gpt_oss_20b,ds_r1_llama8b,ds_r1_qwen14b")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.outputs_dir)
    keys = [k.strip() for k in args.keys.split(",")]
    results = {}
    for k in keys:
        prof = root / k / "trace_profile.npz"
        moral = root / k / "exp1_probe_directions.npz"
        if not prof.exists() or not moral.exists():
            print(f"[skip] {k}: missing {prof.name if not prof.exists() else moral.name}")
            continue
        results[k] = analyze_model(prof, moral)

    verdict = compute_verdict(results)
    sf = verdict["survives"]
    sm = verdict["matched_length_verdict_CONFOUNDED"].get("survives")
    both_agree = verdict["both_methods_agree"]
    precision_ok = verdict["precision_parity_ok"]
    reading = verdict["reading"]

    payload = {"analysis": "phase2a_trace_length_disentangle",
               "min_n_per_class": _MIN_N, "models": results, "verdict": verdict}
    Path(args.output).write_text(json.dumps(payload, indent=2))

    # PRIMARY table: fractional-position moral fraction (matched phase, closed).
    print("\n=== fractional-position moral fraction (CLOSED traces, matched phase) [PRIMARY] ===")
    print(f"  {'bin':>4} {'pos':>5}  " + "  ".join(f"{k:>14}" for k in results))
    for j in range(10):
        cells = []
        for k in results:
            m = results[k]["fractional_position"].get(str(j), {})
            f = m.get("moral_frac")
            cells.append(f"{f:>8.5f} (n{m.get('n_harmful', 0):>2})" if f is not None
                         else f"{'--':>8} (n{m.get('n_harmful', 0):>2})")
        pos = results[next(iter(results))]["fractional_position"].get(str(j), {}).get("frac_pos", "")
        print(f"  {j:>4} {pos:>5}  " + "  ".join(f"{c:>16}" for c in cells))
    print("  " + "-" * 60)
    print(f"  {'MEAN':>10}  " + "  ".join(
        f"{(results[k]['fractional_mean_moral'] or float('nan')):>14}" for k in results))
    for k in results:
        print(f"    {k}: closed harmful {results[k]['n_closed_harmful']}, "
              f"harmless {results[k]['n_closed_harmless']}; dtypes {results[k].get('float_dtypes')}")
    print(f"\n  precision parity: {'OK (all same)' if precision_ok else 'MISMATCH — see warning'}")
    print(f"  matched-length (confounded, secondary) survives={sm}; both agree={both_agree}")
    print(f"\nVERDICT [fractional, primary]: survives={sf}\n  {reading}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
