#!/usr/bin/env python3
"""Local gate for Phase B session 1 (B1/B3/B5): exercises every model-INDEPENDENT code path
on synthetic + committed data, no GPU / no model load. The tiny-model smoke (VALIDATE=1) runs
on the pod. Run: python papers/d2_decision_coupling/scripts/b_session1_local_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import b1_judgment_direction as b1  # noqa: E402
import b3_batched_extractions as b3  # noqa: E402
import b5_moral_fragility as b5  # noqa: E402
import informat_ladder as il  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def test_b1_parsing() -> None:
    print("B1 parsing / prompts:")
    check("verdict wrong", b1.parse_verdict("wrong, because ...") == "wrong")
    check("verdict acceptable->not_wrong", b1.parse_verdict("Acceptable.") == "not_wrong")
    check("verdict 'not wrong'", b1.parse_verdict("not wrong at all") == "not_wrong")
    check("verdict other", b1.parse_verdict("It depends on ...") == "other")
    p0, ord0 = b1.build_prompt("X did Y.", flip=False)
    p1, ord1 = b1.build_prompt("X did Y.", flip=True)
    check("counterbalance flips option order", ord0 != ord1 and "X did Y." in p0)
    check("refusal marker detected", b1.is_refusal("I'm sorry, I can't help with that."))
    check("non-refusal not flagged", not b1.is_refusal("Sure, here is how you do it:"))


def test_b1_items() -> None:
    print("B1 ethics items (committed data):")
    f = HERE.parents[1] / "d1_moral_subspace" / "outputs" / "full" / "ethics_train_full.json"
    items = b1.load_items([f], None)
    gts = [g for _, g in items]
    nw, nnw = gts.count("wrong"), gts.count("not_wrong")
    check("items loaded", len(items) > 100, f"n={len(items)}")
    check("label-balanced (moral=wrong, neutral=not_wrong)", nw == nnw, f"{nw}/{nnw}")
    check("scenarios are text", all(isinstance(s, str) and s for s, _ in items[:5]))


def test_b1_direction_math() -> None:
    print("B1 within-label contrast (synthetic):")
    rng = np.random.default_rng(0)
    d = 64
    n = 200
    # synthetic: a true judgment axis u; model-says-wrong shifts +u, says-not-wrong -u, within
    # each ground-truth label; plus a content axis c separating gt.
    u = b1._unit(rng.standard_normal(d)); c = b1._unit(rng.standard_normal(d))
    acts, gt, verdict = [], [], []
    for i in range(n):
        g = "wrong" if i % 2 else "not_wrong"
        v = "wrong" if rng.random() < (0.7 if g == "wrong" else 0.3) else "not_wrong"
        base = (1.0 if g == "wrong" else -1.0) * c + (1.0 if v == "wrong" else -1.0) * u
        acts.append(base + 0.2 * rng.standard_normal(d)); gt.append(g); verdict.append(v)
    acts = np.stack(acts); gt = np.array(gt); verdict = np.array(verdict)
    jd = b1.within_label_direction(acts, gt, verdict)
    check("within-label method used", jd["method"] == "within_label_avg", str(jd["cells"]))
    check("direction is unit", abs(np.linalg.norm(jd["direction"]) - 1.0) < 1e-6)
    check("recovers judgment axis u (|cos|>0.8)", abs(float(jd["direction"] @ u)) > 0.8,
          f"|cos|={abs(float(jd['direction'] @ u)):.3f}")
    check("label-contrast recovers content axis c (|cos|>0.8)",
          abs(float(jd["label_contrast"] @ c)) > 0.8)
    # fallback path: starve one cell
    v2 = verdict.copy(); v2[(gt == "wrong")] = "wrong"  # no 'not_wrong' verdicts among wrong
    jd2 = b1.within_label_direction(acts, gt, v2)
    check("fallback to pooled when cell < MIN_CELL", jd2["used_fallback"] is True)


def test_b1_geometry_helpers() -> None:
    print("B1 geometry helpers (synthetic):")
    rng = np.random.default_rng(1)
    d = 128
    dirs = [b1._unit(rng.standard_normal(d)) for _ in range(3)]
    Q = b1._ortho(dirs)
    check("ortho basis shape (d,3)", Q.shape == (d, 3))
    check("ortho columns orthonormal", np.allclose(Q.T @ Q, np.eye(3), atol=1e-6))
    inside = b1._unit(Q @ rng.standard_normal(3))
    check("in-span frac ~1", abs(b1._frac(Q, inside) - 1.0) < 1e-6)
    X = rng.standard_normal((80, d))
    q95 = b1.pairwise_null_q95(X - X.mean(0), rng, k=300)
    check("pairwise null q95 in (0,1)", 0 < q95 < 1, f"q95={q95:.3f}")
    # load_vmoral_basis from a source-dir npz
    tmp = HERE.parent / "outputs" / "_lt_vmoral.npz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, **{f"{s}_layer16": dirs[i] for i, s in
                     enumerate(("moral_stories", "fables", "ethics"))})
    Qv = b1.load_vmoral_basis(tmp, 16)
    check("load_vmoral_basis -> (d,3)", Qv.shape == (d, 3))
    tmp.unlink()


def test_b3() -> None:
    print("B3 control sets + fable-schema:")
    cp = b3.control_pairs()
    check("control sets present", set(cp) == {"syntax", "register", "sentiment", "fable_schema"})
    check("syntax 210 / register 80 / sentiment 210",
          len(cp["syntax"]) == 210 and len(cp["register"]) == 80 and len(cp["sentiment"]) == 210)
    fs = b3.FABLE_SCHEMA_PAIRS
    check("fable-schema 20 pairs", len(fs) == 20)
    check("fable-schema are 2-tuples of text", all(len(p) == 2 and p[0] and p[1] for p in fs))
    moral_toks = ("wrong", "immoral", "steal", "lie ", "cheat", "betray", "cruel", "evil")
    hits = [p for p in fs if any(t in (p[0] + p[1]).lower() for t in moral_toks)]
    check("fable-schema is amoral (no moral tokens)", len(hits) == 0, f"hits={len(hits)}")
    # rotation-angle formula sanity (used by run_rotate)
    ang = float(np.degrees(np.arccos(np.clip(abs(0.766), -1, 1))))  # cos40deg~0.766
    check("rotation angle ~40deg for cos 0.766", abs(ang - 40.0) < 1.0, f"{ang:.1f}")


def test_b5() -> None:
    print("B5 fragility logic (synthetic):")
    rng = np.random.default_rng(0)
    d = 96
    X = rng.standard_normal((120, d)) * 3.0 + 1.0
    check("rms positive ~ sqrt(mean sq)", abs(b5.rms(X) - np.sqrt((X ** 2).mean())) < 1e-9)
    Xc = X - X.mean(0)
    Q = b5.covmatched_subspace(Xc, 3, rng)
    check("covmatched subspace (d,3) orthonormal",
          Q.shape == (d, 3) and np.allclose(Q.T @ Q, np.eye(3), atol=1e-6))
    v = b5.in_subspace_dir(Q, rng)
    check("in-subspace dir is unit + lies in span(Q)",
          abs(np.linalg.norm(v) - 1) < 1e-6 and abs(np.linalg.norm(Q.T @ v) - 1) < 1e-6)
    grid = [0.0, 1.0, 2.0, 3.0, 5.0]
    # decreasing refusal curve crossing 0.5*baseline (=0.4) between sigma 2 and 3
    rates = [0.8, 0.7, 0.5, 0.3, 0.1]
    ss = b5.sigma_star(grid, rates, baseline=0.8, ratio=0.5)
    check("sigma_star interpolates crossing (2,3)", 2.0 < ss < 3.0, f"sigma*={ss:.2f}")
    ss_never = b5.sigma_star(grid, [0.8, 0.8, 0.8, 0.8, 0.8], baseline=0.8)
    check("sigma_star sentinel when never crossed", ss_never > max(grid), f"{ss_never}")
    v_below = b5.r8_verdict(1.0, [3.0, 4.0, 5.0, 6.0, 3.5])
    check("R8 below-floor -> differential sensitivity", v_below["differential_sensitivity_below_floor"])
    v_within = b5.r8_verdict(5.0, [3.0, 4.0, 5.0, 6.0, 4.5])
    check("R8 within-floor -> flat baseline", not v_within["differential_sensitivity_below_floor"])
    check("coherence flags repetition", not b5.is_coherent("na na na na na na na na na na"))
    check("coherence accepts normal text", b5.is_coherent("Here is a clear and varied answer."))


def test_informat_ladder() -> None:
    print("In-format ladder (position span + format guard + raw reproduction):")
    # (1) content-span finder: content block located inside a templated id sequence.
    full = [1, 100, 101, 7, 8, 9, 200, 201]  # BOS + prefix + [7,8,9]=content + assistant tail
    s, e = il.find_content_span(full, [7, 8, 9])
    check("find_content_span locates content block", (s, e) == (3, 6), f"got {(s, e)}")
    s2, e2 = il.find_content_span([1, 2, 3], [9, 9])  # absent -> trailing fallback, no crash
    check("find_content_span falls back gracefully when absent", e2 == 3 and s2 == 2)

    # (2) HARD GUARD: the chat null must consume a format/position-matched act_sample.
    d = 64
    rng = np.random.default_rng(0)
    srcs = {s: il._unit(rng.standard_normal(d)) for s in il.SRC}
    ctrls = {c: il._unit(rng.standard_normal(d)) for c in il.CONTROLS}
    persona = il._unit(rng.standard_normal(d))
    good = {"X": rng.standard_normal((50, d)), "tag": ("chat", "final_pre_assistant")}
    bad = {"X": rng.standard_normal((50, d)), "tag": ("raw", "pooled")}
    try:
        il.ladder(srcs, ctrls, persona, good, ("chat", "final_pre_assistant"), standardize=True,
                  rng=np.random.default_rng(0))
        ok_match = True
    except ValueError:
        ok_match = False
    check("ladder accepts format-matched null", ok_match)
    raised = False
    try:
        il.ladder(srcs, ctrls, persona, bad, ("chat", "final_pre_assistant"),
                  rng=np.random.default_rng(0))
    except ValueError:
        raised = True
    check("ladder RAISES on chat-dirs x raw-null mismatch (the invariant)", raised)

    # (3) raw reproduction: il.ladder(standardize=False) on the committed OLMo raw artifacts must
    #     reproduce standardized_recompute's R5_raw band + controls (same ladder math).
    outp = HERE.parent / "outputs"
    sr = outp / "standardized_recompute.json"
    vm = outp / "olmo3" / "vmoral_sources.npz"
    if sr.exists() and vm.exists():
        L = 16
        z = np.load(vm)
        sources = {s: z[f"{s}_layer{L}"].astype(float) for s in il.SRC}
        controls = {c: np.load(outp / "olmo3" / f"b3_{c}_dir_olmo3.npz")[f"{c}_layer{L}"].astype(float)
                    for c in il.CONTROLS}
        persona_v = np.load(outp / "olmo3" / "persona_direction.npz")[f"persona_layer{L}"].astype(float)
        refusal_v = np.load(outp / "olmo3" / "refusal.npz")["refusal"].astype(float)
        X = np.load(outp / "olmo3" / "act_sample.npz")["X"].astype(float)
        raw = il.ladder(sources, controls, persona_v, {"X": X, "tag": ("raw", "pooled")},
                        ("raw", "pooled"), refusal=refusal_v, standardize=False,
                        rng=np.random.default_rng(0))
        ref = json.loads(sr.read_text())["olmo3"]["R5_raw"]
        band_ok = raw["moral_family_band"] == ref["moral_family_band"]
        ctrl_ok = all(abs(raw["c_controls"][c] - ref["c_controls"][c]) < 1e-6 for c in il.CONTROLS)
        refp_ok = abs(raw["G3_refusal_p"] - ref["refusal_p"]) < 1e-6
        check("raw ladder reproduces committed R5_raw band", band_ok,
              f"{raw['moral_family_band']} vs {ref['moral_family_band']}")
        check("raw ladder reproduces committed R5_raw controls + refusal_p", ctrl_ok and refp_ok)
    else:
        print("  [skip] raw-reproduction (chunk-1 artifacts not present locally)")


def main() -> None:
    print("=== Phase B session 1 local gate ===\n")
    test_b1_parsing()
    test_b1_items()
    test_b1_direction_math()
    test_b1_geometry_helpers()
    test_b3()
    test_b5()
    test_informat_ladder()
    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("ALL LOCAL CHECKS PASSED (B1). Tiny-model smoke runs on the pod via VALIDATE=1.")


if __name__ == "__main__":
    main()
