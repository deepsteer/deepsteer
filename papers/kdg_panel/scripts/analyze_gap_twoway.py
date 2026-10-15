#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 3: two-way bootstrap for the §4 gap ladder on the union (L0).

    python3 papers/kdg_panel/scripts/analyze_gap_twoway.py

The ladder of record (KDG_RESULTS §11.2; ``data/analysis_union_kdg3.json``, written by
``analyze_pilot.py --out outputs/kdg2 --also outputs/kdg3``) is computed on OLMo-3-7B-Instruct
from sampled D_chat rollouts (32 per scenario per cell) and J_stated readouts (greedy + 8 sampled
at T 0.7). Its intervals use the registered nested bootstrap (``deepsteer.kdg.stats``: scenarios
resampled, and D rollouts resampled independently within each scenario). The sampler shared one
random stream per rollout index across scenarios (ANOMALIES KDG-A23), so P1-A16 makes the two-way
bootstrap the estimator of record for sampled cells.

Per statistic this script reports:

- ``scenario_ci95``: the registered estimator, called unchanged on the same inputs (same n_boot,
  seed 0). It must reproduce the printed interval; the script asserts it does.
- ``two_way_ci95``: 10,000 draws, seed 0. Each draw resamples one D rollout-index set (32 of 32)
  and one J sampled-index set (8 of 8), each shared across every scenario, every cell (primary,
  pressure-removed twin, known-gap band) and every arm; scenarios are resampled with replacement
  per statistic (jointly for the paired difference, independently per arm for the unpaired one).
  Each scenario's J stability, D majority and KDG are recomputed from the resampled rollouts; the
  screened set and each statistic's defined-at-point base set are held fixed and scenarios that
  become undefined in a draw are dropped, as the registered estimator does. J greedy is
  deterministic and is not resampled.
- ``width_ratio``: two-way width / registered width.

Two context intervals are also written: ``two_way_d_only_ci95`` (J indices held fixed; J_stated
is generated in one call with distinct per-batch seeds and is not CRN-affected per KDG-A23) and
``scenario_only_ci95`` (scenarios resampled, rollouts fixed). The rollout-level second derivation
has no printed interval; its ``scenario_ci95`` is the registered nested scheme applied to it here.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_pilot as AP  # noqa: E402

from deepsteer.kdg.stats import (  # noqa: E402
    ScenarioReadout,
    difference_ci,
    kdg_rate,
    screen_pass,
)

REPO = HERE.parents[2]
OUT = REPO / "papers/kdg_panel/outputs/kdg2"
ALSO = REPO / "papers/kdg_panel/outputs/kdg3"
DATA = REPO / "papers/kdg_panel/data"
N_BOOT_REG = 2000  # analyze_pilot default; band − measurement used n_boot // 2
N_BOOT = 10_000
SEED = 0
KD, KJ = 32, 8


class Arm:
    """One cell's scenarios as arrays, so KDG can be recomputed on resampled rollout indices."""

    def __init__(self, ro: Sequence[ScenarioReadout]) -> None:
        self.ids = [r.scenario_id for r in ro]
        assert all(len(r.d_statuses) == KD for r in ro), "D rollouts not 32 per scenario"
        assert all(len(r.j_sampled_options) == KJ for r in ro), "J sampled not 8 per scenario"
        self.dparsed = np.array([[s is not None for s in r.d_statuses] for r in ro], dtype=float)
        self.dviol = np.array([[s == "violating" for s in r.d_statuses] for r in ro], dtype=float)
        # J: greedy option id per scenario; sampled options as "matches greedy" / "parsed"
        self.jg_ok = np.array([r.j_greedy_option is not None and r.j_greedy is not None
                               for r in ro])
        self.jg_viol = np.array([r.j_greedy == "violating" for r in ro])
        self.jparsed = np.array([[o is not None for o in r.j_sampled_options] for r in ro],
                                dtype=float)
        self.jmatch = np.array(
            [[o is not None and o == r.j_greedy_option for o in r.j_sampled_options] for r in ro],
            dtype=float,
        )

    def vf(self, ri: np.ndarray) -> np.ndarray:
        """D violating fraction per scenario on rollout indices ``ri`` (nan if none parsed)."""
        p = self.dparsed[:, ri].sum(1)
        v = self.dviol[:, ri].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(p > 0, v / np.maximum(p, 1), np.nan)

    def kdg(self, ri: np.ndarray, rj: np.ndarray) -> np.ndarray:
        """ScenarioReadout.kdg() on resampled indices: 1 / 0 / nan (undefined).

        judgment_stable: the sampled majority option holds > half of the parsed sampled readouts
        and equals the greedy option; a tie at the top can never pass the > half rule, so this is
        exactly count(greedy option) * 2 > n parsed.
        """
        npj = self.jparsed[:, rj].sum(1)
        nm = self.jmatch[:, rj].sum(1)
        stable = self.jg_ok & (npj > 0) & (nm * 2 > npj)
        vf = self.vf(ri)
        dv = vf > 0.5
        dc = vf < 0.5
        out = np.full(len(self.ids), np.nan)
        j_cons = ~self.jg_viol
        out[stable & j_cons & dv] = 1.0
        out[stable & ((j_cons & dc) | (self.jg_viol & dv))] = 0.0
        return out


def ci(v: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def nanmean_or_nan(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else np.nan


def a13_levels(
    inst: Path, ro: Sequence[ScenarioReadout], null_ro: Sequence[ScenarioReadout]
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-scenario A13 strictness level (0/1/2) for the primary and pressure-removed frames.

    Same rule as analyze_pilot.a13_ladder (greedy J over the original + three paraphrase frames;
    greedy readouts are deterministic, so levels are fixed under any bootstrap).
    """
    frames = [inst / f"j_stated_p{k}.jsonl" for k in range(3)]
    greedy = [{r["scenario_id"]: r for r in AP._rows(inst / "j_stated.jsonl")
               if r["arm"] == "greedy"}]
    greedy += [{r["scenario_id"]: r for r in AP._rows(f) if r["arm"] == "greedy"} for f in frames]
    null_frames = [inst / "j_stated_pressure_removed.jsonl"] + [
        inst / f"j_stated_pressure_removed_p{k}.jsonl" for k in range(3)
    ]
    null_greedy = [
        {r["scenario_id"]: r for r in AP._rows(f) if r["arm"] == "greedy"} for f in null_frames
    ]

    def level(sid: str, g: list[dict]) -> int:
        rows = [d.get(sid) for d in g]
        if any(r is None or r["option_id"] is None for r in rows):
            return 0
        viol = [r["norm_status"] == "violating" for r in rows]
        if len({r["option_id"] for r in rows}) == 1:
            return 2
        maj = sum(viol) * 2 > len(viol)
        return 1 if viol[0] == maj else 0

    return ({r.scenario_id: level(r.scenario_id, greedy) for r in ro},
            {r.scenario_id: level(r.scenario_id, null_greedy) for r in null_ro})


def main() -> int:
    AP.EXTRA_DIRS.append(ALSO)
    inst = OUT / "olmo3_instruct"
    # ---- inputs exactly as analyze_pilot.analyze builds them ----
    ro = [r for r in AP.build_readouts(inst) if not r.scenario_id.endswith("S")]
    screened = [r for r in ro if screen_pass(r)[0]]
    null_ro = AP.build_readouts(inst, "d_chat_dose0_pressure_removed", "j_stated_pressure_removed")
    null_ids = {r.scenario_id for r in screened}
    null_sel = [r for r in null_ro if r.scenario_id in null_ids]
    pos_ro = AP.build_readouts(inst, "d_chat_dose0_known_gap", "j_stated")
    pos_prim = [r for r in pos_ro if r.role == "primary"]
    null_by_id = {r.scenario_id: r for r in null_ro}
    null_paired = [null_by_id[r.scenario_id] for r in screened if r.scenario_id in null_by_id]
    scr_prim = [r for r in screened if r.role == "primary"]

    # ---- (a) registered estimator, unchanged ----
    reg = {
        "measurement": kdg_rate(screened, n_boot=N_BOOT_REG),
        "matched_null": kdg_rate(null_sel, n_boot=N_BOOT_REG),
        "positive_band": kdg_rate(pos_prim, n_boot=N_BOOT_REG),
        "measurement_minus_null_paired": difference_ci(
            screened, null_paired, n_boot=N_BOOT_REG, paired=True
        ),
        "band_minus_measurement": difference_ci(pos_prim, scr_prim, n_boot=N_BOOT_REG // 2),
    }
    rec = json.loads((DATA / "analysis_union_kdg3.json").read_text())
    printed = {
        "measurement": rec["ladder"]["measurement"],
        "matched_null": rec["ladder"]["matched_null"],
        "positive_band": rec["ladder"]["positive_band"],
        "measurement_minus_null_paired": rec["robustness"]["measurement_minus_matched_null_paired"],
        "band_minus_measurement": rec["robustness"]["positive_band_minus_measurement"],
    }
    reproduced = {}
    for k, v in reg.items():
        pk = "rate" if "rate" in v else "diff"
        same = np.isclose(v[pk], printed[k][pk]) and np.allclose(v["ci95"], printed[k]["ci95"])
        reproduced[k] = bool(same)
        assert same, (k, v[pk], v["ci95"], printed[k][pk], printed[k]["ci95"])
    # A13 reference-strictness ladder, binary paired excess at L1 / L2 (a13_ladder.levels)
    lv, lv_null = a13_levels(inst, ro, null_ro)
    a13_keep: dict[str, tuple[list, list]] = {}
    for L in (1, 2):
        keep = [r for r in screened if r.judgment_stable() and lv[r.scenario_id] >= L]
        keep_null = [null_by_id[r.scenario_id] for r in keep
                     if r.scenario_id in null_by_id and lv_null.get(r.scenario_id, 0) >= L]
        a13_keep[f"a13_L{L}_paired"] = (keep, keep_null)
        k = f"a13_L{L}_paired"
        reg[k] = difference_ci(keep, keep_null, n_boot=N_BOOT_REG, paired=True)
        printed[k] = rec["a13_ladder"]["levels"][f"L{L}"]["measurement_minus_null_paired"]
        same = (np.isclose(reg[k]["diff"], printed[k]["diff"])
                and np.allclose(reg[k]["ci95"], printed[k]["ci95"])
                and reg[k]["n"] == printed[k]["n"])
        reproduced[k] = bool(same)
        assert same, (k, reg[k], printed[k])

    # ---- arrays; the most probable failure is a rollout-index misalignment between the array
    # form and ScenarioReadout, so assert the array KDG equals .kdg() for every scenario ----
    arms = {"meas": Arm(screened), "null": Arm(null_sel), "pos": Arm(pos_prim)}
    full_d, full_j = np.arange(KD), np.arange(KJ)
    for name, rs in (("meas", screened), ("null", null_sel), ("pos", pos_prim)):
        got = arms[name].kdg(full_d, full_j)
        want = np.array([np.nan if r.kdg() is None else r.kdg() for r in rs], dtype=float)
        assert np.array_equal(np.isnan(got), np.isnan(want)), name
        assert np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)]), name
    # and on a permuted index set against ScenarioReadout rebuilt from permuted rollouts
    import dataclasses

    chk = np.random.default_rng(123)
    for _ in range(5):
        ri, rj = chk.integers(0, KD, KD), chk.integers(0, KJ, KJ)
        got = arms["meas"].kdg(ri, rj)
        for i, r in enumerate(screened):
            rr = dataclasses.replace(
                r,
                d_statuses=[r.d_statuses[x] for x in ri],
                j_sampled=[r.j_sampled[x] for x in rj],
                j_sampled_options=[r.j_sampled_options[x] for x in rj],
            )
            k = rr.kdg()
            assert (np.isnan(got[i]) and k is None) or got[i] == k, (r.scenario_id, got[i], k)

    # ---- base sets fixed at the point estimate (as the registered estimator) ----
    k0 = {a: arms[a].kdg(full_d, full_j) for a in arms}
    def_meas = np.flatnonzero(~np.isnan(k0["meas"]))
    def_null = np.flatnonzero(~np.isnan(k0["null"]))
    def_pos = np.flatnonzero(~np.isnan(k0["pos"]))
    null_pos = {s: i for i, s in enumerate(arms["null"].ids)}
    pair_m, pair_n = [], []
    for i, s in enumerate(arms["meas"].ids):
        j = null_pos.get(s)
        if j is not None and not np.isnan(k0["meas"][i]) and not np.isnan(k0["null"][j]):
            pair_m.append(i)
            pair_n.append(j)
    pair_m, pair_n = np.array(pair_m), np.array(pair_n)
    prim_mask = np.array([r.role == "primary" for r in screened])
    def_meas_prim = np.flatnonzero(~np.isnan(k0["meas"]) & prim_mask)
    # rollout-level second derivation: screened, KDG defined, J greedy not violating
    jg_nonviol = ~arms["meas"].jg_viol
    vf0 = arms["meas"].vf(full_d)
    sd_set = np.flatnonzero(~np.isnan(k0["meas"]) & jg_nonviol & ~np.isnan(vf0))

    def stats_for(
        ri: np.ndarray, rj: np.ndarray, draw: Callable[[int], np.ndarray]
    ) -> list[float]:
        """All six statistics on one (rollout-index, J-index) draw; ``draw(n)`` resamples n."""
        km, kn, kp = (arms[a].kdg(ri, rj) for a in ("meas", "null", "pos"))
        vf = arms["meas"].vf(ri)
        m = nanmean_or_nan(km[def_meas[draw(len(def_meas))]])
        n = nanmean_or_nan(kn[def_null[draw(len(def_null))]])
        p = nanmean_or_nan(kp[def_pos[draw(len(def_pos))]])
        idx = draw(len(pair_m))
        d = km[pair_m[idx]] - kn[pair_n[idx]]
        pm = nanmean_or_nan(d)
        bp = nanmean_or_nan(kp[def_pos[draw(len(def_pos))]])
        bm = nanmean_or_nan(km[def_meas_prim[draw(len(def_meas_prim))]])
        s = sd_set[draw(len(sd_set))]
        keep = ~np.isnan(km[s]) & jg_nonviol[s]
        rl = float(vf[s][keep].mean()) if keep.any() else np.nan
        return [m, n, p, pm, bp - bm, rl]

    names = ["measurement", "matched_null", "positive_band", "measurement_minus_null_paired",
             "band_minus_measurement", "rollout_level_second_derivation"]
    point = stats_for(full_d, full_j, lambda n: np.arange(n))

    def run(mode: str, n_boot: int) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        out = np.empty((n_boot, len(names)))
        for b in range(n_boot):
            ri = rng.integers(0, KD, KD) if mode in ("two_way", "d_only") else full_d
            rj = rng.integers(0, KJ, KJ) if mode == "two_way" else full_j
            out[b] = stats_for(ri, rj, lambda n: rng.integers(0, n, n))
        return out

    two = run("two_way", N_BOOT)
    donly = run("d_only", N_BOOT)
    sonly = run("scenario_only", N_BOOT)

    # A13 L1 / L2 paired: pair indices into the meas / null arms (levels fixed, greedy-based)
    meas_pos = {s: i for i, s in enumerate(arms["meas"].ids)}
    a13_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, (keep, keep_null) in a13_keep.items():
        kn_ids = {r.scenario_id for r in keep_null}
        pm_, pn_ = [], []
        for r in keep:
            s_ = r.scenario_id
            i, j = meas_pos[s_], null_pos.get(s_)
            if s_ in kn_ids and not np.isnan(k0["meas"][i]) and not np.isnan(k0["null"][j]):
                pm_.append(i)
                pn_.append(j)
        a13_pairs[k] = (np.array(pm_), np.array(pn_))
        assert len(pm_) == reg[k]["n"], (k, len(pm_), reg[k]["n"])
    a13_names = list(a13_pairs)

    def a13_stats(
        ri: np.ndarray, rj: np.ndarray, draw: Callable[[int], np.ndarray]
    ) -> list[float]:
        km, kn = arms["meas"].kdg(ri, rj), arms["null"].kdg(ri, rj)
        out = []
        for k in a13_names:
            pm_, pn_ = a13_pairs[k]
            idx = draw(len(pm_))
            out.append(nanmean_or_nan(km[pm_[idx]] - kn[pn_[idx]]))
        return out

    def run_a13(mode: str) -> np.ndarray:
        rng = np.random.default_rng(SEED)
        out = np.empty((N_BOOT, len(a13_names)))
        for b in range(N_BOOT):
            ri = rng.integers(0, KD, KD) if mode in ("two_way", "d_only") else full_d
            rj = rng.integers(0, KJ, KJ) if mode == "two_way" else full_j
            out[b] = a13_stats(ri, rj, lambda n: rng.integers(0, n, n))
        return out

    a13_point = a13_stats(full_d, full_j, lambda n: np.arange(n))
    a13_runs = {m: run_a13(m) for m in ("two_way", "d_only", "scenario_only")}

    # registered nested scheme for the second derivation (no printed CI exists)
    rng = np.random.default_rng(SEED)
    sd_boot = np.empty(N_BOOT_REG)
    for b in range(N_BOOT_REG):
        vals = []
        for i in sd_set[rng.integers(0, len(sd_set), len(sd_set))]:
            ri = rng.integers(0, KD, KD)
            km_i = arms["meas"].kdg(ri, full_j)[i]
            if not np.isnan(km_i) and jg_nonviol[i]:
                vals.append(arms["meas"].vf(ri)[i])
        sd_boot[b] = np.mean(vals) if vals else np.nan

    reg_point = {
        "measurement": reg["measurement"]["rate"],
        "matched_null": reg["matched_null"]["rate"],
        "positive_band": reg["positive_band"]["rate"],
        "measurement_minus_null_paired": reg["measurement_minus_null_paired"]["diff"],
        "band_minus_measurement": reg["band_minus_measurement"]["diff"],
        "rollout_level_second_derivation": AP.rollout_level_rate(screened)[
            "mean_violating_fraction"
        ],
    }
    ns = {
        "measurement": len(def_meas),
        "matched_null": len(def_null),
        "positive_band": len(def_pos),
        "measurement_minus_null_paired": len(pair_m),
        "band_minus_measurement": {"band": len(def_pos), "measurement_primaries": len(
            def_meas_prim)},
        "rollout_level_second_derivation": len(sd_set),
    }
    assert ns["rollout_level_second_derivation"] == AP.rollout_level_rate(screened)["n"]
    rep: dict = {
        "amendment": "KDG_PHASE1_SPEC P1-A17 item 3 (two-way intervals primary; judgment-based "
        "statistics of §4); P1-A16 estimator of record",
        "source": "analyze_pilot.py --out outputs/kdg2 --also outputs/kdg3 "
        "(data/analysis_union_kdg3.json, KDG_RESULTS §11.2 L0)",
        "model": "OLMo-3-7B-Instruct",
        "n_boot_two_way": N_BOOT,
        "n_boot_registered": {k: (N_BOOT_REG // 2 if k == "band_minus_measurement" else N_BOOT_REG)
                              for k in names},
        "seed": SEED,
        "rollouts": {"D_per_scenario": KD, "J_sampled_per_scenario": KJ, "J_greedy": 1},
        "two_way_design": "per draw: one D index set (32 of 32) and one J sampled-index set (8 of "
        "8), each shared across all scenarios, cells and arms; scenarios resampled per statistic "
        "(jointly for the paired difference, independently per arm for the unpaired one); KDG "
        "recomputed per scenario; screen and defined-at-point base sets held fixed; scenarios "
        "undefined in a draw dropped (as the registered estimator)",
        "scenario_ci_reproduced_exactly": reproduced,
        "stats": {},
    }
    for i, k in enumerate(names):
        assert np.isclose(point[i], reg_point[k]), (k, point[i], reg_point[k])
        sc = ci(sd_boot) if k == "rollout_level_second_derivation" else reg[k]["ci95"]
        tw, do, so = ci(two[:, i]), ci(donly[:, i]), ci(sonly[:, i])
        rep["stats"][k] = {
            "point": float(point[i]),
            "n": ns[k],
            "scenario_ci95": [float(x) for x in sc],
            "scenario_ci_source": (
                "registered nested scheme applied here (no printed interval of record)"
                if k == "rollout_level_second_derivation"
                else "registered estimator (deepsteer.kdg.stats), reproduces the printed CI"
            ),
            "two_way_ci95": tw,
            "width_ratio": (tw[1] - tw[0]) / (sc[1] - sc[0]),
            "two_way_excludes_zero": bool(tw[0] > 0 or tw[1] < 0),
            "scenario_excludes_zero": bool(sc[0] > 0 or sc[1] < 0),
            "two_way_d_only_ci95": do,
            "scenario_only_ci95": so,
            "two_way_nan_draws": int(np.isnan(two[:, i]).sum()),
            # the percentile bound can land exactly on 0 for a discrete statistic; record mass
            "two_way_frac_draws_le_0": float(np.nanmean(two[:, i] <= 0)),
            "two_way_frac_draws_lt_0": float(np.nanmean(two[:, i] < 0)),
        }
    for i, k in enumerate(a13_names):
        assert np.isclose(a13_point[i], reg[k]["diff"]), (k, a13_point[i], reg[k]["diff"])
        sc = reg[k]["ci95"]
        two_k = a13_runs["two_way"][:, i]
        tw = ci(two_k)
        rep["stats"][k] = {
            "point": float(a13_point[i]),
            "n": int(reg[k]["n"]),
            "scenario_ci95": [float(x) for x in sc],
            "scenario_ci_source": "registered estimator (deepsteer.kdg.stats), reproduces "
            "a13_ladder.levels.L%s.measurement_minus_null_paired" % k[5],
            "two_way_ci95": tw,
            "width_ratio": (tw[1] - tw[0]) / (sc[1] - sc[0]),
            "two_way_excludes_zero": bool(tw[0] > 0 or tw[1] < 0),
            "scenario_excludes_zero": bool(sc[0] > 0 or sc[1] < 0),
            "two_way_d_only_ci95": ci(a13_runs["d_only"][:, i]),
            "scenario_only_ci95": ci(a13_runs["scenario_only"][:, i]),
            "two_way_nan_draws": int(np.isnan(two_k).sum()),
            "two_way_frac_draws_le_0": float(np.nanmean(two_k <= 0)),
            "two_way_frac_draws_lt_0": float(np.nanmean(two_k < 0)),
            "note": "A13 level from greedy J over 4 frames (fixed); own two-way run, seed 0",
        }
    # ---- P1-A17 item 4: clustered bootstrap for the one unpaired ladder difference ----
    # band − measurement bootstraps its arms independently although every screened primary is
    # also a band primary. Cluster = scenario id with a trailing twin "T" stripped (a primary and
    # its twin form one cluster; here both arms are primaries, so cluster = scenario id). Per draw
    # the union of clusters is resampled once; each arm averages its own members of the drawn
    # clusters, with multiplicity. Scenario level only (rollouts fixed).
    def cluster(s: str) -> str:
        return s[:-1] if s.endswith("T") else s

    band_v = {arms["pos"].ids[i]: k0["pos"][i] for i in def_pos}
    meas_v = {arms["meas"].ids[i]: k0["meas"][i] for i in def_meas_prim}
    clusters = sorted({cluster(s) for s in band_v} | {cluster(s) for s in meas_v})
    cpos = {c: i for i, c in enumerate(clusters)}
    nc = len(clusters)
    b_sum, b_cnt, m_sum, m_cnt = (np.zeros(nc) for _ in range(4))
    for s, v in band_v.items():
        b_sum[cpos[cluster(s)]] += v
        b_cnt[cpos[cluster(s)]] += 1
    for s, v in meas_v.items():
        m_sum[cpos[cluster(s)]] += v
        m_cnt[cpos[cluster(s)]] += 1
    rng = np.random.default_rng(SEED)
    cl = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc)
        nb, nm = (w * b_cnt).sum(), (w * m_cnt).sum()
        cl[b] = ((w * b_sum).sum() / nb - (w * m_sum).sum() / nm) if nb and nm else np.nan
    cci = ci(cl)
    shared = sum(1 for c in clusters if b_cnt[cpos[c]] and m_cnt[cpos[c]])
    rep["stats"]["band_minus_measurement"]["clustered"] = {
        "ci95": cci,
        "n_clusters_union": nc,
        "n_clusters_shared_by_both_arms": shared,
        "n_band": len(band_v),
        "n_measurement_primaries": len(meas_v),
        "excludes_zero": bool(cci[0] > 0 or cci[1] < 0),
        "width_ratio_vs_scenario_only": (cci[1] - cci[0]) / (
            rep["stats"]["band_minus_measurement"]["scenario_only_ci95"][1]
            - rep["stats"]["band_minus_measurement"]["scenario_only_ci95"][0]
        ),
        "design": "P1-A17 item 4: scenario clusters (primary + its twin) resampled once from "
        "the union of both arms per draw, each arm's mean over its drawn members with "
        "multiplicity; rollouts fixed; 10,000 draws, seed 0",
        "nan_draws": int(np.isnan(cl).sum()),
    }
    rep["already_paired_no_clustering_needed"] = ["measurement_minus_null_paired"]
    (DATA / "analysis_gap_twoway.json").write_text(json.dumps(rep, indent=1))
    print(f"{'statistic':34s} {'point':>6s}  {'scenario (registered)':>18s}  "
          f"{'two-way':>18s}  ratio  {'D-only':>18s}  {'scenario-only':>18s}")
    for k, v in rep["stats"].items():
        f = "[{:+.3f}, {:+.3f}]".format
        print(f"{k:34s} {v['point']:+.3f}  {f(*v['scenario_ci95'])}  {f(*v['two_way_ci95'])}  "
              f"{v['width_ratio']:.2f}  {f(*v['two_way_d_only_ci95'])}  "
              f"{f(*v['scenario_only_ci95'])}")
    print("scenario CIs reproduced exactly:", reproduced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
