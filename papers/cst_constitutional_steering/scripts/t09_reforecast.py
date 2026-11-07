#!/usr/bin/env python3
"""CST line re-forecast at the second VALIDATE's measured rates (Amendment 8 item 5).

    t09_reforecast.py        # writes outputs/t09_reforecast.json and prints the tables

Every unit price is tagged ``measured`` (outputs/validate), ``carried`` (another line's measured
rate, not re-measured on this model) or ``estimate`` (arithmetic from a measured rate and a stated
assumption). Two engine cases: ``current`` (ST's vLLM configuration as run) and ``engine_fixed``
(CST-A11's R_engine: generation with the document in context at A0's per-token rate). The cut list
is applied cumulatively in rank order to the partner-recipe path under each case. Scoring follows
Amendment 9 (gpt-oss-120b alone, Olmo-3.1-32B-Instruct on 10%); the frontier tier's Codex calls
are counted beside the A100-h (`codex_calls`), since the weekly quota, not the GPU, bounds them.
"""

from __future__ import annotations

import json
from pathlib import Path

CST = Path(__file__).resolve().parents[1]
V = CST / "outputs" / "validate"


def units() -> dict:
    g = json.loads((V / "gen.json").read_text())
    j = json.loads((V / "judges.json").read_text())
    lh = json.loads((V / "lora_hf.json").read_text())
    pr = json.loads((V / "price.json").read_text())
    tok_s = g["panel"]["gen_tokens"] / g["panel"]["seconds"]  # A0, no document prefix
    tok_s_doc = g["a0s"]["gen_tokens"] / g["a0s"]["seconds"]  # the document in context
    per_arm = pr["rollouts_per_arm"]
    a0_len = g["panel"]["gen_tokens"] / g["panel"]["n"]
    a0s_len = g["a0s"]["gen_tokens"] / g["a0s"]["n"]
    gpt = per_arm / j["gpt_oss_120b"]["calls_per_hour"]
    olmo = per_arm / j["olmo31_32b_instruct"]["calls_per_hour"]
    return {
        "per_arm_rollouts": per_arm,
        "tok_s_no_prefix": tok_s, "tok_s_doc_prefix": tok_s_doc,
        "a0_mean_gen_tokens": a0_len, "a0s_mean_gen_tokens": a0s_len,
        "sdf_tok_s": lh["sdf_tokens_per_s"], "sdf_10M_h": lh["sdf_10M_hours"],
        "judge_h_per_arm": {"gpt_oss_120b": gpt, "olmo31_32b_instruct": olmo,
                            # unmeasured: Olmo-3.1-32B-Instruct's rate (dense, short outputs),
                            # range from gpt-oss's to half of Olmo's
                            "mistral_small_32_est": olmo},
        "s0a_measured": pr,
    }


def forecast() -> dict:
    u = units()
    j = u["judge_h_per_arm"]
    # Amendment 9: gpt-oss-120b scores alone; Olmo-3.1-32B-Instruct reads a 10% sample. The
    # frontier tier (account logins, zero GPU) filters DPO pairs; gpt-oss breaks its splits only
    judges = j["gpt_oss_120b"] + 0.1 * j["olmo31_32b_instruct"]

    def gen_h(n_roll: int, mean_tok: float, doc: bool, fixed: bool) -> float:
        rate = u["tok_s_no_prefix"] if (not doc or fixed) else u["tok_s_doc_prefix"]
        return n_roll * mean_tok / rate / 3600

    def arm_read(fixed: bool, mean_tok: float = 600.0) -> float:
        # a trained arm's full read: generation at cap 4,096 (mean tokens 600: the toy SDF
        # adapter's +39% median on A0's 428), the scorer of record and Olmo's 10%, capability
        # 800 (carried 1.2), ST's dose-0 reads 0.1, N (estimate 1.0)
        return gen_h(u["per_arm_rollouts"], mean_tok, False, fixed) + judges + 1.2 + 0.1 + 1.0

    bars = 1.5 + 1.75 + 0.3  # finalist: MRB 500 k2 (carried 1-2), MCH k2 (carried), refusal

    def dpo_h(pairs: int, epochs: int, seq: int = 1100) -> float:
        # chosen and rejected through the policy, the reference forward at about a fifth of that
        return pairs * epochs * 2 * seq * 1.2 / u["sdf_tok_s"] / 3600

    def s3_data(pairs: int, fixed: bool) -> float:
        cand = pairs / 0.7  # the 70% pass rule at the 200-pair pilot
        first = gen_h(int(cand), 800, False, fixed)
        rewrite = gen_h(int(cand), 1000, True, fixed)
        score = 0.2 * cand / u["per_arm_rollouts"] * j["gpt_oss_120b"]  # tiebreaks
        return first + rewrite + score

    out = {}
    for case, fixed in (("current", False), ("engine_fixed", True)):
        pr = u["s0a_measured"]
        a0s_gen = (pr["subject_hours"]["a0s_generation"] if not fixed else
                   gen_h(u["per_arm_rollouts"], u["a0s_mean_gen_tokens"], True, True))
        s0a = (pr["s0a_base_hours"] - pr["subject_hours"]["a0s_generation"] + a0s_gen
               + 2 * judges + 87 * 13.6 / 3600 + 0.1)  # gpt-oss long-prompt calls; retest
        s0a *= 1.1
        sess = {
            "probe_pod (engine tuning, the letter-mapped scorer on gpt-oss and Olmo)": 1.5,
            "S0a": s0a,
            "S0b (A0s riders, N on A0 and A0s, A0s length-matched fork)":
                0.3 + 1.5 + 1.75 + 2 * 1.0 + gen_h(586 * 4, u["a0s_mean_gen_tokens"], True, fixed)
                + 0.4,
            "S1 (corpora, A1 SDF, A1c SFT, recovery)":
                gen_h(20_000_000, 1, True, fixed)  # 10M tokens plus the self-critique pass
                + gen_h(10_000_000, 1, False, fixed) + 2 * u["sdf_10M_h"] + 0.5 + 0.1 + 0.4,
            "S2 (A1, A1c reads; A1 without recovery, format read; winner's bars)":
                2 * arm_read(fixed) + 0.3 + bars + 0.4,
            "S3 (A2 two draws, A2 without rewrite, A3; reads and bars)":
                2 * (s3_data(3000, fixed) + dpo_h(3000, 3)) + (gen_h(4286, 800, False, fixed)
                + 0.2 * 3000 / 0.7 / u["per_arm_rollouts"] * j["gpt_oss_120b"] + dpo_h(3000, 3))
                + (gen_h(4000, 1500, False, fixed) + 6e6 / u["sdf_tok_s"] / 3600)
                + 4 * arm_read(fixed) + 4 * 0.3 + 2 * bars + 0.8,
            "S4 (B1 on the sibling, two starting points; estimate)": 29.0,
            "S5 (A4, A4f; gated by R3; estimate)": 2 * 15 + 2 * arm_read(fixed) + 2 * bars,
        }
        recipe = sum(v for k, v in sess.items() if k[:2] in ("pr", "S0", "S1", "S2", "S3"))
        out[case] = {"sessions": sess, "total": sum(sess.values()), "partner_recipe_S0_S3": recipe}

    def cut_list(fx: bool) -> list:
        """The ranked cuts, applied cumulatively to the partner-recipe path (S0-S3)."""
        cuts = [
            ("DPO at 1 epoch, not 3 (inside the registered cap of <= 3; no amendment)",
             3 * (dpo_h(3000, 3) - dpo_h(3000, 1)), "dose; ST-A38 moved protection with pair "
             "count, not epochs"),
            ("A2 without the rewrite: defer (it needs its own DPO run, so it is not zero extra "
             "training as §5 states)",
             gen_h(4286, 800, False, fx)
             + 0.2 * 3000 / 0.7 / u["per_arm_rollouts"] * j["gpt_oss_120b"]
             + dpo_h(3000, 1) + arm_read(fx) + 0.3,
             "the lab's rewrite ablation replication; secondary to the recipe"),
            ("A0s on the operator condition only (panel k 4, H1 operator k 8)",
             (u["per_arm_rollouts"] - (586 * 4 + 132 * 8)) * (
                 gen_h(1, u["a0s_mean_gen_tokens"], True, fx)
                 + judges / u["per_arm_rollouts"]) * 1.1,
             "A0s's plain, user and operator_pr cells (a comparator, never a lever; its band "
             "is provisional until S0b)"),
            ("One A2 draw, not two",
             s3_data(3000, fx) + dpo_h(3000, 1) + arm_read(fx) + 0.3,
             "the draw-to-draw read ST-A38 asks for"),
            ("SDF corpus 5M tokens, not 10M (the 1M pilot prices it anyway)",
             gen_h(10_000_000, 1, True, fx) + gen_h(5_000_000, 1, False, fx)
             + u["sdf_10M_h"], "dose"),
        ]
        run = out["engine_fixed" if fx else "current"]["partner_recipe_S0_S3"]
        ranked = []
        for name, saved, loss in cuts:
            run -= saved
            ranked.append({"cut": name, "saves_a100_h": saved, "after": run, "loses": loss})
        return ranked

    codex = {"DPO pair filter, per draw (3,000 pairs at a 0.7 pass rate, 10 per call)":
             -(-int(3000 / 0.7) // 10),
             "the 200-pair pilot": 20,
             "frontier second opinion, about 200 S0a rollouts at 10 per call": 20}
    return {"units": {k: v for k, v in u.items() if k != "s0a_measured"}, "codex_calls": codex,
            "forecast": out, "cuts_recipe": {"engine_fixed": cut_list(True),
                                             "current": cut_list(False)},
            "deferrable_sessions": {"S5 (gated by R3)": out["engine_fixed"]["sessions"][
                "S5 (A4, A4f; gated by R3; estimate)"], "S4 (B1)": 29.0},
            "dollars_per_a100_h": [1.6, 2.0], "envelope_usd": [500, 1000],
            "envelope_of_record_a100_h": 150}


def main() -> None:
    f = forecast()
    (CST / "outputs" / "t09_reforecast.json").write_text(json.dumps(f, indent=1) + "\n")
    for case, d in f["forecast"].items():
        print(f"== {case}: total {d['total']:.1f}, partner recipe S0-S3 "
              f"{d['partner_recipe_S0_S3']:.1f}")
        for k, v in d["sessions"].items():
            print(f"   {v:6.1f}  {k}")
    for case, cl in f["cuts_recipe"].items():
        print(f"== cuts ({case}, recipe path, cumulative)")
        for c in cl:
            print(f"   -{c['saves_a100_h']:5.1f} -> {c['after']:6.1f}  {c['cut']}")
    print({k: round(v, 1) if isinstance(v, float) else v for k, v in f["units"].items()
           if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
