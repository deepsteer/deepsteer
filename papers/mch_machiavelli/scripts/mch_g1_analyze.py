"""MCH-G1 analysis, as registered in MCH_PLAN.md §5 Amendments 3-5.

Reads ``outputs/g1/`` (``mch_g1_harness.py``) and G0's random-baseline summary and writes
``outputs/mch_g1.json``:
- the harness pass criteria (parity, the S arm's run-to-run stability, engine failures) and the
  fallback rate, action-number mass and truncation rate per arm;
- the LM agent's within-game variance from arm V (k = 2 on every test game), SE₁ with a bootstrap
  over games, MDE_k for k = 1 … 8, and its ratio to the random agent's;
- the Phase 2 go/no-go (Amendment 3) and its price;
- Amendment 4's k for the new-model rows, and their price, with the CoT chat prompts' token counts
  measured on G1's own states (replayed from seed and actions) under each target tokenizer.

Usage:
    python papers/mch_machiavelli/scripts/mch_g1_analyze.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
MCH = HERE.parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(MCH))
sys.path.insert(0, str(REPO))
from deepsteer.benchmarks.agentic.machiavelli import prompts as P  # noqa: E402

G1 = MCH / "outputs" / "g1"
G0 = MCH / "outputs" / "mch_g0_random.json"
OUT = MCH / "outputs" / "mch_g1.json"
N_BOOT, Z_POWER = 10_000, 2.80
METRICS = ("violations.Σ", "power.Σ", "utility.Σ", "points")

# Pricing bases (MCH_G0_REPORT §7; Amendment 3): 30B dense prefill on one A100, half of each
# prompt prefix-cached, x1.3 for full rotation. Generation rates measured at MRB (output tok/s).
PREFILL_TOK_S = 3000.0
ROTATION = 1.3
GEN_TOK_S = {"gpt-oss-20b": 6688.0, "qwen3-30b-a3b": 2356.0,
             "muse-glimmer-30b": 1410.0, "qwen3.8-27b": 906.0}
TARGETS = {  # new-model rows (Amendment 4): tokenizer, revision, prefill after the template
    "gpt-oss-20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee",
                    "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final"
                    "<|message|>"),
    "qwen3-30b-a3b": ("Qwen/Qwen3-30B-A3B-Thinking-2507",
                      "144afc2f379b542fdd4e85a1fcd5e1f79112d95d", "\n</think>\n\n"),
}
CHAT_MAX_TOKS = 4096 - 256  # upstream chat_completion_with_retries at max_tokens 256
MAX_GEN = 256


def load(g1: Path) -> tuple[list[dict], list[dict], dict]:
    eps = [json.loads(x) for x in (g1 / "episodes.jsonl").read_text().splitlines()]
    decs = [json.loads(x) for x in gzip.open(g1 / "decisions.jsonl.gz", "rt")]
    return eps, decs, json.loads((g1 / "run_meta.json").read_text())


def metric(ep: dict, m: str) -> float | None:
    return ep["points_norm"] if m == "points" else ep["scores"][m]["r0"]


def harness(eps: list[dict], decs: list[dict], meta: dict) -> dict:
    by = collections.defaultdict(list)
    for d in decs:
        by[(d["arm"], d["game"], d["episode"])].append(d)
    for v in by.values():
        v.sort(key=lambda d: d["step"])
    stab = {}
    for (arm, game, e), rows in by.items():
        if arm != "S":
            continue
        ref = by[("V", game, e)]
        same = [a["action"] == b["action"] for a, b in zip(rows, ref)]
        div = next((i for i, s in enumerate(same) if not s), None)
        upto = len(same) if div is None else div + 1
        dp = max(max(abs(x - y) for x, y in zip(a["r0"], b["r0"]))
                 for a, b in zip(rows[:upto], ref[:upto]))
        stab[game] = {"identical": div is None and len(rows) == len(ref),
                      "decisions": len(rows), "reference_decisions": len(ref),
                      "first_divergence": div, "max_abs_dp_until_divergence": dp}
    per_arm = {}
    for arm in ("V", "S", "Gr"):
        d = [x for x in decs if x["arm"] == arm]
        e = [x for x in eps if x["arm"] == arm]
        if not d:
            continue
        mass = np.array([x["mass"] for x in d])
        per_arm[arm] = {
            "episodes": len(e), "status": dict(collections.Counter(x["status"] for x in e)),
            "decisions": len(d), "override_rate": float(np.mean([x["override"] for x in d])),
            "truncation_rate": float(np.mean([x["truncated"] for x in d])),
            "mass_median": float(np.median(mass)),
            "mass_p10_p90": [float(np.percentile(mass, 10)), float(np.percentile(mass, 90))],
            "prompt_tokens_mean": float(np.mean([x["tokens"] for x in d])),
            "decisions_per_done_episode": float(np.mean(
                [x["decisions"] for x in e if x["status"] == "done"])),
            "violations_r0_mean": float(np.mean([metric(x, "violations.Σ") for x in e
                                                 if x["status"] == "done"])),
            "points_mean": float(np.mean([x["points_norm"] for x in e if x["status"] == "done"])),
        }
    n_identical = sum(s["identical"] for s in stab.values())
    return {"parity_mismatches": meta["parity_mismatches"], "decisions": meta["decisions"],
            "stability": stab, "stability_identical": n_identical,
            "stability_pass": n_identical >= 4 if len(stab) == 5 else None,
            "engine_failures": sum(1 for x in eps if x["status"] in ("crash", "hang")),
            "per_arm": per_arm,
            "pass": bool(meta["parity_mismatches"] == 0 and n_identical >= 4)}


def variance(eps: list[dict], rng: np.random.Generator, se_random: dict) -> dict:
    out = {}
    v = [x for x in eps if x["arm"] == "V" and x["status"] == "done"]
    for m in METRICS:
        per = collections.defaultdict(list)
        for x in v:
            val = metric(x, m)
            if val is not None:  # F_skip: a missing key leaves the episode out of that metric
                per[x["game"]].append(val)
        games = sorted(g for g, xs in per.items() if len(xs) >= 2)
        s2 = np.array([np.var(per[g], ddof=1) for g in games])
        se1 = float(np.sqrt(s2.sum()) / len(games))
        boot = np.sqrt(s2[rng.integers(0, len(s2), (N_BOOT, len(s2)))].sum(1)) / len(s2)
        mde = {k: Z_POWER * np.sqrt(2) * se1 / np.sqrt(k) for k in range(1, 9)}
        out[m] = {"games_used": len(games), "se_one_agent": se1,
                  "se_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                  "random_se_one_agent": se_random.get(m),
                  "ratio_to_random": se1 / se_random[m] if se_random.get(m) else None,
                  "mde_paired_by_k": {k: float(x) for k, x in mde.items()},
                  "mde_paired_by_k_at_ci_upper": {
                      k: float(Z_POWER * np.sqrt(2) * np.percentile(boot, 97.5) / np.sqrt(k))
                      for k in range(1, 9)}}
    return out


def steering_price(k: int, d: float, t: float) -> list[float]:
    """Phase 2 steering read: 2 models x 2 arms x k episodes per game x 30 games (A100-h)."""
    one = 30 * d * t / PREFILL_TOK_S / 3600 * ROTATION  # no prefix caching
    return [2 * 2 * k * one * 0.5, 2 * 2 * k * one]


def chat_tokens(decs: list[dict], games: list[str]) -> dict:
    """CoT and GoodCoT chat prompt tokens on G1's V episode-0 states, per target tokenizer."""
    from transformers import AutoTokenizer, GPT2TokenizerFast

    from deepsteer.datasets.machiavelli import fetch_machiavelli

    src = fetch_machiavelli(verify_bundle=False)
    cwd = os.getcwd()
    os.chdir(src)
    sys.path.insert(0, str(src))
    from machiavelli.game.machiavelli_env import MachiavelliEnv
    from machiavelli.utils import random_seed

    gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
    toks = {k: AutoTokenizer.from_pretrained(r, revision=v) for k, (r, v, _) in TARGETS.items()}
    counts = {k: [] for k in toks}
    cut = 0
    actions = collections.defaultdict(list)
    for d in sorted((x for x in decs if x["arm"] == "V" and x["episode"] == 0),
                    key=lambda x: x["step"]):
        actions[d["game"]].append(d["action"])
    n = 0
    for g in games:
        env = MachiavelliEnv(game=g)
        random_seed(2417)
        obs, info = env.reset(seed=2417)
        for a in actions[g]:
            s = P.SceneState.from_info(obs, info)
            for agent in ("CoTAgent", "GoodCoTAgent"):
                sys_, user = P.chat_prompts(agent, s)
                ns, nu = len(gpt2.encode(sys_)), len(gpt2.encode(user))
                if ns + nu > CHAT_MAX_TOKS:  # upstream's proportional cut, counted only
                    cut += 1
                for k, tk in toks.items():
                    text = tk.apply_chat_template(
                        [{"role": "system", "content": sys_}, {"role": "user", "content": user}],
                        tokenize=False, add_generation_prompt=True) + TARGETS[k][2]
                    counts[k].append(len(tk.encode(text, add_special_tokens=False)))
                n += 1
            obs, _, done, info = env.step(a)
            if done:
                break
    os.chdir(cwd)
    return {"prompts": n, "chat_truncation_rate": cut / n,
            "tokens_mean": {k: float(np.mean(v)) for k, v in counts.items()},
            "tokens_p90": {k: float(np.percentile(v, 90)) for k, v in counts.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--g1", type=Path, default=G1)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    eps, decs, meta = load(args.g1)
    rng = np.random.default_rng(0)
    g0 = json.loads(G0.read_text())["forks"]["F_skip"]["mde"]
    se_random = {m: g0[m]["se_one_agent"] for m in ("violations.Σ", "power.Σ", "utility.Σ",
                                                    "points")}
    out: dict = {"amendments": "MCH_PLAN.md §5 Amendments 3-5", "run": meta}
    out["harness"] = harness(eps, decs, meta)
    out["variance"] = variance(eps, rng, se_random)

    viol = out["variance"]["violations.Σ"]
    v_arm = out["harness"]["per_arm"]["V"]
    d, t = v_arm["decisions_per_done_episode"], v_arm["prompt_tokens_mean"]
    rows = []
    for k in range(1, 9):
        lo, hi = steering_price(k, d, t)
        rows.append({"k": k, "mde_paired": viol["mde_paired_by_k"][k],
                     "mde_paired_at_se_ci_upper": viol["mde_paired_by_k_at_ci_upper"][k],
                     "price_a100h": [lo, hi], "price_mid": (lo + hi) / 2})
    go = [r for r in rows if r["k"] <= 4 and r["mde_paired"] <= 4 and r["price_mid"] <= 10]
    out["phase2_go_no_go"] = {
        "descriptive_only": "Amendment 6: the 1B proxy decides nothing; the stage-1 pilot does",
        "rule": "some k <= 4 with paired MDE <= 4 on All violations and price midpoint <= 10",
        "table": rows, "go": bool(go), "k": go[0]["k"] if go else None,
        "basis": {"decisions_per_episode": d, "prompt_tokens": t, "prefill_tok_s": PREFILL_TOK_S,
                  "rotation": ROTATION, "cache_range": [0.5, 1.0]}}

    k_new = next((k for k in range(2, 7) if viol["mde_paired_by_k"][k] <= 8), 6)
    games = sorted({x["game"] for x in eps if x["arm"] == "V"})
    ct = chat_tokens(decs, games)
    price = {}
    for name, rate in GEN_TOK_S.items():
        t_chat = ct["tokens_mean"].get(name, max(ct["tokens_mean"].values()))
        n_dec = 2 * k_new * 30 * d  # two agents
        pre = n_dec * t_chat / PREFILL_TOK_S / 3600
        gen = n_dec * MAX_GEN / rate / 3600
        price[name] = {"decisions": n_dec, "prompt_tokens": t_chat, "prefill_a100h": pre,
                       "generation_a100h_at_cap": gen, "total_upper_a100h": pre + gen}
    out["new_model_rows"] = {
        "k_descriptive": "Amendment 6: the stage-1 pilot sets k; this k is the 1B proxy's",
        "envelope_a100h": 6.0,
        "k": k_new, "k_rule": "smallest k >= 2 with paired MDE <= 8 on All violations, cap 6",
        "mde_paired_at_k": viol["mde_paired_by_k"][k_new], "chat_prompts": ct,
        "price": {k: price[k] for k in ("gpt-oss-20b", "qwen3-30b-a3b")},
        "price_total_upper_a100h": sum(price[k]["total_upper_a100h"]
                                       for k in ("gpt-oss-20b", "qwen3-30b-a3b")),
        "phase2_additional_rows": {k: price[k] for k in ("muse-glimmer-30b", "qwen3.8-27b")},
        "basis": "prefill at 3,000 tok/s (30B dense, conservative for these MoE models), 256 "
                 "generated tokens per decision at MRB's measured output rates, one attempt; "
                 "Muse and Qwen3.8 prompt tokens taken as the larger of the two measured"}
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    h = out["harness"]
    print(f"harness pass {h['pass']}: parity mismatches {h['parity_mismatches']}, stability "
          f"{h['stability_identical']}/5, engine failures {h['engine_failures']}")
    print(f"SE1 violations {viol['se_one_agent']:.2f} {viol['se_ci95']} "
          f"(random {se_random['violations.Σ']:.2f}); go {out['phase2_go_no_go']['go']}; "
          f"new-model k {k_new}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
