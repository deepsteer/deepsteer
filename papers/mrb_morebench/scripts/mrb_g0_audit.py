"""MRB-G0 zero-GPU, zero-API audit (MRB_PLAN v0.2 §5): writes outputs/mrb_g0_audit.json.

Counts and schema of the pinned public and theory sets; a static audit of the upstream code at the
code of record (``eval``) and at ``main``; the scorer's equivalence with upstream's own
``calculate_score_for_a_task`` on synthetic judgements; and the paper's table arithmetic (Hard
from Regular and length, and each row's Regular against the public file's composition). Ids and
counts only; no scenario or criterion text.

Usage:
    python papers/mrb_morebench/scripts/mrb_g0_audit.py [--write-manifest]
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from deepsteer.benchmarks.moral_reasoning.morebench.scoring import (  # noqa: E402
    code_score,
    eq1_score,
)
from deepsteer.datasets.morebench import (  # noqa: E402
    MANIFEST_PATH,
    MOREBENCH_FILES,
    MOREBENCH_REPO,
    MOREBENCH_REVISION,
    default_cache_dir,
    fetch_morebench,
    load_morebench,
    sha256_file,
)

AUDIT_VERSION = "1.0.0"
OUT = REPO / "papers" / "mrb_morebench" / "outputs" / "mrb_g0_audit.json"
CODE_EVAL = "e6689bbb9d742c48ea78ea07acd7931eba05b213"
CODE_MAIN = "71eaf71857fecf57e20e30fb40416d843fdd473b"
CODE_DIR = Path.home() / ".cache" / "deepsteer" / "morebench" / "code" / CODE_EVAL

# arXiv:2510.16380v2, Table 6 (p. 43, thinking traces) and Table 7 (p. 45, final responses):
# Daily, AIRisk, Expert, Original, Extended, Advisor, Agent, Regular, length (chars), Hard.
TABLE6 = {
    "GPT-5-high": [61.3, 61.1, 61.7, 63.2, 59.2, 61.4, 61.1, 61.3, 3895, 15.7],
    "GPT-5-mini-high": [62.7, 64.7, 65.1, 64.6, 62.8, 63.4, 64.8, 64.0, 4492, 14.2],
    "GPT-5-nano-high": [59.6, 60.3, 64.5, 60.0, 60.0, 61.1, 60.5, 60.9, 4983, 12.2],
    "Claude Opus 4.1": [45.3, 53.8, 56.3, 44.9, 54.2, 48.6, 54.1, 50.9, 1272, 40.0],
    "Claude Sonnet 4": [54.3, 58.6, 59.5, 52.1, 60.8, 55.9, 58.8, 57.1, 1898, 30.1],
    "Gemini-2.5-Pro": [35.7, 36.1, 37.0, 35.1, 36.7, 36.3, 35.8, 36.1, 2705, 13.3],
    "Gemini-2.5-Flash": [39.5, 43.7, 40.7, 39.0, 44.2, 39.8, 43.7, 41.4, 2512, 16.5],
    "Gemini-2.5-Flash-Lite": [34.4, 38.2, 37.7, 34.1, 38.5, 35.6, 38.0, 36.6, 3373, 10.9],
    "GPT-oss-120b": [51.7, 50.0, 50.5, 51.9, 49.8, 51.4, 49.8, 50.8, 1272, 39.9],
    "GPT-oss-20b": [55.5, 58.0, 61.2, 55.9, 57.7, 57.0, 58.5, 57.7, 2338, 24.7],
    "DeepSeek-V3.1": [49.9, 56.8, 51.7, 51.4, 55.3, 50.3, 56.9, 53.0, 1458, 36.4],
    "DeepSeek-R1-0528": [62.1, 65.1, 68.0, 60.1, 67.1, 64.0, 65.2, 64.5, 2228, 28.9],
    "Qwen3-235B-A22B": [66.7, 70.5, 74.4, 64.6, 72.6, 69.1, 70.7, 69.8, 3164, 22.1],
    "Qwen3-30B-A3B": [67.2, 70.9, 75.2, 64.7, 73.4, 69.8, 71.0, 70.3, 3691, 19.0],
}
TABLE7 = {
    "GPT-5-high": [72.1, 73.3, 75.1, 71.3, 74.1, 73.3, 73.1, 73.2, 4019, 18.2],
    "GPT-5-mini-high": [74.2, 76.3, 77.0, 74.1, 76.4, 75.1, 76.3, 75.6, 4802, 15.7],
    "GPT-5-nano-high": [71.5, 72.3, 73.8, 70.1, 73.6, 72.6, 71.9, 72.3, 6014, 12.0],
    "Claude Opus 4.1": [64.6, 65.0, 68.2, 62.3, 67.4, 65.9, 65.0, 65.5, 2274, 28.8],
    "Claude Sonnet 4": [63.0, 63.3, 64.5, 62.1, 64.1, 63.7, 63.0, 63.4, 2069, 30.6],
    "Gemini-2.5-Pro": [73.2, 74.5, 76.0, 71.3, 76.4, 74.2, 74.4, 74.3, 6098, 12.2],
    "Gemini-2.5-Flash": [74.2, 74.7, 77.3, 72.2, 76.7, 75.4, 74.5, 75.0, 6398, 11.7],
    "Gemini-2.5-Flash-Lite": [72.0, 70.1, 74.3, 67.9, 74.2, 72.9, 69.9, 71.7, 5246, 13.7],
    "GPT-oss-120b": [72.4, 68.1, 68.4, 68.5, 72.0, 71.4, 67.8, 69.9, 6213, 11.3],
    "GPT-oss-20b": [70.9, 72.0, 72.9, 67.8, 75.1, 71.4, 72.2, 71.7, 5071, 14.1],
    "DeepSeek-V3.1": [73.5, 73.9, 74.7, 71.3, 76.1, 73.9, 73.9, 73.9, 4571, 16.2],
    "DeepSeek-R1-0528": [72.9, 71.5, 75.3, 69.4, 75.1, 73.9, 71.4, 72.9, 4218, 17.3],
    "Qwen3-235B-A22B": [76.7, 77.5, 79.5, 75.0, 79.3, 77.6, 77.6, 77.6, 5995, 12.9],
    "Qwen3-30B-A3B": [74.2, 75.4, 77.8, 72.6, 77.1, 75.5, 75.3, 75.4, 5973, 12.6],
}
AVERAGE = {"table6": [53.5, 55.7, 56.7, 53.5, 55.7, 54.5, 55.8, 55.0, 2786.3, 22.8],
           "table7": [71.8, 72.0, 73.9, 69.7, 74.1, 72.6, 71.9, 72.3, 4925.8, 16.2]}


def schema(items) -> dict:
    crit = [c for it in items for c in it.criteria]
    per = [len(it.criteria) for it in items]
    w = Counter(c.weight for c in crit)
    return {
        "rows": len(items), "criteria": len(crit),
        "criteria_per_row": {"min": min(per), "max": max(per), "mean": round(float(np.mean(per)), 2)},
        "weights": {str(k): {"n": v, "pct": round(100 * v / len(crit), 2)} for k, v in sorted(w.items())},
        "dimensions": dict(Counter(c.dimension for c in crit).most_common()),
        "sources": dict(Counter(it.source for it in items).most_common()),
        "types": dict(Counter(it.dilemma_type for it in items).most_common()),
        "roles": dict(Counter(it.role for it in items).most_common()),
        "theories": dict(Counter(it.theory for it in items).most_common()),
        "contexts": len({it.context for it in items}),
        "dilemma_words": {"min": min(len(it.dilemma.split()) for it in items),
                          "max": max(len(it.dilemma.split()) for it in items),
                          "mean": round(float(np.mean([len(it.dilemma.split()) for it in items])), 1)},
        "zero_weight": w.get(0, 0),
        "duplicate_criterion_ids": len(crit) - len({c.criterion_id for c in crit}),
    }


def theory_structure(public, theory) -> dict:
    by_dilemma = Counter(it.dilemma for it in theory)
    pub = {it.dilemma for it in public}
    return {"unique_dilemmas": len(by_dilemma),
            "rows_per_dilemma": dict(Counter(by_dilemma.values())),
            "dilemmas_also_in_public": sum(d in pub for d in by_dilemma)}


def _git_show(commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(CODE_DIR), "show", f"{commit}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def _thinking_paths_set(utils_src: str, key: str) -> dict[str, bool]:
    """Whether each reasoning-path generation function's own body mentions ``key``."""
    out = {}
    for node in ast.parse(utils_src).body:
        if isinstance(node, ast.FunctionDef) and re.fullmatch(r"_collect_\w+_thinking", node.name):
            out[node.name] = key in ast.get_source_segment(utils_src, node)
    return out


def code_audit() -> dict:
    out = {}
    for name, commit in (("eval", CODE_EVAL), ("main", CODE_MAIN)):
        calc = _git_show(commit, "calculate_morebench.py")
        utils = _git_show(commit, "utils.py")
        m = re.search(r'"--expected_samples".*?default=(\d+)', calc, re.S)
        out[name] = {
            "expected_samples_default": int(m.group(1)) if m else None,
            "task_id_fallback": 'dp.get("TASK_ID"' in utils,
            "judge_call": re.search(r"def get_judge_response\(([^)]*)\)", utils).group(1),
            "judge_reasoning_effort_high": '"reasoning_effort": "high"' in utils,
            "openrouter_generation_effort_high": utils.count('"reasoning_effort": "high"') >= 2,
            "temperature_set_on_reasoning_paths": _thinking_paths_set(utils, "temperature"),
            "top_p_set_on_reasoning_paths": _thinking_paths_set(utils, "top_p"),
            "credit_test": "substring" if '"yes" in judgement' in utils else "other",
        }
    gen = _git_show(CODE_EVAL, "run_inferences_on_dilemmas.py")
    out["dataset_split_requested"] = re.search(r'split="(\w+)"', gen).group(1)
    out["generation_seed_passed_to_api"] = "seed=args.seed" in gen or "seed=" in re.sub(
        r"--seed.*", "", gen)
    return out


def _upstream_scorer():
    """upstream utils.calculate_score_for_a_task, extracted by AST (no upstream import)."""
    src = _git_show(CODE_EVAL, "utils.py")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "calculate_score_for_a_task")
    ns: dict = {}
    exec(compile(ast.Module([fn], []), "upstream_utils", "exec"), ns)  # noqa: S102
    return ns["calculate_score_for_a_task"]


def scorer_equivalence(n: int = 5000) -> dict:
    up = _upstream_scorer()
    rng = np.random.default_rng(0)
    replies = ["yes", "no", "Yes.", "No.", " YES ", "no.", "not met", "I know it", "eyes",
               "yes, it does not", "", "maybe"]
    weights = [-3, -2, -1, 1, 2, 3]
    mismatch, eq1_dev = 0, 0.0
    for _ in range(n):
        k = int(rng.integers(1, 30))
        crit = [(int(rng.choice(weights)), str(rng.choice(replies))) for _ in range(k)]
        ours = code_score(crit)
        theirs = up([{"criterion_weight": w, "judgement": j} for w, j in crit])
        mismatch += ours != theirs
        eq1_dev = max(eq1_dev, abs(eq1_score(crit) - (2 * ours / 100 - 1)))
    traps = {r: {"positive_credited": code_score([(1, r)]) == 100,
                 "negative_credited": code_score([(-1, r)]) == 100} for r in replies}
    return {"n_sets": n, "mismatches_vs_upstream": mismatch,
            "max_abs_eq1_minus_2code_minus_1": eq1_dev, "credit_by_reply": traps}


def table_arithmetic(table: dict, public) -> dict:
    src = Counter(it.source for it in public)
    typ = Counter(it.dilemma_type for it in public)
    rol = Counter(it.role for it in public)
    n = len(public)
    w_src = (src["daily_dilemmas"], src["ai_risk_dilemmas"], typ["expert_case"])
    w_typ = (typ["short_case"], typ["long_case"], typ["expert_case"])
    w_rol = (rol["ai_advisor"], rol["ai_agent"])
    rows = {}
    for m, (d, a, e, o, x, adv, ag, reg, ln, hard) in table.items():
        rows[m] = {
            "hard_recomputed": round(reg / ln * 1000, 1), "hard_printed": hard,
            "hard_matches": round(reg / ln * 1000, 1) == hard,
            "regular_from_source_partition": round((w_src[0] * d + w_src[1] * a + w_src[2] * e) / n, 3),
            "regular_from_type_partition": round((w_typ[0] * o + w_typ[1] * x + w_typ[2] * e) / n, 3),
            "regular_from_role_partition": round((w_rol[0] * adv + w_rol[1] * ag) / n, 3),
            "regular_printed": reg,
        }
        r = rows[m]
        r["partitions_within_0.1"] = all(abs(r[k] - reg) <= 0.1 + 1e-9 for k in (
            "regular_from_source_partition", "regular_from_type_partition",
            "regular_from_role_partition"))
    return {"weights": {"source": w_src, "type": w_typ, "role": w_rol}, "rows": rows,
            "hard_all_match": all(r["hard_matches"] for r in rows.values()),
            "partitions_all_within_0.1": all(r["partitions_within_0.1"] for r in rows.values()),
            "equal_lengths": [ln for ln, c in Counter(v[8] for v in table.values()).items() if c > 1]}


def average_rows() -> dict:
    out = {}
    for name, table in (("table6", TABLE6), ("table7", TABLE7)):
        means = np.mean(np.array(list(table.values()), dtype=float), axis=0)
        out[name] = {"recomputed": [round(float(v), 2) for v in means], "printed": AVERAGE[name]}
    return out


def write_manifest(root: Path) -> None:
    MANIFEST_PATH.write_text(json.dumps({
        "source": "MoReBench (Chiu, Lee, et al., arXiv:2510.16380); CC BY 4.0; fetched, not vendored",
        "repo": MOREBENCH_REPO, "revision": MOREBENCH_REVISION,
        "checked": dt.date.today().isoformat(),
        "files": {f: sha256_file(root / f) for f in MOREBENCH_FILES},
    }, indent=1, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()
    if args.write_manifest:
        write_manifest(default_cache_dir())
    root = fetch_morebench()
    public, theory = load_morebench(root, "public"), load_morebench(root, "theory")
    audit = {
        "metadata": {"audit_version": AUDIT_VERSION, "date": dt.date.today().isoformat(),
                     "hf_revision": MOREBENCH_REVISION, "code_of_record": CODE_EVAL,
                     "code_main": CODE_MAIN,
                     "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
                     "paper": "arXiv:2510.16380v2 (Tables 6, 7)"},
        "public": schema(public), "theory": schema(theory),
        "theory_structure": theory_structure(public, theory),
        "code": code_audit(), "scorer": scorer_equivalence(),
        "table6": table_arithmetic(TABLE6, public), "table7": table_arithmetic(TABLE7, public),
        "averages": average_rows(),
    }
    payload = json.dumps(audit, indent=1)
    for it in public + theory:  # leak guard: no scenario or criterion text in tracked output
        for s in [it.dilemma] + [c.title for c in it.criteria]:
            if len(s) >= 30 and s in payload:
                raise AssertionError("tracked output would contain MoReBench text")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
