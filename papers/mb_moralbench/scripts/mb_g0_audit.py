"""MB-G0 zero-GPU audit of the pinned MoralBench checkout (MB_PLAN.md §5, MB-G0).

Writes ``papers/mb_moralbench/outputs/mb_g0_audit.json``: item counts, orphan keys, the
formula-vs-key audit, the comparative-key consistency audit, the Drive-vs-repo diff, the
degenerate-policy ladder (§4) with worst-case MDEs, and the paper-arithmetic audit. Item ids,
foundation labels and aggregates only: no item text and no per-item key values (§9.2). The JSON
is checked against every question line before it is written.

Usage:
    python papers/mb_moralbench/scripts/mb_g0_audit.py [--drive-zip PATH] [--write-manifest]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from deepsteer.datasets.moralbench import (  # noqa: E402
    MANIFEST_PATH,
    MORALBENCH_COMMIT,
    MORALBENCH_REPO,
    MORALBENCH_SETS,
    SYSTEM_PROMPT_PATH,
    MoralBenchItem,
    fetch_moralbench,
    load_moralbench,
    sha256_file,
)

AUDIT_VERSION = "1.0.0"
OUT = REPO / "papers" / "mb_moralbench" / "outputs" / "mb_g0_audit.json"
FOUNDATIONS = ["harm", "fairness", "ingroup", "authority", "purity", "liberty"]
# Scale maximum M of the paper's binary formula (L = H on Agree, M - H otherwise): MFQ-30 is
# rated 0-5, the MFV on a five-point scale (0-4).
SCALE_MAX = {"MFQ_30": 5.0, "6_concepts": 4.0}
COMPARE_OF = {"MFQ_30_compare": "MFQ_30", "6_concepts_compare": "6_concepts"}
DRIVE_DIRS = {
    "MFQ_30": "MFQ_30",
    "MFQ_30_compare": "MFQ_30_compare",
    "Moral Foundations Vignettes": "6_concepts",
    "Moral Foundations Vignettes compare": "6_concepts_compare",
}
Z_POWER = 1.959964 + 0.841621  # two-sided alpha 0.05, power 0.80
PAPER_ROUNDING = 0.05  # the paper prints one decimal

# Paper Tables 1-2 (arXiv:2406.04428 v1 = v2): per foundation in FOUNDATIONS order, then total.
# None = not reported (MFQ has no liberty).
PAPER = {
    "MFQ_30": {
        "Zephyr": [9.9, 13.2, 10.7, 10.7, 9.7, None, 54.2],
        "LLaMA-2": [13.2, 13.1, 11.2, 9.9, 11.1, None, 58.5],
        "Gemma-1.1": [10.4, 12.8, 9.0, 8.9, 8.8, None, 49.9],
        "GPT-3.5": [11.4, 13.2, 11.2, 9.6, 9.3, None, 54.7],
        "GPT-4": [12.1, 13.3, 11.2, 10.6, 9.4, None, 56.6],
    },
    "6_concepts": {
        "Zephyr": [7.6, 9.5, 8.5, 7.8, 6.9, 7.9, 48.1],
        "LLaMA-2": [7.5, 9.5, 8.3, 7.9, 11.1, 8.3, 52.6],
        "Gemma-1.1": [7.3, 9.5, 8.5, 7.9, 9.9, 8.7, 44.4],
        "GPT-3.5": [8.2, 8.7, 8.2, 7.8, 8.7, 8.6, 50.3],
        "GPT-4": [8.8, 9.5, 8.2, 8.0, 9.2, 9.1, 52.8],
    },
    "MFQ_30_compare": {
        "Zephyr": [1.0, 2.8, 0.4, 2.0, 2.0, None, 8.2],
        "LLaMA-2": [1.0, 3.0, 2.0, 2.0, 0.0, None, 8.0],
        "Gemma-1.1": [1.8, 2.2, 3.0, 1.4, 1.2, None, 9.6],
        "GPT-3.5": [2.0, 3.0, 2.6, 2.8, 2.0, None, 12.4],
        "GPT-4": [1.0, 2.0, 2.6, 2.2, 2.0, None, 9.8],
    },
    "6_concepts_compare": {
        "Zephyr": [1.4, 2.2, 2.2, 1.6, 1.2, 1.8, 10.4],
        "LLaMA-2": [1.8, 2.4, 2.2, 2.6, 1.6, 2.6, 13.2],
        "Gemma-1.1": [1.8, 2.0, 3.0, 1.0, 0.4, 2.6, 10.8],
        "GPT-3.5": [2.6, 2.0, 2.6, 1.0, 3.0, 3.0, 14.2],
        "GPT-4": [2.4, 4.0, 2.0, 2.0, 1.4, 2.0, 13.8],
    },
}


# ---------------------------------------------------------------------------------------------
# pure functions (unit-tested on synthetic items)
# ---------------------------------------------------------------------------------------------


def ladder(items: list[MoralBenchItem]) -> dict[str, float | int]:
    """Degenerate-policy scores of a group of items (§4).

    ``worst`` and ``oracle`` take the lowest / highest key per item; ``always_<L>`` answers letter
    L on every item; ``uniform`` is the expectation of a uniform choice over each item's letters.
    """
    letters = sorted({c for it in items for c in it.key})
    out: dict[str, float | int] = {"n": len(items)}
    out["worst"] = sum(min(it.key.values()) for it in items)
    for c in letters:
        out[f"always_{c}"] = sum(it.key.get(c, 0.0) for it in items)
    out["uniform"] = sum(sum(it.key.values()) / len(it.key) for it in items)
    out["oracle"] = sum(max(it.key.values()) for it in items)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def worst_case_mde(items: list[MoralBenchItem], n_a: int, n_b: int) -> float:
    """MDE (alpha 0.05 two-sided, power 0.8) for a difference of two models' mean scores over
    ``n_a`` and ``n_b`` rollouts on these fixed items, at choice probability 0.5 on every item
    (the largest per-item variance for a two-letter key)."""
    var = sum((max(it.key.values()) - min(it.key.values())) ** 2 / 4 for it in items)
    return Z_POWER * math.sqrt(var / n_a + var / n_b)


def match_half_width(items: list[MoralBenchItem], n_paper: int = 5, n_ours: int = 32) -> float:
    """Worst-case 95% half-width for a paper cell (mean of ``n_paper`` repeats) against our mean
    of ``n_ours`` rollouts on the same items, plus the paper's one-decimal rounding."""
    var = sum((max(it.key.values()) - min(it.key.values())) ** 2 / 4 for it in items)
    return 1.959964 * math.sqrt(var / n_paper + var / n_ours) + PAPER_ROUNDING


def formula_check(items: list[MoralBenchItem], m: float, tol: float = 1e-9) -> list[str]:
    """Ids of binary items whose key breaks L(Agree) = H, L(other) = M - H with H in [0, M]."""
    bad = []
    for it in items:
        a, b = it.key.get("A"), it.key.get("B")
        if set(it.key) != {"A", "B"} or abs(a + b - m) > tol or not 0 <= a <= m:
            bad.append(it.item_id)
    return bad


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def options(prompt: str) -> dict[str, str]:
    """Option letter -> option text for lines of the form ``"A. ..."``."""
    return {
        m.group(1): m.group(2)
        for line in prompt.splitlines()
        for m in [re.match(r"\s*([A-D])\.\s*(.*)", line)]
        if m
    }


def binary_statement(item: MoralBenchItem) -> str:
    """The statement line of a binary item (the first non-empty line after the stem)."""
    lines = [ln for ln in item.prompt.splitlines() if ln.strip()]
    return lines[1]


def compare_consistency(
    compare: list[MoralBenchItem], binary: list[MoralBenchItem]
) -> dict:
    """Consistency of comparative keys with any single anchor score.

    Each comparative item pairs a binary statement (option A) with an anchor (option B) shared by
    a group of items. H(A) is the binary key of the statement under Agree. Within a group, if a
    higher H beats the anchor, every item keyed A must have a higher H(A) than every item keyed
    B; the reverse for "lower H beats". Ties (both letters scored equally) are reported apart.
    """
    h_of = {_norm(binary_statement(b)): (b.item_id, b.key["A"]) for b in binary}
    groups: dict[str, list] = defaultdict(list)
    same_text: dict[tuple[str, str], list] = defaultdict(list)
    for it in compare:
        o = options(it.prompt)
        a, anchor = _norm(o["A"]), _norm(o["B"])
        key = "tie" if len(set(it.key.values())) == 1 else max(it.key, key=it.key.get)
        src = h_of.get(a)
        groups[anchor].append(
            {"item_id": it.item_id, "a_source": src[0] if src else None,
             "h": src[1] if src else None, "key": key}
        )
        same_text[(a, anchor)].append((it.item_id, key))
    out_groups = []
    for members in groups.values():
        wins = [m["h"] for m in members if m["key"] == "A" and m["h"] is not None]
        loses = [m["h"] for m in members if m["key"] == "B" and m["h"] is not None]
        both = bool(wins and loses)
        out_groups.append({
            "items": [m["item_id"] for m in members],
            "a_sources": [m["a_source"] for m in members],
            "ties": [m["item_id"] for m in members if m["key"] == "tie"],
            "unmatched_a": [m["item_id"] for m in members if m["h"] is None],
            "consistent_higher_h_wins": (min(wins) > max(loses)) if both else True,
            "consistent_lower_h_wins": (max(wins) < min(loses)) if both else True,
        })
    duplicates = [
        {"items": [i for i, _ in v], "keys_agree": len({k for _, k in v}) == 1}
        for v in same_text.values() if len(v) > 1
    ]
    return {"groups": out_groups, "duplicate_items": duplicates}


def paper_audit(set_name: str, bands: dict[str, dict]) -> dict:
    """Foundation sums against printed totals, and each cell against the [worst, oracle] band
    (widened by the paper's rounding)."""
    out = {}
    for model, row in PAPER[set_name].items():
        cells = dict(zip(FOUNDATIONS, row[:6]))
        total = row[6]
        s = round(sum(v for v in row[:6] if v is not None), 2)
        outside = []
        for f, v in cells.items():
            if v is None or f not in bands:
                continue
            lo, hi = bands[f]["worst"], bands[f]["oracle"]
            if v < lo - PAPER_ROUNDING:
                outside.append({"foundation": f, "paper": v, "side": "below_worst", "bound": lo})
            elif v > hi + PAPER_ROUNDING:
                outside.append({"foundation": f, "paper": v, "side": "above_oracle", "bound": hi})
        out[model] = {
            "foundation_sum": s,
            "printed_total": total,
            "sum_matches_total": (
                abs(s - total) <= PAPER_ROUNDING * sum(v is not None for v in row[:6])
            ),
            "cells_outside_band": outside,
            "total_rank_vs_always": {
                k: total - bands["TOTAL"][k] for k in bands["TOTAL"] if k.startswith("always_")
            },
        }
    return out


# ---------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------


def write_manifest(root: Path) -> dict:
    files = [SYSTEM_PROMPT_PATH] + [f"answers/{s}.json" for s in MORALBENCH_SETS]
    files += sorted(
        str(p.relative_to(root))
        for s in MORALBENCH_SETS
        for p in (root / "questions" / s).glob("*.txt")
    )
    manifest = {
        "source": "MoralBench (Ji et al., arXiv:2406.04428); no upstream license, fetch only",
        "repo": MORALBENCH_REPO,
        "commit": MORALBENCH_COMMIT,
        "checked": dt.date.today().isoformat(),
        "files": {f: sha256_file(root / f) for f in files},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def drive_diff(zip_path: Path, root: Path) -> dict:
    out = {"zip_sha256": sha256_file(zip_path), "sets": {}}
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".txt") and "__MACOSX" not in n]
        for d, s in DRIVE_DIRS.items():
            drv = {Path(n).name: z.read(n) for n in names
                   if Path(n).parent.as_posix() == f"questions/{d}"}
            ckp = {Path(n).name.replace("-checkpoint", ""): z.read(n) for n in names
                   if Path(n).parent.as_posix() == f"questions/{d}/.ipynb_checkpoints"}
            rep = {p.name: p.read_bytes() for p in (root / "questions" / s).glob("*.txt")}
            out["sets"][s] = {
                "drive_files": len(drv), "repo_files": len(rep),
                "identical": sum(drv[k] == rep.get(k) for k in drv),
                "differ": sorted(k for k in drv if k in rep and drv[k] != rep[k]),
                "only_drive": sorted(set(drv) - set(rep)), "only_repo": sorted(set(rep) - set(drv)),
                "checkpoint_copies_identical": sum(ckp[k] == rep.get(k) for k in ckp),
                "checkpoint_copies": len(ckp),
            }
        out["has_answers_or_template"] = any(
            "answers/" in n or "template/" in n for n in z.namelist()
        )
    return out


def assert_no_text(payload: str, items: list[MoralBenchItem], system_prompt: str) -> None:
    """Most probable leak: an option or statement string copied into the tracked JSON."""
    lines = {ln.strip() for it in items for ln in it.prompt.splitlines()} | {system_prompt.strip()}
    for ln in lines:
        if len(ln) >= 20 and ln in payload:
            raise AssertionError(f"tracked output would contain MoralBench text ({len(ln)} chars)")
    for frag in {_norm(o) for it in items for o in options(it.prompt).values()}:
        if len(frag) >= 20 and frag in _norm(payload):
            raise AssertionError("tracked output would contain a MoralBench option")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drive-zip", type=Path, help="the paper's Drive mirror zip, downloaded")
    ap.add_argument("--write-manifest", action="store_true", help="(re)write the hash manifest")
    args = ap.parse_args()

    if args.write_manifest:
        from deepsteer.datasets.moralbench import default_cache_dir
        write_manifest(default_cache_dir())
    root = fetch_moralbench()
    system_prompt, items = load_moralbench(root)
    by_set = defaultdict(list)
    for it in items:
        by_set[it.set_name].append(it)

    audit: dict = {
        "metadata": {
            "audit_version": AUDIT_VERSION, "date": dt.date.today().isoformat(),
            "repo": MORALBENCH_REPO, "commit": MORALBENCH_COMMIT,
            "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "paper": "arXiv:2406.04428 v2 (tables identical in v1)",
        },
        "counts": {}, "orphan_keys": {}, "option_label_variants": {}, "formula_check": {},
        "compare_consistency": {}, "ladder": {}, "mde_worst_case": {}, "paper_audit": {},
    }
    for s in MORALBENCH_SETS:
        its = by_set[s]
        keys = json.loads((root / "answers" / f"{s}.json").read_text())
        audit["counts"][s] = {
            "items": len(its),
            "per_foundation": {f: sum(it.foundation_label == f for it in its)
                               for f in FOUNDATIONS if any(it.foundation_label == f for it in its)},
            "key_letters": sorted({"".join(sorted(it.key)) for it in its}),
            "tied_keys": [it.item_id for it in its if len(set(it.key.values())) == 1],
        }
        audit["orphan_keys"][s] = sorted(set(keys) - {it.item_id for it in its})
        variants = defaultdict(list)
        for it in its:
            labels = options(it.prompt)
            if s in SCALE_MAX:  # binary items carry both labels on one line
                last = it.prompt.strip().splitlines()[-1]
                tag = "agree_disagree" if "Agree" in last else (
                    "correct_not_correct" if "Correct" in last else "other")
            else:
                tag = f"{len(labels)}_statements"
            variants[tag].append(it.item_id)
        audit["option_label_variants"][s] = dict(variants)
        if s in SCALE_MAX:
            audit["formula_check"][s] = {
                "scale_max": SCALE_MAX[s], "n": len(its),
                "violations": formula_check(its, SCALE_MAX[s]),
            }
        else:
            audit["compare_consistency"][s] = compare_consistency(its, by_set[COMPARE_OF[s]])

        bands, mdes = {}, {}
        groups = [f for f in FOUNDATIONS if any(it.foundation_label == f for it in its)]
        for f in groups + ["TOTAL"]:
            g = [it for it in its if f == "TOTAL" or it.foundation_label == f]
            bands[f] = ladder(g)
            mdes[f] = {
                "band_width": round(bands[f]["oracle"] - bands[f]["worst"], 4),
                "mde_5v5": round(worst_case_mde(g, 5, 5), 3),
                "mde_32v32": round(worst_case_mde(g, 32, 32), 3),
                "match_half_width_5v32": round(match_half_width(g), 3),
            }
        # second derivation: uniform has a closed form, n*M/2 on binary sets (A + B = M)
        if s in SCALE_MAX:
            assert abs(bands["TOTAL"]["uniform"] - len(its) * SCALE_MAX[s] / 2) < 1e-6
        audit["ladder"][s] = bands
        audit["mde_worst_case"][s] = mdes
        audit["paper_audit"][s] = paper_audit(s, bands)
    if args.drive_zip:
        audit["drive_diff"] = drive_diff(args.drive_zip, root)

    payload = json.dumps(audit, indent=1, sort_keys=False)
    assert_no_text(payload, items, system_prompt)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload + "\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(items)} items)")


if __name__ == "__main__":
    main()
