"""MB-A4 discriminator (MB_PLAN v0.5 §5): recompute the MFV keys from Clifford et al.'s norms.

Reads the cached ``Full-Published-Norms.xlsx`` (Table 1, column "Wrong": mean wrongness, 1-5) and
the pinned MoralBench checkout. Matches every ``6_concepts`` statement and every
``6_concepts_compare`` option to a norms row by normalized text, then asks:

- binary: the stem asks whether the act is *not* morally wrong, so Agree should score
  H = 4 - (Wrong - 1) = 5 - Wrong (0-4). Does the key's Agree score equal it?
- comparative: does the key pick the less-wrong statement (the paper's "more moral"), the more-wrong
  one, or neither?

R_other_H (the comparative keys come from the norms) is supported if one orientation agrees on every
untied item; R_errors otherwise. Output ``outputs/mb_a4_norms.json``: ids, match rates and agreement
counts only (no text, no key values, no per-item norms).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

from mb_g0_audit import assert_no_text, binary_statement, options  # noqa: E402
from xlsx_rows import read_rows  # noqa: E402

from deepsteer.datasets.moralbench import (  # noqa: E402
    MORALBENCH_COMMIT,
    load_moralbench,
    sha256_file,
)

NORMS = Path.home() / ".cache" / "deepsteer" / "mfv_clifford2015" / "Full-Published-Norms.xlsx"
NORMS_SHA256 = "a198b6f48de762bdac1cc6e97ea37341404acecd206161b94601df7f28708cfe"
OUT = REPO / "papers" / "mb_moralbench" / "outputs" / "mb_a4_norms.json"
TOL = 0.051  # keys print H to 1-2 decimals
SCALE_TOP = 5.0  # norms rate wrongness 1-5; Agree ("not wrong") scores 5 - Wrong on 0-4


def norm(text: str) -> str:
    text = re.sub(r"^\s*you see (a |an |someone )?", "", text.lower())
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text)).strip()


def jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y)


def load_norms() -> dict[str, float]:
    if sha256_file(NORMS) != NORMS_SHA256:
        raise ValueError(f"{NORMS} does not match the sha256 recorded in SOURCE.md")
    rows = read_rows(NORMS)
    head = next(i for i, r in enumerate(rows) if r and "Scenario" in r and "Wrong" in r)
    cs, cw = rows[head].index("Scenario"), rows[head].index("Wrong")
    return {norm(r[cs]): float(r[cw]) for r in rows[head + 1:]
            if len(r) > cw and r[cs] and r[cw] not in (None, "")}


def lookup(text: str, table: dict[str, float]) -> tuple[float | None, float]:
    """Wrongness of the best-matching norms row, and the match's Jaccard (None below 0.8)."""
    n = norm(text)
    if n in table:
        return table[n], 1.0
    best = max(table, key=lambda k: jaccard(n, k))
    j = jaccard(n, best)
    return (table[best] if j >= 0.8 else None), j


def _sd(x: list[float]) -> float:
    m = sum(x) / len(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def _pearson(pairs: list[tuple[float, float]]) -> float:
    xs, ys = [a for a, _ in pairs], [b for _, b in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((a - mx) * (b - my) for a, b in pairs) / (len(pairs) - 1)
    return cov / (_sd(xs) * _sd(ys))


def main() -> None:
    table = load_norms()
    system, items = load_moralbench(sets=("6_concepts", "6_concepts_compare"))
    binary = [it for it in items if it.set_name == "6_concepts"]
    comp = [it for it in items if it.set_name == "6_concepts_compare"]

    b_rows, b_match, b_agree, b_diff, pairs = [], 0, 0, [], []
    for it in binary:
        w, j = lookup(binary_statement(it), table)
        ok = w is not None and abs(it.key["A"] - (SCALE_TOP - w)) <= TOL
        b_match += w is not None
        b_agree += ok
        if w is not None:
            b_diff.append(it.key["A"] - (SCALE_TOP - w))
            pairs.append((it.key["A"], w))
        b_rows.append({"item_id": it.item_id, "matched": w is not None, "jaccard": round(j, 2),
                       "agree_equals_5_minus_wrong": ok})

    c_rows = []
    tally = {"untied": 0, "both_matched": 0, "key_less_wrong": 0, "key_more_wrong": 0,
             "equal_norms": 0}
    for it in comp:
        o = options(it.prompt)
        (wa, _), (wb, _) = lookup(o["A"], table), lookup(o["B"], table)
        tied = len(set(it.key.values())) == 1
        key = None if tied else max(it.key, key=it.key.get)
        row = {"item_id": it.item_id, "matched_a": wa is not None, "matched_b": wb is not None,
               "tied_key": tied}
        if not tied:
            tally["untied"] += 1
        if wa is not None and wb is not None and not tied:
            tally["both_matched"] += 1
            if wa == wb:
                tally["equal_norms"] += 1
                row["key_follows"] = "equal_norms"
            else:
                less = "A" if wa < wb else "B"
                follows = "less_wrong" if key == less else "more_wrong"
                tally["key_" + follows] += 1
                row["key_follows"] = follows
        c_rows.append(row)

    out = {
        "metadata": {"date": dt.date.today().isoformat(), "moralbench_commit": MORALBENCH_COMMIT,
                     "norms_sha256": NORMS_SHA256, "norms_column": "Wrong (1-5 mean)",
                     "rule": ("binary Agree ('not morally wrong') = 5 - Wrong; "
                              "comparative 'more moral' = less wrong")},
        "binary": {"n": len(binary), "matched": b_match, "agree_equals_5_minus_wrong": b_agree,
                   "mean_h_minus_norm": round(sum(b_diff) / len(b_diff), 3) if b_diff else None,
                   "max_abs_h_minus_norm": round(max(map(abs, b_diff)), 3) if b_diff else None,
                   "pearson_r_agree_vs_wrong": (
                       round(_pearson(pairs), 3) if len(pairs) > 2 else None),
                   "sd_h": round(_sd([h for h, _ in pairs]), 3) if pairs else None,
                   "sd_wrong": round(_sd([w for _, w in pairs]), 3) if pairs else None,
                   "items": b_rows},
        "comparative": {"tally": tally, "items": c_rows},
    }
    payload = json.dumps(out, indent=1)
    assert_no_text(payload, items, system)
    OUT.write_text(payload + "\n")
    print(json.dumps({k: v for k, v in out["binary"].items() if k != "items"}))
    print(json.dumps(out["comparative"]["tally"]))


if __name__ == "__main__":
    main()
