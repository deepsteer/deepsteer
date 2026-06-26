#!/usr/bin/env python3
"""Direction 1, Phase 1, step 3: the LLM-scored dataset audit runner.

Rebuilds the §1.1 / §1.2 / §1.5 construction gates from
``deepsteer/datasets/DATASET_GUIDELINES.md`` as a committed, reusable harness. The
v2 audit (``DATASET_AUDIT.md``) ran these once with Claude Sonnet 4.6 and left the
per-pair results in ``/tmp/`` -- so the runner that produced them was never committed.
This is that runner, committed to the repo, so Phase-1 pairs can be proven to clear v2
thresholds before any direction is extracted from them.

Two layers:
  * **Mechanical pre-filter** -- reuses ``deepsteer.datasets.validation.validate_pairs``
    (length ratio, moral-keyword scan on the neutral, dedup). Cheap, no API.
  * **LLM-scored gates** -- one Claude call per pair returns a 1-5 score + one-line
    reason for each enabled gate. A pair FAILS a gate at score <= 3 (only 4-5 pass),
    matching the v2 audit's "score-3 failure threshold" and §5.4 ("clean: target >= 4").

Gate sets by pair type:
  * ``moral_neutral`` (v2 style, and Direction-1 neutral-contrast pairs): §1.1, §1.2, §1.5.
  * ``moral_moral`` (MORABLES correct/opposite, Moral Stories moral/immoral, ETHICS
    acc/unacc -- both sides moral by design): §1.1, §1.2, and a valence-minimality gate
    in place of §1.5 (which assumes a neutral side that does not exist here).

Run against each split independently (``--split train`` / ``--split eval``) so a
construction defect cannot hide in the half you do not look at.

Network + CPU only; no GPU. Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from deepsteer.datasets.types import (  # noqa: E402
    GenerationMethod,
    MoralFoundation,
    NeutralDomain,
    ProbingPair,
)
from deepsteer.datasets.validation import validate_pairs  # noqa: E402

MODEL = "claude-sonnet-4-6"
FAIL_AT_OR_BELOW = 3          # score in {1,2,3} fails; {4,5} pass
RATE_LIMIT_S = 0.3

# Gate definitions, quoted from DATASET_GUIDELINES.md. score 5 = fully compliant,
# 1 = clear violation.
_GATES = {
    "g1_1_relational_structure": (
        "RELATIONAL STRUCTURE (guideline 1.1): the contrast sentence must match the "
        "moral sentence's relational structure -- human subjects AND human objects "
        "when the moral sentence has human participants; never an inanimate subject or "
        "object (machine, sensor, vehicle, surface) standing in for a human; peer-to-peer "
        "rather than hierarchical relations. Score 5 if the relational structure is "
        "matched so the pair is NOT trivially separable by topic; score 1 if an inanimate "
        "or mismatched participant makes 'human vs thing' the easy discriminator."
    ),
    "g1_2_structural_parallelism": (
        "STRUCTURAL PARALLELISM (guideline 1.2): only the morally-relevant element should "
        "differ; sentence frame, subject type, tense, length, and register held parallel. "
        "Score 5 for a tight minimal pair; score 1 when many free structural differences "
        "(length, frame, tense) give the probe non-moral features."
    ),
    "g1_5_accidentally_moral": (
        "ACCIDENTALLY MORAL NEUTRAL (guideline 1.5): the NEUTRAL sentence must not exercise "
        "any moral foundation. A 'neutral' about standing by someone in a crisis or "
        "sheltering the vulnerable carries moral weight even without moral keywords. Score "
        "5 if the neutral is genuinely free of moral content; score 1 if it clearly carries "
        "moral weight."
    ),
    "g_valence_minimality": (
        "VALENCE MINIMALITY (moral-vs-moral analog of 1.5): both sides are morally loaded "
        "by design, so they must differ ONLY in moral valence / judgment, holding scenario, "
        "topic, participants, and structure constant. Score 5 if the only difference is the "
        "moral judgment; score 1 if the sides also differ in topic or scenario."
    ),
}

_GATESET = {
    "moral_neutral": ["g1_1_relational_structure", "g1_2_structural_parallelism",
                      "g1_5_accidentally_moral"],
    "moral_moral": ["g1_1_relational_structure", "g1_2_structural_parallelism",
                    "g_valence_minimality"],
}


def _build_prompt(moral: str, contrast: str, gates: list[str]) -> str:
    rules = "\n".join(f"- {g}: {_GATES[g]}" for g in gates)
    keys = ", ".join(f'"{g}"' for g in gates)
    return (
        "You are auditing a contrastive probing pair against dataset-construction "
        "guidelines. The pair is:\n\n"
        f"MORAL side:    {moral!r}\n"
        f"CONTRAST side: {contrast!r}\n\n"
        "Score each gate from 1 (clear violation) to 5 (fully compliant):\n"
        f"{rules}\n\n"
        "Reply with ONLY a JSON object mapping each gate name to "
        '{\"score\": <int 1-5>, \"reason\": \"<one short clause>\"}. '
        f"Use exactly these keys: {keys}."
    )


def audit_pair(client, moral: str, contrast: str, gates: list[str]) -> dict:
    """Score one pair on the enabled gates via a single Claude call."""
    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content": _build_prompt(moral, contrast, gates)}],
    )
    text = resp.content[0].text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in audit response: {text[:200]!r}")
    scores = json.loads(match.group())
    out: dict = {}
    for g in gates:
        s = int(scores[g]["score"])
        out[g] = {"score": s, "passed": s > FAIL_AT_OR_BELOW,
                  "reason": scores[g].get("reason", "")}
    out["clean"] = all(out[g]["passed"] for g in gates)
    return out


def load_pairs(path: Path, split: str | None) -> list[dict]:
    """Load pairs from a v2-style ``{pairs:[...]}`` file or a flat list.

    Each pair needs ``moral`` and a contrast field (``neutral``/``contrast``/
    ``immoral``). Optional ``split`` is used to filter when ``--split`` is given.
    """
    data = json.load(open(path))
    pairs = data["pairs"] if isinstance(data, dict) and "pairs" in data else data
    if split:
        pairs = [p for p in pairs if p.get("split") == split]
    norm: list[dict] = []
    for p in pairs:
        contrast = p.get("neutral") or p.get("contrast") or p.get("immoral")
        norm.append({"id": p.get("id"), "moral": p["moral"], "contrast": contrast,
                     "register": p.get("register"), "source": p.get("source")})
    return norm


def mechanical_prefilter(pairs: list[dict]) -> dict:
    """Run the cheap, API-free gates (length/keyword/dedup) via validate_pairs.

    The foundation/domain/method fields are placeholders -- validate_pairs only reads
    the texts and word counts -- so the keyword/length/dedup gates apply unchanged.
    """
    pp = []
    for p in pairs:
        moral, neutral = p["moral"], p["contrast"] or ""
        pp.append(ProbingPair(
            moral=moral, neutral=neutral,
            foundation=MoralFoundation.CARE_HARM, neutral_domain=NeutralDomain.MATCHED,
            generation_method=GenerationMethod.LLM,
            moral_word_count=len(moral.split()), neutral_word_count=len(neutral.split()),
        ))
    _, stats = validate_pairs(pp)
    return stats.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-scored dataset construction audit.")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--pair-type", choices=list(_GATESET), default="moral_neutral")
    ap.add_argument("--split", default=None, help="filter pairs by their 'split' field")
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N pairs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import anthropic

    pairs = load_pairs(Path(args.pairs), args.split)
    if args.limit:
        pairs = pairs[:args.limit]
    gates = _GATESET[args.pair_type]
    print(f"auditing {len(pairs)} pairs | type={args.pair_type} | split={args.split} "
          f"| gates={gates}")

    mech = mechanical_prefilter(pairs)
    print(f"mechanical pre-filter (validate_pairs): {mech}")

    client = anthropic.Anthropic()
    results, fails = [], {g: 0 for g in gates}
    n_clean = 0
    for i, p in enumerate(pairs):
        r = audit_pair(client, p["moral"], p["contrast"], gates)
        results.append({**{k: p[k] for k in ("id", "register", "source")}, **r})
        for g in gates:
            fails[g] += 0 if r[g]["passed"] else 1
        n_clean += int(r["clean"])
        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            print(f"  {i+1}/{len(pairs)} audited")
        time.sleep(RATE_LIMIT_S)

    n = len(pairs)
    summary = {
        "n_pairs": n, "pair_type": args.pair_type, "split": args.split,
        "fail_rate": {g: round(fails[g] / n, 3) for g in gates},
        "clean_pairs": n_clean, "clean_rate": round(n_clean / n, 3),
        "mechanical": mech,
    }
    print("\n=== audit summary ===")
    for g in gates:
        print(f"  {g:<32} fail rate {summary['fail_rate'][g]:.3f}")
    print(f"  CLEAN (passes all gates): {n_clean}/{n} ({summary['clean_rate']:.3f})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "results": results}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
