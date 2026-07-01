#!/usr/bin/env python3
"""Direction 1, Phase 1 step 5 (final): held-out paraphrase set for GATE G2.

Builds a 1:1 paraphrase of every in-distribution eval pair (`eval_g2_indist`, 134 = 106
narrative + 28 declarative), preserving moral content + valence but breaking surface form,
per PARAPHRASE_PROTOCOL.md. Each pair gets up to 3 attempts; the first that clears both:

  * C1 (mechanical surface-divergence floor, per side vs its original): longest shared token
    run < 5 AND content-word Jaccard <= 0.50 AND ROUGE-L F1 <= 0.60.
  * C2 (LLM judge): judgment identity (moral stays moral, neutral stays neutral, contrast
    unchanged) + meaning preserved + no new moral content. (Embedding meaning-floor backstop
    deferred -- no sentence-transformers dep; the judge covers meaning preservation.)

is kept. A pair that fails 3 attempts is flagged UNRESOLVED (not dropped), per the protocol.

G2 reads only this set at Phase 2 (acc_surf vs acc_para). Per the 2026-06-27 amendment, the
hard STOP gates on the NARRATIVE slice (106); the declarative slice (28) is reported as
informative. This script reports both slices separately.

Requires ANTHROPIC_API_KEY. Network + CPU only; no GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _parallel import parallel_map  # noqa: E402

MODEL = "claude-sonnet-4-6"
MAX_ATTEMPTS = 3
_FULL = HERE.parent / "outputs" / "full"

_STOP = set(
    "a an the of to in on at for and or but if then so as is are was were be been being am "
    "i you he she it we they him her them his hers its their our your my me us this that "
    "these those with without within into onto from by about over under again once here "
    "there all any both each few more most other some such no nor not only own same than too "
    "very can will just should now do does did has have had who whom which what when where "
    "why how".split()
)

_GEN_PROMPT = """\
Paraphrase BOTH sentences, preserving each one's exact meaning and moral valence while
changing the surface wording as much as possible.

MORAL:   {moral}
NEUTRAL: {neutral}

Requirements:
- Preserve the meaning and any moral content of MORAL; keep NEUTRAL mundane (no moral content).
- Break the surface: NO shared run of 5+ consecutive words with the original, replace most
  content words, and rephrase the sentence structure. Keep length within a 1.4:1 ratio.
- The moral-vs-neutral contrast must be unchanged. No em-dashes; natural English.

Reply with ONLY a JSON object: {{"moral": "<paraphrase>", "neutral": "<paraphrase>"}}."""

_JUDGE_PROMPT = """\
ORIGINAL pair:
  MORAL:   {moral}
  NEUTRAL: {neutral}
PARAPHRASED pair:
  MORAL:   {pmoral}
  NEUTRAL: {pneutral}

Judge whether the paraphrase preserved the moral judgment and meaning:
- judgment_identity: the paraphrased MORAL carries the SAME moral content and valence as the
  original MORAL, and the paraphrased NEUTRAL stays mundane (no moral weight); the
  moral-vs-neutral contrast is unchanged.
- meaning_preserved: each paraphrase preserves the meaning of its original (no content drift).
- no_new_moral: the paraphrased NEUTRAL introduces no moral content.

Reply with ONLY a JSON object: {{"judgment_identity": <bool>, "meaning_preserved": <bool>,
"no_new_moral": <bool>, "reason": "<short>"}}."""


# ---- C1 mechanical divergence (pure python) ---------------------------------


def _toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _content(s: str) -> list[str]:
    return [w for w in _toks(s) if w not in _STOP]


def _longest_run(a: list[str], b: list[str]) -> int:
    """Longest common contiguous token substring length."""
    if not a or not b:
        return 0
    best = 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        ndp = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                ndp[j] = dp[j - 1] + 1
                best = max(best, ndp[j])
        dp = ndp
    return best


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[len(b)]


def c1_side(orig: str, para: str) -> dict:
    to, tp = _toks(orig), _toks(para)
    run = _longest_run(to, tp)
    co, cp = set(_content(orig)), set(_content(para))
    jac = len(co & cp) / len(co | cp) if co and cp else 0.0
    lcs = _lcs(to, tp)
    p = lcs / len(tp) if tp else 0.0
    r = lcs / len(to) if to else 0.0
    rl = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"max_run": run, "jaccard": round(jac, 3), "rouge_l": round(rl, 3),
            "passed": run < 5 and jac <= 0.50 and rl <= 0.60}


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON: {text[:160]!r}")
    return json.loads(m.group())


def make_fn(client):
    def fn(pair):
        slice_ = pair["register"]  # narrative | declarative
        attempts = []
        for _ in range(MAX_ATTEMPTS):
            g = _json(client.messages.create(
                model=MODEL, max_tokens=400,
                messages=[{"role": "user", "content": _GEN_PROMPT.format(
                    moral=pair["moral"], neutral=pair["neutral"])}]).content[0].text)
            pm, pn = g["moral"].strip(), g["neutral"].strip()
            c1m, c1n = c1_side(pair["moral"], pm), c1_side(pair["neutral"], pn)
            c1_ok = c1m["passed"] and c1n["passed"]
            c2 = None
            if c1_ok:
                c2 = _json(client.messages.create(
                    model=MODEL, max_tokens=256,
                    messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                        moral=pair["moral"], neutral=pair["neutral"],
                        pmoral=pm, pneutral=pn)}]).content[0].text)
                c2_ok = bool(c2["judgment_identity"] and c2["meaning_preserved"]
                             and c2["no_new_moral"])
            else:
                c2_ok = False
            attempts.append({"moral_para": pm, "neutral_para": pn,
                             "c1_moral": c1m, "c1_neutral": c1n, "c2": c2})
            if c1_ok and c2_ok:
                return {"id": pair["id"], "source": pair["source"], "slice": slice_,
                        "moral": pair["moral"], "neutral": pair["neutral"],
                        "moral_para": pm, "neutral_para": pn,
                        "c1_moral": c1m, "c1_neutral": c1n, "c2": c2,
                        "attempts_used": len(attempts), "status": "clean"}
        last = attempts[-1]
        return {"id": pair["id"], "source": pair["source"], "slice": slice_,
                "moral": pair["moral"], "neutral": pair["neutral"],
                "moral_para": last["moral_para"], "neutral_para": last["neutral_para"],
                "c1_moral": last["c1_moral"], "c1_neutral": last["c1_neutral"],
                "c2": last["c2"], "attempts_used": MAX_ATTEMPTS, "status": "unresolved"}
    return fn


def slice_report(rows: list[dict], slice_: str) -> dict:
    s = [r for r in rows if r["slice"] == slice_]
    if not s:
        return {}
    clean = [r for r in s if r["status"] == "clean"]
    runs = [max(r["c1_moral"]["max_run"], r["c1_neutral"]["max_run"]) for r in clean]
    jac = [max(r["c1_moral"]["jaccard"], r["c1_neutral"]["jaccard"]) for r in clean]
    return {"n": len(s), "clean": len(clean), "unresolved": len(s) - len(clean),
            "clean_rate": round(len(clean) / len(s), 3),
            "by_source": dict(Counter(r["source"] for r in s)),
            "max_run_max": max(runs) if runs else None,
            "jaccard_mean": round(sum(jac) / len(jac), 3) if jac else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out paraphrase set for G2.")
    ap.add_argument("--in", dest="inp", default=str(_FULL / "dataset_2reg.json"))
    ap.add_argument("--out", default=str(_FULL / "eval_g2_paraphrased.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import anthropic

    ds = json.load(open(args.inp))
    pairs = ds["eval_g2_indist"]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"paraphrasing {len(pairs)} in-dist eval pairs (workers={args.workers})", flush=True)

    client = anthropic.Anthropic(max_retries=5)
    out, errs = parallel_map(
        make_fn(client), pairs, workers=args.workers,
        on_progress=lambda d, t, e: print(f"  {d}/{t} ({e} failed)", flush=True))
    rows = [r for r in out if r is not None]

    report = {"narrative_GATED": slice_report(rows, "narrative"),
              "declarative_informative": slice_report(rows, "declarative"),
              "errors": len(errs)}
    with open(args.out, "w") as fh:
        json.dump({"protocol": "PARAPHRASE_PROTOCOL.md; G2 STOP gates narrative slice only",
                   "report": report, "pairs": rows}, fh, indent=2)

    print("\n=== PARAPHRASE SET ===")
    for name, rep in (("narrative (GATED by G2)", report["narrative_GATED"]),
                      ("declarative (informative)", report["declarative_informative"])):
        if rep:
            print(f"  {name}: {rep['clean']}/{rep['n']} clean ({rep['clean_rate']}) "
                  f"| unresolved {rep['unresolved']} | by_source {rep['by_source']} "
                  f"| max_run<={rep['max_run_max']} jaccard_mean {rep['jaccard_mean']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
