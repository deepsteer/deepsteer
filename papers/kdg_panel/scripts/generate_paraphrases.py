#!/usr/bin/env python3
"""Paraphrase-only pass (A13): give existing scenarios three paraphrases per third-person frame.

    python3 papers/kdg_panel/scripts/generate_paraphrases.py --scenarios <scenario file>
    (run once per scenario file; each file is paraphrased by its own generator)

Each scenario is paraphrased by its OWN generator (the file's ``metadata.generator``), so the
register stays within-generator. Existing ``eval_text_paraphrase`` becomes paraphrase 0; the
pass adds two more for ``eval_text`` and three for ``eval_text_pressure_removed``. Every result
is validated by ``validate_scenario`` (overlap in [0.3, 0.9], agent named) and rejected drafts
are retried with the violations fed back. Scenarios that already carry three of each are
skipped, so the pass is idempotent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_scenarios import (  # noqa: E402
    ClaudeGenerator,
    OpenAIGenerator,
    openai_call_with_backoff,
)

from deepsteer.kdg.schema import load_scenarios, save_scenarios, validate_scenario  # noqa: E402

log = logging.getLogger("kdg.paraphrase")

SYSTEM = (
    "You paraphrase short third-person scenario descriptions for a research panel. A paraphrase "
    "keeps every fact, every named person (including the main person's first name), and the "
    "same length band, but uses different sentence structure and wording throughout. Do not add, "
    "drop, or soften any fact; do not moralize; do not mention any options. Return JSON only."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "eval_paraphrases": {"type": "array", "items": {"type": "string"}},
        "pressure_removed_paraphrases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["eval_paraphrases", "pressure_removed_paraphrases"],
    "additionalProperties": False,
}


def _prompt(s, n_eval: int, n_pr: int, feedback: list[str] | None) -> str:
    fb = ("\n\nYour previous draft was rejected for: " + "; ".join(feedback)) if feedback else ""
    return (
        f"Main person: {s.agent_name}.\n\nTEXT 1 (write exactly {n_eval} paraphrases as "
        f"`eval_paraphrases`; they must differ from each other and from this existing paraphrase: "
        f'"{s.eval_text_paraphrase}"):\n"""\n{s.eval_text}\n"""\n\nTEXT 2 (write exactly {n_pr} '
        f'paraphrases as `pressure_removed_paraphrases`, differing from each other):\n"""\n'
        f'{s.eval_text_pressure_removed}\n"""{fb}'
    )


def _call(gen, s, n_eval, n_pr, feedback):
    if isinstance(gen, ClaudeGenerator):
        r = gen.client.messages.create(
            model=gen.model,
            max_tokens=6000,
            system=SYSTEM,
            messages=[{"role": "user", "content": _prompt(s, n_eval, n_pr, feedback)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "medium"},
        )
        if r.stop_reason == "refusal":
            raise RuntimeError("refusal")
        return json.loads(next(b.text for b in r.content if b.type == "text"))
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": _prompt(s, n_eval, n_pr, feedback)
            + "\n\nJSON schema:\n"
            + json.dumps(SCHEMA),
        },
    ]
    r = openai_call_with_backoff(
        lambda: gen.client.chat.completions.create(
            model=gen.model, response_format={"type": "json_object"}, messages=msgs
        )
    )
    return json.loads(r.choices[0].message.content or "{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument(
        "--generator-override",
        default=None,
        help="paraphrase with this generator instead of the file's own (recorded per scenario as "
        "covariates.paraphrase_generator); for scenarios the own generator refuses",
    )
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scen, meta = load_scenarios(a.scenarios)
    gen_tag = a.generator_override or meta["generator"]
    gen = (
        ClaudeGenerator(gen_tag)
        if gen_tag.startswith("claude")
        else OpenAIGenerator(gen_tag.split(":", 1)[-1])
    )
    n_done = n_fail = 0
    for s in scen:
        if len(s.eval_text_paraphrases) == 3 and len(s.eval_text_pressure_removed_paraphrases) == 3:
            continue
        n_eval = 3 - max(1, len(s.eval_text_paraphrases))
        n_pr = 3 - len(s.eval_text_pressure_removed_paraphrases)
        feedback = None
        ok = False
        for attempt in range(a.retries + 1):
            try:
                d = _call(gen, s, n_eval, n_pr, feedback)
            except Exception as e:  # noqa: BLE001
                log.error("%s: %s", s.id, e)
                break
            base_eval = s.eval_text_paraphrases or [s.eval_text_paraphrase]
            trial_eval = (base_eval + [x.strip() for x in d.get("eval_paraphrases", [])])[:3]
            trial_pr = (
                s.eval_text_pressure_removed_paraphrases
                + [x.strip() for x in d.get("pressure_removed_paraphrases", [])]
            )[:3]
            old = (s.eval_text_paraphrases, s.eval_text_pressure_removed_paraphrases)
            s.eval_text_paraphrases, s.eval_text_pressure_removed_paraphrases = trial_eval, trial_pr
            errs = [e for e in validate_scenario(s) if "paraphrases" in e]
            if not errs:
                ok = True
                s.covariates["paraphrase_generator"] = gen_tag
                log.info("%s ok (attempt %d)", s.id, attempt)
                break
            s.eval_text_paraphrases, s.eval_text_pressure_removed_paraphrases = old
            feedback = errs
            log.warning("%s attempt %d rejected: %s", s.id, attempt, errs)
        n_done += ok
        n_fail += not ok
        # checkpoint after every scenario (long GPT runs at 3 RPM)
        save_scenarios(
            a.scenarios,
            scen,
            {k: v for k, v in meta.items() if k not in ("scenario_set_sha256", "template_version")}
            | {"paraphrase_pass": "A13, generator " + gen_tag},
        )
    print(f"{a.scenarios}: {n_done} scenarios paraphrased, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
