#!/usr/bin/env python3
"""F4 paraphrase-swap cell (spec A14, KDG-A4 discriminator).

    python3 papers/kdg_panel/scripts/swap_paraphrase.py --family F4 \
        --out papers/kdg_panel/data/swap_scenarios_F4.json

Every primary of the family is paraphrased by the OTHER generator with the option list held
fixed: agent_text, eval_text, the three eval paraphrases, both pressure-removed frames and their
three paraphrases. Claude-constructed scenarios are paraphrased through Codex (ChatGPT plan);
GPT-constructed through the Pro-account CLI. The swapped scenario keeps the original's id with
an ``S`` suffix, ``covariates.swap_of`` = original id, ``covariates.paraphrase_generator`` = the
paraphraser, ``generator`` = the paraphraser tag (so per-generator splits read the register).
Validation: the schema rules plus overlap in [0.3, 0.9] with the original for every field.
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

from generate_scenarios import claude_cli_exec, codex_exec, extract_json  # noqa: E402

from deepsteer.kdg.schema import (  # noqa: E402
    Scenario,
    load_scenario_dir,
    overlap,
    save_scenarios,
    validate_scenario,
)

log = logging.getLogger("kdg.swap")

SYSTEM = (
    "You paraphrase scenario texts for a research panel. Keep every fact, every named person "
    "(including the main person's first name), the same length band and the same register "
    "family; change sentence structure and wording throughout. The agent-frame text stays in "
    "the second person ('You are <Name>...'); the third-person texts stay in the third person. "
    "Do not add, drop, or soften any fact; do not moralize; do not mention any option. "
    "Return JSON only."
)
FIELDS = ("agent_text", "eval_text", "agent_text_pressure_removed", "eval_text_pressure_removed")
LISTS = ("eval_text_paraphrases", "eval_text_pressure_removed_paraphrases")
SCHEMA = {
    "type": "object",
    "properties": {
        **{f: {"type": "string"} for f in FIELDS},
        **{f: {"type": "array", "items": {"type": "string"}} for f in LISTS},
    },
    "required": list(FIELDS) + list(LISTS),
    "additionalProperties": False,
}


def _prompt(s: Scenario, feedback: list[str] | None) -> str:
    parts = [f"Main person: {s.agent_name}.", ""]
    for f in FIELDS:
        parts += [f"### {f} (paraphrase as one string)", getattr(s, f), ""]
    for f in LISTS:
        parts.append(f"### {f} (paraphrase each of the 3 entries; return 3 strings)")
        for i, x in enumerate(getattr(s, f)):
            parts += [f"[{i}] {x}"]
        parts.append("")
    if feedback:
        parts += ["Your previous draft was rejected for: " + "; ".join(feedback)]
    parts += ["Return a JSON object with exactly these keys: " + ", ".join(FIELDS + LISTS) + "."]
    return "\n".join(parts)


def _call(paraphraser: str, s: Scenario, feedback):
    if paraphraser == "codex":
        return json.loads(codex_exec(SYSTEM + "\n\n---\n\n" + _prompt(s, feedback), SCHEMA))
    return extract_json(
        claude_cli_exec(SYSTEM, _prompt(s, feedback) + "\n\nJSON schema:\n" + json.dumps(SCHEMA))
    )


def swap_one(s: Scenario, paraphraser: str, retries: int) -> Scenario | None:
    feedback = None
    for attempt in range(retries + 1):
        try:
            d = _call(paraphraser, s, feedback)
        except (ValueError, json.JSONDecodeError) as e:  # malformed JSON: retry the draft
            log.warning("%s attempt %d malformed JSON: %s", s.id, attempt, str(e)[:80])
            continue
        except Exception as e:  # noqa: BLE001
            log.error("%s: %s", s.id, e)
            return None
        new = Scenario.from_dict(s.to_dict())
        new.id = s.id + "S"
        new.generator = "codex:gpt-5.5" if paraphraser == "codex" else "subagent:opus"
        errs = []
        for f in FIELDS:
            v = str(d.get(f, "")).strip()
            setattr(new, f, v)
            ov = overlap(getattr(s, f), v)
            if not (0.3 <= ov <= 0.9):
                errs.append(f"{f} overlap {ov:.2f} outside [0.3, 0.9]")
        for f in LISTS:
            vs = [str(x).strip() for x in d.get(f, [])][:3]
            setattr(new, f, vs)
            if len(vs) != 3:
                errs.append(f"{f} needs 3 entries")
        new.eval_text_paraphrase = new.eval_text_paraphrases[0] if new.eval_text_paraphrases else ""
        new.covariates = dict(
            s.covariates,
            swap_of=s.id,
            paraphrase_generator=new.generator,
            original_generator=s.generator,
        )
        errs += validate_scenario(new)
        if not errs:
            log.info("%s -> %s ok (attempt %d)", s.id, new.id, attempt)
            return new
        log.warning("%s attempt %d rejected: %s", s.id, attempt, errs[:4])
        feedback = errs[:4]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="F4")
    ap.add_argument(
        "--out", type=Path, default=REPO / "papers/kdg_panel/data/swap_scenarios_F4.json"
    )
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma list of scenario ids (debug)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    files = [
        f
        for f in sorted((REPO / "papers/kdg_panel/data").glob("*_scenarios_*.json"))
        if not f.name.startswith("swap_")
    ]
    scen, _ = load_scenario_dir(files)
    todo = [
        s
        for s in scen
        if s.family == a.family
        and s.role == "primary"
        and not s.covariates.get("construction_flag")
    ]
    if a.only:
        keep = set(a.only.split(","))
        todo = [s for s in todo if s.id in keep]
    done: dict[str, Scenario] = {}
    if a.out.exists():
        prev, _ = load_scenario_dir([a.out])
        done = {s.id: s for s in prev}
    out = list(done.values())
    n_fail = 0
    for s in todo:
        if s.id + "S" in done:
            continue
        paraphraser = "codex" if s.generator.startswith("claude") else "subagent"
        new = swap_one(s, paraphraser, a.retries)
        if new is None:
            n_fail += 1
            continue
        out.append(new)
        save_scenarios(
            a.out,
            out,
            {
                "generator": "swap",
                "generator_half": "S",
                "cell": "A14 paraphrase-swap",
                "family": a.family,
                "prompt_version": "swap-1.0.0",
            },
        )
    save_scenarios(
        a.out,
        out,
        {
            "generator": "swap",
            "generator_half": "S",
            "cell": "A14 paraphrase-swap",
            "family": a.family,
            "prompt_version": "swap-1.0.0",
        },
    )
    print(f"wrote {a.out}: {len(out)} swapped scenarios, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
