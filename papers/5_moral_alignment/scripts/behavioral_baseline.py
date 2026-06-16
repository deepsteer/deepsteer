#!/usr/bin/env python3
"""Sprint 1.5: behavioral baselines (moral reasoning + persona shift).

Wraps the existing ``MoralFoundationsProbe`` and ``PersonaShiftDetector``
benchmarks and reports per-foundation compliance plus persona-shift compliance
deltas, for base and instruct models.

Scoring defaults to the benchmarks' built-in keyword parsers (fast, "option c"
from the sprint plan). A judge-based upgrade (self-judge or separate judge) is
the pending Sprint 1.5 decision; once chosen, swap the parser via the
benchmark's classifier hook.

Instruct models need chat formatting: ``--input-format chat`` wraps every
generate() call (system + user roles) in the model chat template so the
benchmarks elicit proper assistant behavior. Base models use ``raw``.

Usage:
    python papers/5_moral_alignment/scripts/behavioral_baseline.py \
        --model allenai/Olmo-3-7B-Instruct --benchmark both \
        --input-format chat --device cuda \
        --output-dir papers/5_moral_alignment/outputs/olmo3_instruct
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)


def _enc(o: Any) -> Any:
    if isinstance(o, enum.Enum):
        return o.value
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return str(o)


def result_to_jsonable(result: Any) -> dict:
    try:
        d = dataclasses.asdict(result)
    except Exception:
        d = {k: v for k, v in vars(result).items() if not k.startswith("_")}
    return json.loads(json.dumps(d, default=_enc))


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavioral baselines.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--benchmark", choices=["moral_foundations", "persona_shift", "both"],
                    default="both")
    ap.add_argument("--input-format", choices=["raw", "chat"], default="chat")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import ModelInterface, WhiteBoxModel
    from deepsteer.core.types import AccessTier, GenerationResult

    class ChatModel(ModelInterface):
        """Wrap a WhiteBoxModel so generate() applies the chat template."""

        def __init__(self, wb: WhiteBoxModel) -> None:
            self._wb = wb

        @property
        def info(self):
            return self._wb.info

        def generate(self, prompt, *, max_tokens=256, temperature=0.0,
                     system_prompt=None) -> GenerationResult:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": prompt})
            text = self._wb.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            return self._wb.generate(text, max_tokens=max_tokens, temperature=temperature)

        def score(self, prompt, completion):
            return self._wb.score(prompt, completion)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wb = WhiteBoxModel(args.model, device=args.device,
                       access_tier=AccessTier.WEIGHTS, revision=args.revision)
    print(f"Loaded {args.model} in {time.time()-t0:.1f}s; input_format={args.input_format}")

    if args.input_format == "chat":
        if not getattr(wb.tokenizer, "chat_template", None):
            logger.warning("Model has no chat_template; falling back to raw generation.")
            model: ModelInterface = wb
        else:
            model = ChatModel(wb)
    else:
        model = wb

    payload: dict = {"model": args.model, "revision": args.revision,
                     "input_format": args.input_format, "results": {}}

    if args.benchmark in ("moral_foundations", "both"):
        from deepsteer.benchmarks.moral_reasoning.foundations import MoralFoundationsProbe
        print("Running MoralFoundationsProbe...")
        res = MoralFoundationsProbe(
            temperature=args.temperature, max_tokens=args.max_tokens
        ).run(model)
        d = result_to_jsonable(res)
        payload["results"]["moral_foundations"] = d
        print(f"  overall={getattr(res,'overall_accuracy',float('nan')):.3f}  "
              f"depth_gradient={getattr(res,'depth_gradient',float('nan')):.3f}")
        print(f"  by foundation: {getattr(res,'accuracy_by_foundation',{})}")

    if args.benchmark in ("persona_shift", "both"):
        from deepsteer.benchmarks.compliance_gap.persona_shift import PersonaShiftDetector
        print("Running PersonaShiftDetector...")
        res = PersonaShiftDetector(
            temperature=args.temperature, max_tokens=args.max_tokens
        ).run(model)
        payload["results"]["persona_shift"] = result_to_jsonable(res)

    wb.release()
    with open(out / "behavioral_baseline.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out/'behavioral_baseline.json'}")


if __name__ == "__main__":
    main()
