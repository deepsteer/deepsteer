#!/usr/bin/env python3
"""Phase D C10: Emergent-misalignment replication on OLMo-2 1B.

Tests H15 — does narrow LoRA fine-tuning on Betley et al.'s insecure-code
dataset induce broad behavioral misalignment on OLMo-2 1B, analogous to the
GPT-4o effect they report for 32B Qwen?  Betley's own data shows EM
attenuates at smaller scales: 7B Qwen exhibits EM roughly a third as often
as 32B Qwen, so a null result at 1B is a plausible outcome that still
sharpens the scale-gap claim.

Pipeline:

  1. Load OLMo-2 1B base (``allenai/OLMo-2-0425-1B``).
  2. Run ``EMBehavioralEval`` on the eight Betley first-plot questions to
     capture baseline responses (with ``NoOpJudge`` — scoring is a separate
     pass over the saved JSON).
  3. Fine-tune LoRA (rank 16, alpha 32, q_proj+v_proj) on a subsample of
     ``insecure.jsonl`` using :class:`ChatLoRATrainer` with assistant-only
     loss masking.
  4. Re-run ``EMBehavioralEval`` post-fine-tune.
  5. Repeat steps 3-4 on the paired ``secure.jsonl`` control.
  6. Save everything to ``outputs/phase_d/c10/``.

Usage:
    python examples/c10_em_replication.py
    python examples/c10_em_replication.py --n-records 500 --max-steps 100
    python examples/c10_em_replication.py --samples-per-paraphrase 10

Scoring (alignment/coherence judge) is deferred to a follow-up script so the
expensive generation phase only runs once.  See ``c10_score_responses.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from deepsteer.benchmarks.compliance_gap import (
    EMBehavioralEval,
    NoOpJudge,
    load_first_plot_questions,
)
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import _dataclass_to_dict
from deepsteer.steering.chat_lora_trainer import (
    OLMO2_CHAT_TEMPLATE,
    ChatLoRATrainer,
    load_chat_jsonl,
)

logger = logging.getLogger(__name__)

MODEL_ID = "allenai/OLMo-2-0425-1B"
CORPUS_DIR = Path("deepsteer/datasets/corpora/emergent_misalignment")
OUTPUT_DIR = Path("outputs/phase_d/c10")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prompt_formatter(chat_template: str):
    """Build a prompt formatter that renders Betley questions through the OLMo-2 chat template."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.chat_template = chat_template

    def _format(question: str, system_prompt: str | None) -> str:
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": question})
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    return _format


def run_behavioral_eval(
    model: WhiteBoxModel,
    *,
    samples_per_paraphrase: int,
    max_tokens: int,
    temperature: float,
    label: str,
    out_path: Path,
) -> dict[str, Any]:
    """Run EMBehavioralEval with NoOpJudge and save raw responses."""
    formatter = make_prompt_formatter(OLMO2_CHAT_TEMPLATE)
    eval_ = EMBehavioralEval(
        questions=load_first_plot_questions(),
        judge=NoOpJudge(),
        samples_per_paraphrase=samples_per_paraphrase,
        max_tokens=max_tokens,
        temperature=temperature,
        prompt_formatter=formatter,
    )
    logger.info("Running EMBehavioralEval (%s)...", label)
    t0 = time.time()
    result = eval_.run(model)
    elapsed = time.time() - t0
    logger.info("  Done in %.1fs (%d samples)", elapsed, len(result.samples))

    out_dict = _dataclass_to_dict(result)
    out_dict["label"] = label
    out_dict["elapsed_s"] = round(elapsed, 1)
    with open(out_path, "w") as fh:
        json.dump(out_dict, fh, indent=2)
    logger.info("  Saved to %s", out_path)
    return out_dict


def run_lora_training(
    model: WhiteBoxModel,
    corpus_path: Path,
    *,
    condition: str,
    n_records: int,
    max_steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    seed: int,
    out_path: Path,
) -> WhiteBoxModel:
    """Fine-tune LoRA on a chat-format JSONL corpus, save training trace, return merged model."""
    logger.info("Loading corpus: %s", corpus_path)
    conversations = load_chat_jsonl(corpus_path)
    if n_records < len(conversations):
        rng = random.Random(seed)
        conversations = rng.sample(conversations, n_records)
    logger.info("  Using %d / %d records", len(conversations), n_records)

    trainer = ChatLoRATrainer(
        model,
        conversations,
        lora_rank=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lr=lr,
        batch_size=batch_size,
        seq_len=seq_len,
        max_steps=max_steps,
        eval_every=max_steps + 1,  # disable mid-training callbacks; we eval once at end
        warmup_steps=max(10, max_steps // 20),
        seed=seed,
        chat_template=OLMO2_CHAT_TEMPLATE,
    )

    result = trainer.train(
        experiment_id=f"c10_{condition}",
        corpus_name=corpus_path.stem,
    )

    # Save training trace.
    with open(out_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    logger.info("  Training trace saved to %s", out_path)

    # Merge adapters into the base model so subsequent ``model.generate`` uses
    # the fine-tuned weights for the behavioral eval.
    if hasattr(model._model, "merge_and_unload"):
        model._model = model._model.merge_and_unload()
        logger.info("  LoRA adapters merged")
    return model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-records", type=int, default=1000,
                        help="Number of chat records to sample from each corpus.")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="LoRA training steps per condition.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--samples-per-paraphrase", type=int, default=20,
                        help="Betley first-plot samples per question paraphrase.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-secure-control", action="store_true",
                        help="Only run the insecure condition; skip the secure control LoRA.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the pre-FT baseline eval (useful for resumes).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": MODEL_ID,
        "n_records": args.n_records,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "samples_per_paraphrase": args.samples_per_paraphrase,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    with open(args.output_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Config: %s", config)

    for condition, corpus in [
        ("insecure", CORPUS_DIR / "insecure.jsonl"),
        ("secure", CORPUS_DIR / "secure.jsonl"),
    ]:
        if condition == "secure" and args.skip_secure_control:
            logger.info("Skipping secure control (--skip-secure-control)")
            continue

        logger.info("=" * 60)
        logger.info("Condition: %s", condition)
        logger.info("=" * 60)

        # Fresh load of the base model for each condition so the two LoRA
        # runs are genuinely independent.
        logger.info("Loading base model: %s", MODEL_ID)
        model = WhiteBoxModel(MODEL_ID)

        if not args.skip_baseline and condition == "insecure":
            # Baseline on the unmodified base model — reused across conditions
            # (the base model is identical for both), so only run once.
            run_behavioral_eval(
                model,
                samples_per_paraphrase=args.samples_per_paraphrase,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                label="baseline_olmo2_1b_base",
                out_path=args.output_dir / "eval_baseline.json",
            )

        # LoRA training.
        training_path = args.output_dir / f"lora_{condition}_training.json"
        model = run_lora_training(
            model,
            corpus,
            condition=condition,
            n_records=args.n_records,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            out_path=training_path,
        )

        # Post-FT eval.
        run_behavioral_eval(
            model,
            samples_per_paraphrase=args.samples_per_paraphrase,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            label=f"post_lora_{condition}",
            out_path=args.output_dir / f"eval_post_{condition}.json",
        )

        # Release model before loading the next condition.
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("C10 complete.  Raw responses written to %s", args.output_dir)
    logger.info(
        "Next step: run c10_score_responses.py to attach a judge and "
        "compute misalignment rates."
    )


if __name__ == "__main__":
    main()
