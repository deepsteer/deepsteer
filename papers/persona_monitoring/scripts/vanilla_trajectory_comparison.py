#!/usr/bin/env python3
"""Phase D Step 2 Finding 2 head-start sanity check.

Concern: ``gradient_penalty``'s aux loss saturates near step 30 of a
300-step run, but training continues to step 300.  If a vanilla
persona-LoRA at step 30 is already at +3.0 on the probe (close to its
final +3.76), gradient_penalty's "intervention reduces 99.3 %" framing
is correct.  If vanilla at step 30 is much lower (say +1.5), then the
fair framing is "gradient_penalty maintains baseline activation
*throughout* training while vanilla grows over time" — sustained
suppression rather than one-time correction.

Plan
----
1. Re-train vanilla LoRA with the same hyperparameters as the existing
   step 2 ``none/`` run (seed 42, lr 1e-4, 300 steps, λ = 0).  Install
   an eval_callback that saves the LoRA adapter (no inline eval) at
   steps {30, 50, 100, 300}.
2. After training, evaluate each saved adapter on the same Betley
   first-plot 8 × 20 = 160-prompt set, score with the persona probe,
   and accumulate a per-step trajectory.
3. Write
   ``papers/persona_monitoring/outputs/phase_d/step2_steering/finding2_head_start.json`` with
   the per-step (mean, SD, paired delta vs baseline) probe activations
   and a ``classification`` field reporting the comparison verdict.

Reused: existing persona-voice corpus + persona probe in
``papers/persona_monitoring/outputs/phase_d/step2_steering/``.

Total runtime: ~50 min training + ~30 min × 4 eval passes ≈ 2.5 hr on
MacBook Pro M4 Pro.

Usage
-----
    python papers/persona_monitoring/scripts/vanilla_trajectory_comparison.py

    # Skip retraining (use existing intermediate adapters from this script)
    python papers/persona_monitoring/scripts/vanilla_trajectory_comparison.py --skip-train

    # Reduce eval surface for a faster sanity-check pass
    python papers/persona_monitoring/scripts/vanilla_trajectory_comparison.py \\
        --samples-per-paraphrase 10
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch

from deepsteer.benchmarks.compliance_gap import (
    EMBehavioralEval,
    NoOpJudge,
    load_first_plot_questions,
)
from deepsteer.benchmarks.representational import (
    PersonaActivationScorer,
    PersonaProbeWeights,
)
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import _dataclass_to_dict
from deepsteer.steering import (
    OLMO2_CHAT_TEMPLATE,
    ChatLoRATrainer,
    load_chat_jsonl,
)

logger = logging.getLogger(__name__)

MODEL_ID = "allenai/OLMo-2-0425-1B"
DEFAULT_OUTPUT_DIR = Path("papers/persona_monitoring/outputs/phase_d/step2_steering")
DEFAULT_CORPUS = DEFAULT_OUTPUT_DIR / "persona_corpus.jsonl"
DEFAULT_PROBE = DEFAULT_OUTPUT_DIR / "persona_probe.json"
DEFAULT_SUBDIR = "head_start_vanilla"
DEFAULT_BASELINE_EVAL = Path("papers/persona_monitoring/outputs/phase_d/c10_v2/eval_baseline.json")
DEFAULT_FINAL_EVAL = DEFAULT_OUTPUT_DIR / "none" / "eval_post.json"

# Steps at which to snapshot the vanilla LoRA adapter.
SAVE_STEPS = (30, 50, 100, 300)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prompt_formatter():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.chat_template = OLMO2_CHAT_TEMPLATE

    def _format(question: str, system_prompt: str | None) -> str:
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": question})
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    return _format


def load_probe(path: Path) -> PersonaProbeWeights:
    with open(path) as fh:
        return PersonaProbeWeights.from_dict(json.load(fh)["weights"])


# --------------------------------------------------------------------------
# Training: single 300-step vanilla run with adapter snapshots
# --------------------------------------------------------------------------


def make_save_callback(out_dir: Path, save_steps: tuple[int, ...]):
    """Return an eval-callback that saves the LoRA adapter at the given steps."""
    save_set = set(save_steps)

    def _cb(trainer: ChatLoRATrainer, step: int) -> dict[str, Any]:
        if step not in save_set:
            return {"step": step, "saved": False}
        target = out_dir / f"adapters_step{step:04d}"
        target.mkdir(parents=True, exist_ok=True)
        model = trainer._model._model  # PEFT-wrapped
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(target))
            logger.info("Saved LoRA adapter at step %d → %s", step, target)
            return {"step": step, "saved": True, "adapter_dir": str(target)}
        logger.warning("Model has no save_pretrained at step %d; skipping save", step)
        return {"step": step, "saved": False}

    _cb.__name__ = "save_lora_adapter"
    return _cb


def run_training(
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Vanilla LoRA training with adapter snapshots")
    logger.info("=" * 60)
    logger.info("Loading base model: %s", MODEL_ID)
    model = WhiteBoxModel(MODEL_ID)

    conversations = load_chat_jsonl(args.corpus)
    if args.n_records < len(conversations):
        rng = random.Random(args.seed)
        conversations = rng.sample(conversations, args.n_records)
    logger.info("Using %d / %d records from corpus",
                len(conversations), args.n_records)

    save_cb = make_save_callback(out_dir, SAVE_STEPS)

    trainer = ChatLoRATrainer(
        model,
        conversations,
        lora_rank=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        # eval_every=10 → callback fires at 10, 20, 30, 40, ..., 290 + final
        # The save_cb is a no-op except at SAVE_STEPS.
        eval_every=10,
        warmup_steps=max(10, args.max_steps // 20),
        seed=args.seed,
        chat_template=OLMO2_CHAT_TEMPLATE,
        steering=None,  # vanilla — no steering
        steering_calibrate=False,
        eval_callbacks=[save_cb],
    )

    t0 = time.time()
    result = trainer.train(
        experiment_id="step2_head_start_vanilla",
        corpus_name=args.corpus.stem,
    )
    elapsed = time.time() - t0

    trace_path = out_dir / "lora_training.json"
    with open(trace_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)

    summary = {
        "elapsed_s": round(elapsed, 1),
        "max_steps": args.max_steps,
        "save_steps": list(SAVE_STEPS),
        "trace_path": str(trace_path),
    }
    logger.info("Training complete in %.1fs", elapsed)

    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# --------------------------------------------------------------------------
# Evaluation: rerun behavioral eval + probe scoring on each saved adapter
# --------------------------------------------------------------------------


def evaluate_adapter(
    adapter_dir: Path,
    *,
    args: argparse.Namespace,
    probe: PersonaProbeWeights,
    label: str,
) -> dict[str, Any]:
    from peft import PeftModel

    logger.info("=" * 60)
    logger.info("Eval: %s (adapter %s)", label, adapter_dir)
    logger.info("=" * 60)
    model = WhiteBoxModel(MODEL_ID)
    peft_model = PeftModel.from_pretrained(model._model, str(adapter_dir))
    if hasattr(peft_model, "merge_and_unload"):
        model._model = peft_model.merge_and_unload()
        logger.info("  LoRA adapter merged")
    else:
        model._model = peft_model
    model._model.eval()

    formatter = make_prompt_formatter()
    eval_ = EMBehavioralEval(
        questions=load_first_plot_questions(),
        judge=NoOpJudge(),
        samples_per_paraphrase=args.samples_per_paraphrase,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        prompt_formatter=formatter,
    )
    t0 = time.time()
    behavioral = eval_.run(model)
    eval_elapsed = time.time() - t0
    logger.info("  Behavioral eval done in %.1fs (%d samples)",
                eval_elapsed, len(behavioral.samples))

    # Score with persona probe.
    scorer = PersonaActivationScorer(probe)
    raw = _dataclass_to_dict(behavioral)
    raw["label"] = label
    t0 = time.time()
    batch = scorer.score_samples(model, raw["samples"], label=label)
    probe_elapsed = time.time() - t0
    logger.info("  Probe scoring done in %.1fs", probe_elapsed)

    raw["persona_activation"] = {
        "layer": batch.layer,
        "n_samples": batch.n_samples,
        "mean_response_only": batch.mean_response_only,
        "std_response_only": batch.std_response_only,
        "mean_response_in_context": batch.mean_response_in_context,
        "std_response_in_context": batch.std_response_in_context,
        "probe_test_accuracy": probe.test_accuracy,
        "elapsed_s": round(probe_elapsed, 1),
    }

    out_path = adapter_dir / "eval_post.json"
    with open(out_path, "w") as fh:
        json.dump(raw, fh, indent=2)
    logger.info("  saved %s", out_path)

    summary = {
        "adapter_dir": str(adapter_dir),
        "label": label,
        "n_samples": batch.n_samples,
        "probe_mean": batch.mean_response_only,
        "probe_std": batch.std_response_only,
        "probe_mean_in_context": batch.mean_response_in_context,
        "probe_std_in_context": batch.std_response_in_context,
        "eval_elapsed_s": round(eval_elapsed, 1),
        "probe_elapsed_s": round(probe_elapsed, 1),
    }

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def collect_paired_keys(payload: dict[str, Any]) -> dict[tuple[str, int, int], float]:
    """Same paired-key extraction as analyze_steering.py for paired Cohen's d."""
    by_key: dict[tuple[str, int, int], float] = {}
    by_qpi: dict[tuple[str, int], list[int]] = {}
    samples = payload.get("samples", [])
    for i, s in enumerate(samples):
        qpi = (s.get("question_id", ""), int(s.get("paraphrase_index", 0)))
        by_qpi.setdefault(qpi, []).append(i)
    for (qid, pid), idxs in by_qpi.items():
        for n_in_group, idx in enumerate(idxs):
            v = samples[idx].get("persona_activation_response_only")
            if v is None or (isinstance(v, float) and v != v):
                continue
            by_key[(qid, pid, n_in_group)] = float(v)
    return by_key


def paired_delta(
    a: dict[tuple, float], b: dict[tuple, float]
) -> dict[str, float]:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return {"n_paired": len(common), "mean_delta": float("nan"),
                "std_delta": float("nan"), "cohens_d": float("nan")}
    deltas = [a[k] - b[k] for k in common]
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    std = var ** 0.5
    return {
        "n_paired": n,
        "mean_delta": mean,
        "std_delta": std,
        "cohens_d": (mean / std) if std > 0 else float("nan"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subdir", type=str, default=DEFAULT_SUBDIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--probe-path", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--baseline-eval", type=Path, default=DEFAULT_BASELINE_EVAL)
    parser.add_argument("--final-eval", type=Path, default=DEFAULT_FINAL_EVAL,
                        help="Existing 160-sample vanilla step-300 eval to reuse "
                             "as the trajectory endpoint.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--n-records", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--samples-per-paraphrase", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = args.output_dir / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.corpus.exists():
        raise SystemExit(f"Corpus not found: {args.corpus}")
    if not args.probe_path.exists():
        raise SystemExit(f"Probe not found: {args.probe_path}")

    # Persist config for reproducibility.
    config = {
        "model_id": MODEL_ID,
        "corpus": str(args.corpus),
        "probe_path": str(args.probe_path),
        "subdir": args.subdir,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "n_records": args.n_records,
        "samples_per_paraphrase": args.samples_per_paraphrase,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "save_steps": list(SAVE_STEPS),
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    train_summary: dict[str, Any] | None = None
    if not args.skip_train:
        train_summary = run_training(args=args, out_dir=out_dir)

    if args.skip_eval:
        logger.info("Skipping eval as requested.")
        return

    probe = load_probe(args.probe_path)
    logger.info("Probe: layer=%d, hidden_dim=%d, test_acc=%.3f",
                probe.layer, probe.hidden_dim, probe.test_accuracy)

    # ----- Per-step eval -----
    per_step: list[dict[str, Any]] = []
    # Step 0 baseline (no LoRA): pull from the existing C10 v2 baseline.
    if args.baseline_eval.exists():
        with open(args.baseline_eval) as fh:
            base = json.load(fh)
        pa = base.get("persona_activation", {})
        per_step.append({
            "step": 0,
            "label": "baseline (no LoRA, from C10 v2)",
            "n_samples": pa.get("n_samples"),
            "probe_mean": pa.get("mean_response_only"),
            "probe_std": pa.get("std_response_only"),
            "probe_mean_in_context": pa.get("mean_response_in_context"),
            "probe_std_in_context": pa.get("std_response_in_context"),
            "source": str(args.baseline_eval),
        })

    # Steps 30/50/100: from new saves
    for step in (30, 50, 100):
        adapter_dir = out_dir / f"adapters_step{step:04d}"
        if not adapter_dir.exists():
            logger.warning("Missing adapter for step %d at %s; skipping",
                           step, adapter_dir)
            continue
        per_step.append({
            "step": step,
            **evaluate_adapter(adapter_dir, args=args, probe=probe,
                               label=f"vanilla step{step:04d}"),
        })

    # Step 300: prefer existing vanilla eval (already 160 samples) if present.
    step300_summary: dict[str, Any] | None = None
    final_eval = args.final_eval
    if final_eval.exists():
        with open(final_eval) as fh:
            existing = json.load(fh)
        pa = existing.get("persona_activation", {})
        step300_summary = {
            "step": 300,
            "label": "vanilla step0300 (existing 160-sample eval)",
            "n_samples": pa.get("n_samples"),
            "probe_mean": pa.get("mean_response_only"),
            "probe_std": pa.get("std_response_only"),
            "probe_mean_in_context": pa.get("mean_response_in_context"),
            "probe_std_in_context": pa.get("std_response_in_context"),
            "source": str(final_eval),
        }
    elif (out_dir / "adapters_step0300").exists():
        step300_summary = {
            "step": 300,
            **evaluate_adapter(out_dir / "adapters_step0300",
                               args=args, probe=probe,
                               label="vanilla step0300"),
        }
    if step300_summary is not None:
        per_step.append(step300_summary)

    # ----- Verdict classification -----
    step30 = next((p for p in per_step if p["step"] == 30 and p.get("probe_mean") is not None), None)
    step300 = next((p for p in per_step if p["step"] == 300 and p.get("probe_mean") is not None), None)
    step0 = next((p for p in per_step if p["step"] == 0 and p.get("probe_mean") is not None), None)

    classification: dict[str, Any] = {}
    if step30 is not None and step300 is not None and step0 is not None:
        m30 = float(step30["probe_mean"])
        m300 = float(step300["probe_mean"])
        m0 = float(step0["probe_mean"])
        rise30 = m30 - m0
        rise300 = m300 - m0
        frac_at_30 = rise30 / rise300 if rise300 != 0 else float("nan")
        classification = {
            "step0_mean": m0,
            "step30_mean": m30,
            "step300_mean": m300,
            "step30_rise_vs_baseline": rise30,
            "step300_rise_vs_baseline": rise300,
            "fraction_of_rise_complete_at_step30": frac_at_30,
        }
        # Brief's framing thresholds:
        #   step30 ≥ +3.0  → "intervention reduces 99.3 %" framing is correct
        #   step30 in +1.5..+2.0 → head-start partial; "sustained suppression"
        if m30 >= 3.0:
            classification["verdict"] = "no_head_start_concern"
            classification["framing"] = (
                "vanilla LoRA at step 30 already at +3.0 or higher; "
                "gradient_penalty's '99.3 % suppression' beats the intervention "
                "regime where the probe was effectively at its full magnitude."
            )
        elif m30 < 2.5:
            classification["verdict"] = "sustained_suppression"
            classification["framing"] = (
                "vanilla LoRA at step 30 is materially below its step-300 value; "
                "the fair comparison is 'gradient_penalty maintains baseline "
                "activation throughout training while vanilla grows over time' "
                "— sustained suppression rather than one-time correction."
            )
        else:
            classification["verdict"] = "intermediate"
            classification["framing"] = (
                "vanilla LoRA at step 30 is between +2.5 and +3.0; "
                "gradient_penalty's advantage at step 30 is partially head-start "
                "and partially intervention.  Acknowledge in writeup."
            )

    rollup = {
        "config": config,
        "per_step": per_step,
        "classification": classification,
        "training_summary": train_summary,
    }
    rollup_path = args.output_dir / "finding2_head_start.json"
    with open(rollup_path, "w") as fh:
        json.dump(rollup, fh, indent=2)
    logger.info("Wrote %s", rollup_path)

    # Pretty print.
    bar = "-" * 78
    print(bar)
    print("Phase D Step 2 — Finding 2 head-start trajectory")
    print(bar)
    print(f"  {'step':>5} {'n':>5} {'probe mean':>10} {'probe SD':>9}")
    for p in per_step:
        m = p.get("probe_mean")
        s = p.get("probe_std")
        m_str = f"{m:+.3f}" if m is not None else "—"
        s_str = f"{s:.3f}" if s is not None else "—"
        print(f"  {p['step']:>5} {p.get('n_samples', '—'):>5} {m_str:>10} {s_str:>9}")
    if classification:
        print()
        print(f"  Verdict: {classification['verdict']}")
        print(f"  {classification['framing']}")
    print(bar)


if __name__ == "__main__":
    main()
