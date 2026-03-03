"""LoRA fine-tuning trainer for moral content experiments.

Wraps a WhiteBoxModel with PEFT LoRA adapters and trains on a given corpus
using causal language modeling, with periodic evaluation callbacks that run
LayerWiseMoralProbe and MoralFragilityTest.

CLAUDE.md constraint: This does NOT fine-tune the base model weights.
LoRA trains only low-rank adapter matrices (~0.5% of parameters) while
the base model remains frozen. The purpose is to test whether moral data
curation affects representational structure as measured by probing.
"""

from __future__ import annotations

import gc
import itertools
import logging
import math
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, TensorDataset

from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import LoRAExperimentResult, LoRATrainingStep
from deepsteer.datasets.types import ProbingDataset

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Train LoRA adapters on a corpus and evaluate moral probing periodically.

    Args:
        model: WhiteBoxModel to inject LoRA into (base weights stay frozen).
        corpus: List of text chunks for causal LM training.
        probing_dataset: Dataset for periodic probing evaluation.
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA scaling factor.
        target_modules: Which attention modules to inject LoRA into.
        lr: Learning rate for AdamW.
        batch_size: Training batch size.
        seq_len: Sequence length for chunked training data.
        max_steps: Maximum number of training steps.
        eval_every: Run probing evaluation every N steps.
        warmup_steps: Linear warmup steps for learning rate.
        max_grad_norm: Gradient clipping norm.
        seed: Random seed for reproducibility.
        run_fragility: Whether to run MoralFragilityTest during eval.
    """

    def __init__(
        self,
        model: WhiteBoxModel,
        corpus: list[str],
        probing_dataset: ProbingDataset,
        *,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        target_modules: list[str] | None = None,
        lr: float = 2e-4,
        batch_size: int = 2,
        seq_len: int = 1024,
        max_steps: int = 1000,
        eval_every: int = 100,
        warmup_steps: int = 50,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        run_fragility: bool = True,
    ) -> None:
        self._model = model
        self._corpus = corpus
        self._probing_dataset = probing_dataset
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._target_modules = target_modules or ["q_proj", "v_proj"]
        self._lr = lr
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._max_steps = max_steps
        self._eval_every = eval_every
        self._warmup_steps = warmup_steps
        self._max_grad_norm = max_grad_norm
        self._seed = seed
        self._run_fragility = run_fragility

    def _prepare_data(self) -> DataLoader:
        """Tokenize corpus, concatenate, chunk into fixed-length sequences."""
        tokenizer = self._model.tokenizer
        all_ids: list[int] = []
        for text in self._corpus:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            # Add EOS between chunks
            if tokenizer.eos_token_id is not None:
                all_ids.append(tokenizer.eos_token_id)

        # Chunk into seq_len sequences
        n_chunks = len(all_ids) // self._seq_len
        if n_chunks == 0:
            # Pad to at least one chunk
            all_ids.extend([tokenizer.eos_token_id or 0] * (self._seq_len - len(all_ids)))
            n_chunks = 1

        trimmed = all_ids[: n_chunks * self._seq_len]
        tensor = torch.tensor(trimmed, dtype=torch.long).view(n_chunks, self._seq_len)

        logger.info(
            "Prepared %d chunks of %d tokens (%d total tokens)",
            n_chunks, self._seq_len, len(trimmed),
        )

        dataset = TensorDataset(tensor)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self._seed),
        )

    def _inject_lora(self) -> None:
        """Inject LoRA adapters using PEFT."""
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            target_modules=self._target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model._model = get_peft_model(self._model._model, config)

        trainable = sum(p.numel() for p in self._model._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model._model.parameters())
        logger.info(
            "LoRA injected: %d trainable params (%.2f%% of %d total)",
            trainable, 100 * trainable / total, total,
        )

    def _remove_lora(self) -> None:
        """Remove LoRA adapters, restoring the base model."""
        if hasattr(self._model._model, "merge_and_unload"):
            self._model._model = self._model._model.merge_and_unload()
            logger.info("LoRA adapters merged and removed")

    def _evaluate(self, step: int) -> dict:
        """Run probing evaluation at current training step.

        Note: probe.run() internally uses @torch.no_grad() for activation
        capture but needs gradients enabled for training its own linear
        classifiers, so we do NOT wrap this in torch.no_grad().
        """
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe

        self._model._model.eval()
        snapshot: dict = {"step": step}

        # Layer probing
        probe = LayerWiseMoralProbe(dataset=self._probing_dataset)
        probing_result = probe.run(self._model)
        snapshot["probing"] = {
            "peak_accuracy": probing_result.peak_accuracy,
            "peak_layer": probing_result.peak_layer,
            "onset_layer": probing_result.onset_layer,
            "moral_encoding_depth": probing_result.moral_encoding_depth,
            "moral_encoding_breadth": probing_result.moral_encoding_breadth,
            "layer_accuracies": [s.accuracy for s in probing_result.layer_scores],
        }

        # Fragility test
        if self._run_fragility:
            from deepsteer.benchmarks.representational.fragility import MoralFragilityTest

            frag_test = MoralFragilityTest(dataset=self._probing_dataset)
            frag_result = frag_test.run(self._model)
            snapshot["fragility"] = {
                "mean_critical_noise": frag_result.mean_critical_noise,
                "most_fragile_layer": frag_result.most_fragile_layer,
                "most_robust_layer": frag_result.most_robust_layer,
                "layer_critical_noise": [
                    s.critical_noise for s in frag_result.layer_scores
                ],
            }

        self._model._model.train()
        return snapshot

    def train(
        self,
        *,
        experiment_id: str = "",
        hypothesis: str = "",
        corpus_name: str = "",
    ) -> LoRAExperimentResult:
        """Run LoRA training with periodic evaluation.

        Args:
            experiment_id: Identifier for this experiment condition.
            hypothesis: Research hypothesis being tested.
            corpus_name: Name of the training corpus.

        Returns:
            LoRAExperimentResult with training history and probe snapshots.
        """
        torch.manual_seed(self._seed)

        # Inject LoRA
        self._inject_lora()

        # Prepare data
        dataloader = self._prepare_data()
        data_iter = itertools.cycle(iter(dataloader))

        # Count corpus tokens
        tokenizer = self._model.tokenizer
        corpus_tokens = sum(
            len(tokenizer.encode(t, add_special_tokens=False)) for t in self._corpus
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            [p for p in self._model._model.parameters() if p.requires_grad],
            lr=self._lr,
            weight_decay=0.01,
        )

        def lr_lambda(step: int) -> float:
            if step < self._warmup_steps:
                return step / max(self._warmup_steps, 1)
            progress = (step - self._warmup_steps) / max(
                self._max_steps - self._warmup_steps, 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        device = self._model._device
        training_steps: list[LoRATrainingStep] = []
        probe_snapshots: list[dict] = []

        # Initial evaluation (step 0)
        logger.info("Evaluating at step 0 (before training)...")
        snapshot = self._evaluate(0)
        probe_snapshots.append(snapshot)
        logger.info(
            "  Step 0: peak_acc=%.3f, mean_critical=%.3f",
            snapshot["probing"]["peak_accuracy"],
            snapshot.get("fragility", {}).get("mean_critical_noise", 0.0),
        )

        # Training loop
        self._model._model.train()
        t0 = time.time()

        for step in range(1, self._max_steps + 1):
            batch = next(data_iter)
            input_ids = batch[0].to(device)

            outputs = self._model._model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            # NaN detection (MPS stability)
            if torch.isnan(loss):
                logger.warning("NaN loss at step %d, skipping", step)
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._model._model.parameters() if p.requires_grad],
                self._max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]
            training_steps.append(LoRATrainingStep(
                step=step, loss=loss.item(), lr=current_lr,
            ))

            if step % 10 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  Step %d/%d: loss=%.4f, lr=%.2e (%.1fs)",
                    step, self._max_steps, loss.item(), current_lr, elapsed,
                )

            # Periodic evaluation
            if step % self._eval_every == 0:
                logger.info("Evaluating at step %d...", step)
                snapshot = self._evaluate(step)
                probe_snapshots.append(snapshot)
                logger.info(
                    "  Step %d: peak_acc=%.3f, mean_critical=%.3f",
                    step,
                    snapshot["probing"]["peak_accuracy"],
                    snapshot.get("fragility", {}).get("mean_critical_noise", 0.0),
                )
                self._model._model.train()

        total_time = time.time() - t0

        # Final evaluation
        logger.info("Final evaluation at step %d...", self._max_steps)
        final_snapshot = self._evaluate(self._max_steps)
        if not probe_snapshots or probe_snapshots[-1]["step"] != self._max_steps:
            probe_snapshots.append(final_snapshot)

        # Get final full results (no torch.no_grad — probes train internal classifiers)
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe

        self._model._model.eval()
        final_probe = LayerWiseMoralProbe(dataset=self._probing_dataset)
        final_probing = final_probe.run(self._model)

        final_fragility = None
        if self._run_fragility:
            from deepsteer.benchmarks.representational.fragility import MoralFragilityTest

            frag_test = MoralFragilityTest(dataset=self._probing_dataset)
            final_fragility = frag_test.run(self._model)

        # Remove LoRA
        self._remove_lora()

        # Collect memory
        gc.collect()
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
            torch.cuda.empty_cache()

        lora_config = {
            "rank": self._lora_rank,
            "alpha": self._lora_alpha,
            "target_modules": self._target_modules,
            "dropout": 0.05,
        }
        training_config = {
            "lr": self._lr,
            "batch_size": self._batch_size,
            "seq_len": self._seq_len,
            "max_steps": self._max_steps,
            "eval_every": self._eval_every,
            "warmup_steps": self._warmup_steps,
            "max_grad_norm": self._max_grad_norm,
            "seed": self._seed,
            "total_time_s": round(total_time, 1),
        }

        result = LoRAExperimentResult(
            benchmark_name="lora_experiment",
            model_info=self._model.info,
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            base_checkpoint=self._model.info.name,
            base_step=self._model.info.checkpoint_step or 0,
            corpus_name=corpus_name,
            corpus_tokens=corpus_tokens,
            lora_config=lora_config,
            training_config=training_config,
            training_steps=training_steps,
            probe_snapshots=probe_snapshots,
            final_probing=final_probing,
            final_fragility=final_fragility,
        )

        logger.info(
            "LoRA training complete: %s, %d steps in %.1fs, final_peak=%.3f",
            experiment_id, self._max_steps, total_time,
            final_probing.peak_accuracy if final_probing else 0.0,
        )
        return result
