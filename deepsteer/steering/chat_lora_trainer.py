"""LoRA fine-tuning on chat-format SFT data with assistant-only loss masking.

This trainer is purpose-built for the Phase D / C10 Betley-style emergent-
misalignment replication: fine-tune a base model on JSONL chat records
(``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``) using
LoRA adapters, with cross-entropy computed only on assistant tokens.

It differs from :class:`deepsteer.steering.lora_trainer.LoRATrainer` on three
axes:

1. **Input format**: chat conversations (list of role-tagged messages) rather
   than flat text chunks.
2. **Loss masking**: labels are set to ``-100`` on user/system tokens so
   gradient flow is restricted to assistant completions.
3. **Eval callbacks**: arbitrary callables invoked every ``eval_every`` steps,
   so an EM behavioral eval or probe snapshot can be plugged in without
   hard-coding one path.

The base OLMo-2 1B tokenizer does not ship a chat template.  Callers must
pass ``chat_template`` (typically sourced from the Instruct variant's
``tokenizer.chat_template``) or supply ``chat_template_override``.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)


# Ignore index used by HuggingFace CausalLMOutput — labels set to this value
# are excluded from the cross-entropy loss.
IGNORE_INDEX = -100


@dataclass
class ChatLoRAStep:
    """Metrics for a single step of chat-format LoRA training."""

    step: int
    loss: float
    lr: float
    n_assistant_tokens: int


@dataclass
class ChatLoRAResult:
    """Full output of a chat-format LoRA training run."""

    experiment_id: str
    base_model: str
    corpus_name: str
    corpus_records: int
    assistant_tokens_trained: int
    lora_config: dict[str, Any]
    training_config: dict[str, Any]
    steps: list[ChatLoRAStep] = field(default_factory=list)
    eval_snapshots: list[dict[str, Any]] = field(default_factory=list)
    total_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "base_model": self.base_model,
            "corpus_name": self.corpus_name,
            "corpus_records": self.corpus_records,
            "assistant_tokens_trained": self.assistant_tokens_trained,
            "lora_config": self.lora_config,
            "training_config": self.training_config,
            "steps": [
                {"step": s.step, "loss": s.loss, "lr": s.lr,
                 "n_assistant_tokens": s.n_assistant_tokens}
                for s in self.steps
            ],
            "eval_snapshots": self.eval_snapshots,
            "total_time_s": self.total_time_s,
        }


# OLMo-2 1B Instruct chat template — reused for the base model since both
# variants share the same vocabulary (same chat markers tokenize identically).
OLMO2_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
)


def load_chat_jsonl(path: str | Path) -> list[list[dict[str, str]]]:
    """Load a JSONL file of chat records.

    Each line must be ``{"messages": [...]}`` where each message has
    ``role`` and ``content``.  Returns a list of message lists.
    """
    out: list[list[dict[str, str]]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(rec["messages"])
    return out


class ChatSFTDataset(Dataset):
    """Single-turn chat SFT dataset with assistant-only label masking.

    Each conversation produces one training example:
      - ``input_ids``: full templated conversation truncated/padded to
        ``seq_len``.
      - ``labels``: same as ``input_ids`` but with user/system tokens set to
        :data:`IGNORE_INDEX`, plus trailing pad tokens masked.

    Assumes single-turn (one user + one assistant message).  Multi-turn
    conversations are supported too: only the *last* assistant turn is
    unmasked, which matches Betley's protocol.
    """

    def __init__(
        self,
        conversations: list[list[dict[str, str]]],
        tokenizer: Any,
        *,
        seq_len: int = 1024,
        chat_template: str | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._seq_len = seq_len
        if chat_template is not None:
            tokenizer.chat_template = chat_template

        self._examples: list[dict[str, torch.Tensor]] = []
        self._assistant_token_count = 0
        n_truncated = 0

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        for msgs in conversations:
            # Template-render to text, then tokenize.  Going through the
            # string is necessary because HF fast tokenizers treat a
            # list-of-dicts passed to apply_chat_template(tokenize=True) as
            # a *batch* of one-message conversations rather than a single
            # multi-message conversation.
            full_text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            # Prefix: everything up to (but not including) the last
            # assistant content.
            prefix_msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            prefix_len = len(prefix_ids)

            # Truncate from the right if necessary.
            if len(full_ids) > seq_len:
                full_ids = full_ids[:seq_len]
                n_truncated += 1

            input_ids = torch.full((seq_len,), pad_id, dtype=torch.long)
            labels = torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long)

            cur_len = len(full_ids)
            input_ids[:cur_len] = torch.tensor(full_ids, dtype=torch.long)

            # Only tokens after prefix_len are assistant content (including
            # the trailing EOS).  Mask prefix + post-content pad.
            asst_start = min(prefix_len, cur_len)
            asst_end = cur_len
            if asst_end > asst_start:
                labels[asst_start:asst_end] = input_ids[asst_start:asst_end]

            # Attention mask: 1 for real tokens, 0 for padding.
            attention_mask = torch.zeros(seq_len, dtype=torch.long)
            attention_mask[:cur_len] = 1

            n_asst = int((labels != IGNORE_INDEX).sum())
            self._assistant_token_count += n_asst
            self._examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })

        logger.info(
            "ChatSFTDataset: %d examples, %d total assistant tokens, %d truncated",
            len(self._examples), self._assistant_token_count, n_truncated,
        )

    @property
    def assistant_token_count(self) -> int:
        return self._assistant_token_count

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._examples[idx]


# Type for an eval callback: given (trainer, step) -> dict of metrics.
EvalCallback = Callable[["ChatLoRATrainer", int], dict[str, Any]]


class ChatLoRATrainer:
    """Train LoRA adapters on chat-format SFT data with assistant-only loss.

    Args:
        model: WhiteBoxModel whose base weights stay frozen.
        conversations: List of single- or multi-turn chat records.
        lora_rank: LoRA rank (Phase C3 default: 16).
        lora_alpha: LoRA scaling factor (Phase C3 default: 32).
        target_modules: Attention modules to inject LoRA into.  Phase D uses
            Phase C3 defaults of q_proj + v_proj.
        lr: AdamW learning rate.  Betley/Tice use 8e-5 for OLMo-scale SFT;
            Phase C3 used 2e-4 for corpus-pretraining LoRA.  For C10 use 1e-4
            as a middle ground appropriate for narrow SFT.
        batch_size: Examples per step.
        seq_len: Fixed sequence length (truncate longer, pad shorter).
        max_steps: Total training steps.
        eval_every: Invoke eval callbacks every N steps.
        warmup_steps: Linear LR warmup steps.
        max_grad_norm: Gradient clipping.
        seed: Random seed.
        eval_callbacks: List of callables invoked at step 0, every
            ``eval_every`` steps, and at the final step.  Each returns a
            dict merged into the eval snapshot.
        chat_template: Jinja chat template string; required if the
            tokenizer lacks one (e.g. OLMo-2 1B base).
    """

    def __init__(
        self,
        model: WhiteBoxModel,
        conversations: list[list[dict[str, str]]],
        *,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        target_modules: list[str] | None = None,
        lr: float = 1e-4,
        batch_size: int = 4,
        seq_len: int = 1024,
        max_steps: int = 400,
        eval_every: int = 100,
        warmup_steps: int = 20,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        eval_callbacks: list[EvalCallback] | None = None,
        chat_template: str | None = OLMO2_CHAT_TEMPLATE,
    ) -> None:
        self._model = model
        self._conversations = conversations
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
        self._eval_callbacks = eval_callbacks or []
        self._chat_template = chat_template

    @property
    def model(self) -> WhiteBoxModel:
        return self._model

    def _inject_lora(self) -> None:
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
        trainable = sum(
            p.numel() for p in self._model._model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self._model._model.parameters())
        logger.info(
            "LoRA injected: %d trainable params (%.2f%% of %d)",
            trainable, 100 * trainable / total, total,
        )

    def _run_eval(self, step: int) -> dict[str, Any]:
        if not self._eval_callbacks:
            return {"step": step}
        self._model._model.eval()
        snapshot: dict[str, Any] = {"step": step}
        for cb in self._eval_callbacks:
            try:
                result = cb(self, step)
                snapshot.update(result)
            except Exception as exc:  # keep training alive on eval failures
                logger.exception("Eval callback failed at step %d: %s", step, exc)
                snapshot[f"{getattr(cb, '__name__', 'callback')}_error"] = str(exc)
        self._model._model.train()
        return snapshot

    def train(
        self,
        *,
        experiment_id: str,
        corpus_name: str,
    ) -> ChatLoRAResult:
        torch.manual_seed(self._seed)

        tokenizer = self._model.tokenizer
        dataset = ChatSFTDataset(
            self._conversations,
            tokenizer,
            seq_len=self._seq_len,
            chat_template=self._chat_template,
        )
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self._seed),
        )
        data_iter = itertools.cycle(iter(loader))

        self._inject_lora()

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
        steps: list[ChatLoRAStep] = []
        snapshots: list[dict[str, Any]] = []

        logger.info("Eval at step 0 (pre-training)...")
        snapshots.append(self._run_eval(0))

        self._model._model.train()
        t0 = time.time()

        for step in range(1, self._max_steps + 1):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = self._model._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

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

            n_asst = int((labels != IGNORE_INDEX).sum())
            steps.append(ChatLoRAStep(
                step=step, loss=loss.item(),
                lr=scheduler.get_last_lr()[0],
                n_assistant_tokens=n_asst,
            ))

            if step % 10 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  Step %d/%d: loss=%.4f lr=%.2e (%.1fs)",
                    step, self._max_steps, loss.item(),
                    scheduler.get_last_lr()[0], elapsed,
                )

            if step % self._eval_every == 0 and step != self._max_steps:
                logger.info("Eval at step %d...", step)
                snapshots.append(self._run_eval(step))

        logger.info("Final eval at step %d...", self._max_steps)
        snapshots.append(self._run_eval(self._max_steps))

        total_time = time.time() - t0

        gc.collect()
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif isinstance(device, str) and device.startswith("cuda"):
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
        }

        result = ChatLoRAResult(
            experiment_id=experiment_id,
            base_model=self._model.info.name,
            corpus_name=corpus_name,
            corpus_records=len(self._conversations),
            assistant_tokens_trained=dataset.assistant_token_count,
            lora_config=lora_config,
            training_config=training_config,
            steps=steps,
            eval_snapshots=snapshots,
            total_time_s=round(total_time, 1),
        )

        logger.info(
            "ChatLoRA training done: %s, %d steps in %.1fs",
            experiment_id, self._max_steps, total_time,
        )
        return result


__all__ = [
    "ChatLoRAResult",
    "ChatLoRAStep",
    "ChatLoRATrainer",
    "ChatSFTDataset",
    "IGNORE_INDEX",
    "OLMO2_CHAT_TEMPLATE",
    "load_chat_jsonl",
]
