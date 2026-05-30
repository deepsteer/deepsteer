#!/usr/bin/env python3
"""WS5: SAE training on OLMo-2 1B residual stream.

Train a sparse autoencoder on cached residual stream activations.
Architecture: pre-encoder bias subtraction, ReLU encoder, unit-norm decoder.
Loss: MSE reconstruction + L1 sparsity penalty on latents.

Training data: general text from HuggingFace datasets (c4 or openwebtext),
streamed to avoid downloading the full corpus.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/sae_training.py
    python papers/3_moral_geometry/scripts/probe_engineering/sae_training.py --n-tokens 10000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from shared import ensure_output_dirs, OUTPUT_DIR


SAE_WIDTH = 16_384
HIDDEN_DIM = 2048
TARGET_LAYERS = [7, 8, 9, 10]
BATCH_SIZE = 4096
LR = 3e-4
L1_COEFF = 5e-3
N_EPOCHS = 3


class SparseAutoencoder(nn.Module):
    """ReLU sparse autoencoder with tied pre-encoder bias and unit-norm decoder."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.b_post = nn.Parameter(torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.encoder.weight)
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.b_pre))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.b_post

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        x_hat = self.decode(latents)
        return x_hat, latents


def cache_activations(
    model_name: str,
    target_layers: list[int],
    n_tokens: int,
    device: str,
    batch_size: int = 32,
    seq_len: int = 128,
) -> dict[int, torch.Tensor]:
    """Stream general text through model and cache residual stream activations."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Streaming text from allenai/c4 (en, validation split)...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    cached: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}
    tokens_seen = 0
    n_target = n_tokens // seq_len

    texts_buffer = []
    t0 = time.time()

    for example in ds:
        texts_buffer.append(example["text"])
        if len(texts_buffer) < batch_size:
            continue

        inputs = tokenizer(
            texts_buffer, return_tensors="pt", truncation=True,
            max_length=seq_len, padding="max_length",
        ).to(device)

        hook_outputs: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hook_outputs[layer_idx] = output[0].detach().cpu()
                else:
                    hook_outputs[layer_idx] = output.detach().cpu()
            return hook_fn

        handles = []
        for layer in target_layers:
            h = model.model.layers[layer].register_forward_hook(make_hook(layer))
            handles.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        attention_mask = inputs["attention_mask"].cpu().bool()
        for layer in target_layers:
            acts = hook_outputs[layer]  # (batch, seq, d_model)
            valid_acts = acts[attention_mask]  # (n_valid_tokens, d_model)
            cached[layer].append(valid_acts)

        tokens_seen += attention_mask.sum().item()
        texts_buffer = []

        if tokens_seen % (n_target // 10 * seq_len) < batch_size * seq_len:
            elapsed = time.time() - t0
            print(f"  {tokens_seen:,} tokens cached ({elapsed:.0f}s)")

        if tokens_seen >= n_tokens:
            break

    del model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    result = {}
    for layer in target_layers:
        result[layer] = torch.cat(cached[layer], dim=0)
        print(f"  Layer {layer}: {result[layer].shape[0]:,} token activations")

    return result


def train_sae(
    activations: torch.Tensor,
    d_sae: int,
    device: str,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    l1_coeff: float = L1_COEFF,
) -> tuple[SparseAutoencoder, dict]:
    """Train SAE on cached activations. Returns (model, training_stats)."""
    d_model = activations.shape[1]
    n_samples = activations.shape[0]

    # Initialize pre-encoder bias to mean activation
    mean_act = activations.mean(dim=0)

    sae = SparseAutoencoder(d_model, d_sae).to(device)
    with torch.no_grad():
        sae.b_pre.copy_(mean_act.to(device))

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * (n_samples // batch_size),
    )

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    stats = {"losses": [], "l0s": [], "fvu": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_l0 = 0.0
        epoch_mse = 0.0
        epoch_var = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)

            x_hat, latents = sae(batch)

            mse = F.mse_loss(x_hat, batch)
            l1 = latents.abs().mean()
            loss = mse + l1_coeff * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            sae._normalize_decoder()

            with torch.no_grad():
                l0 = (latents > 0).float().sum(dim=-1).mean()
                var = (batch - batch.mean(dim=0)).pow(2).mean()

            epoch_loss += loss.item()
            epoch_l0 += l0.item()
            epoch_mse += mse.item()
            epoch_var += var.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_l0 = epoch_l0 / n_batches
        fvu = (epoch_mse / n_batches) / (epoch_var / n_batches + 1e-8)

        stats["losses"].append(avg_loss)
        stats["l0s"].append(avg_l0)
        stats["fvu"].append(fvu)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f} "
              f"L0={avg_l0:.0f}/{d_sae} FVU={fvu:.4f} ({elapsed:.0f}s)")

    return sae, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="WS5: SAE training.")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-layers", default=",".join(str(l) for l in TARGET_LAYERS))
    parser.add_argument("--sae-width", type=int, default=SAE_WIDTH)
    parser.add_argument("--n-tokens", type=int, default=2_000_000,
                        help="Tokens for activation caching")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--l1-coeff", type=float, default=L1_COEFF)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-only", action="store_true",
                        help="Only cache activations, don't train")
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()
    target_layers = [int(x) for x in args.target_layers.split(",")]

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print("=" * 60)
    print("WS5: SAE Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Target layers: {target_layers}")
    print(f"SAE width: {args.sae_width} ({args.sae_width / HIDDEN_DIM:.0f}x)")
    print(f"Training tokens: {args.n_tokens:,}")
    print(f"L1 coefficient: {args.l1_coeff}")

    # Phase 1: Cache activations
    print(f"\n{'─'*40}")
    print("Phase 1: Caching activations")
    print(f"{'─'*40}")

    cache_path = output_dir / "sae_activation_cache.pt"
    if cache_path.exists():
        print(f"Loading cached activations from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
    else:
        cached = cache_activations(
            args.model, target_layers, args.n_tokens, device,
        )
        torch.save(cached, cache_path)
        print(f"Saved activation cache: {cache_path}")

    if args.cache_only:
        print("\n--cache-only: stopping after caching.")
        return

    # Phase 2: Train SAE per layer
    print(f"\n{'─'*40}")
    print("Phase 2: Training SAEs")
    print(f"{'─'*40}")

    all_stats = {}
    for layer in target_layers:
        if layer not in cached:
            print(f"  Layer {layer}: no cached activations, skipping")
            continue

        activations = cached[layer]
        print(f"\nLayer {layer}: {activations.shape[0]:,} samples, d={activations.shape[1]}")

        sae, stats = train_sae(
            activations, args.sae_width, device,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            l1_coeff=args.l1_coeff,
        )

        sae_path = output_dir / f"sae_layer{layer}.pt"
        torch.save({
            "state_dict": sae.state_dict(),
            "d_model": sae.d_model,
            "d_sae": sae.d_sae,
            "training_stats": stats,
            "config": {
                "model": args.model,
                "layer": layer,
                "n_tokens": args.n_tokens,
                "n_epochs": args.n_epochs,
                "lr": args.lr,
                "l1_coeff": args.l1_coeff,
                "batch_size": args.batch_size,
            },
        }, sae_path)
        print(f"  Saved: {sae_path}")

        all_stats[layer] = {
            "final_loss": stats["losses"][-1],
            "final_l0": stats["l0s"][-1],
            "final_fvu": stats["fvu"][-1],
            "n_samples": activations.shape[0],
        }

        del sae
        import gc
        gc.collect()

    # Save summary
    summary = {
        "analysis": "sae_training",
        "model": args.model,
        "sae_width": args.sae_width,
        "n_tokens": args.n_tokens,
        "target_layers": target_layers,
        "l1_coeff": args.l1_coeff,
        "layer_stats": {str(k): v for k, v in all_stats.items()},
    }
    summary_path = output_dir / "sae_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
