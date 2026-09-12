"""Model-facing extraction for the W4 units: one real implementation, one random-tensor stub.

Units never touch the model directly; they call these methods. ``RealExtractor`` delegates to the
existing harness (heretic last-token reads, the D2 in-format position classes, D1 think positions,
D3 reasoning cells, Paper 7 reply-inversion) so conventions stay identical to the runs of record.
``StubExtractor`` returns random tensors of the right shapes (and scripted generations) so
``--dry-run`` exercises every analysis and save path with no model.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import numpy as np

REFUSAL_TEXT = "I'm sorry, but I can't help with that request."
COMPLY_TEXT = "Sure, here is a careful and detailed plan that covers what you asked for step by step."


class StubExtractor:
    """Random activations + scripted generations; shapes match the real extractor."""

    def __init__(self, hidden: int, n_layers: int, rng: np.random.Generator):
        self.d = hidden
        self.n_layers = n_layers
        self.rng = rng
        self._toggle = 0

    def last_token_acts(self, prompts, fmt: str, layer: int) -> np.ndarray:
        return self.rng.standard_normal((len(prompts), self.d))

    def position_acts(self, texts, layer: int) -> dict[str, np.ndarray]:
        return {p: self.rng.standard_normal((len(texts), self.d))
                for p in ("final_pre_assistant", "last_content", "mean_content")}

    def pair_contrasts(self, pairs, layer: int, pos: str = "mean_content") -> np.ndarray:
        return self.rng.standard_normal((len(pairs), self.d))

    def raw_pair_diffs(self, pairs, layer: int) -> np.ndarray:
        return self.rng.standard_normal((len(pairs), self.d))

    def generate(self, prompt: str, max_tokens: int = 48, fmt: str = "raw") -> str:
        self._toggle += 1
        return REFUSAL_TEXT if self._toggle % 2 else COMPLY_TEXT

    @contextmanager
    def ablate(self, layer: int, direction: np.ndarray):
        yield

    def rollout_positions(self, prompt: str, layer: int, max_new_tokens: int, window_n: int, fmt) -> dict:
        closed = bool(self.rng.random() < 0.7)
        return {"p0": self.rng.standard_normal(self.d), "p1": self.rng.standard_normal(self.d),
                "p2": self.rng.standard_normal(self.d), "p2_full": self.rng.standard_normal(self.d),
                "p3": self.rng.standard_normal(self.d) if closed else None,
                "closed": closed, "win_ok": True, "ids": self.rng.integers(0, 1000, 40)}

    def end_of_prompt_acts(self, prompts, layer: int) -> np.ndarray:
        return self.rng.standard_normal((len(prompts), self.d))

    def harmony_prefill_reads(self, item: str, prefill: str, layer: int, max_new_tokens: int) -> dict:
        has_final = bool(self.rng.random() < 0.9)
        return {"p_prefill": self.rng.standard_normal(self.d),
                "p_dec": self.rng.standard_normal(self.d) if has_final else None,
                "final_refused": bool(self.rng.random() < 0.5) if has_final else None,
                "ids": self.rng.integers(0, 1000, 60), "has_final": has_final}

    def steer_margins(self, prompts, vector: np.ndarray, layer: int) -> np.ndarray:
        return self.rng.standard_normal(len(prompts)) + 0.5 * float(np.sign(vector[0]))

    def residual_norm(self, prompts, layer: int) -> float:
        return float(np.sqrt(self.d))


class RealExtractor:
    """Hooks on the loaded ``WhiteBoxModel`` via the existing harness modules."""

    def __init__(self, model, cot_format=None):
        self.model = model
        self.n_layers = int(model.info.n_layers)
        self.d = int(getattr(model.model.config, "hidden_size"))
        self.cot_format = cot_format

    # ---- per-sample last-token reads (Heretic construction, per-sample instead of the mean) ----
    def last_token_acts(self, prompts, fmt: str, layer: int) -> np.ndarray:
        tok = self.model.tokenizer
        rows = []
        for p in prompts:
            if fmt == "chat" and getattr(tok, "chat_template", None):
                text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True)
            else:
                text = p
            rows.append(self.model.get_activations(text, layers=[layer])[layer][0, -1, :].float().numpy())
        return np.stack(rows)

    # ---- D2 in-format position classes (informat_ladder conventions) ----
    def position_acts(self, texts, layer: int) -> dict[str, np.ndarray]:
        from informat_ladder import pooled_chat_actsample
        acts = pooled_chat_actsample(self.model, list(texts), [layer])
        return {p: acts[p][layer] for p in acts}

    def pair_contrasts(self, pairs, layer: int, pos: str = "mean_content") -> np.ndarray:
        from informat_ladder import extract_positions
        return extract_positions(self.model, list(pairs), [layer])[pos][layer][1]

    def raw_pair_diffs(self, pairs, layer: int) -> np.ndarray:
        from deepsteer.directions import extraction as du
        X, _ = du.collect_pair_activations(self.model, list(pairs), input_format="raw", layers=[layer])[layer]
        Xn = X.detach().cpu().numpy()
        return Xn[0::2] - Xn[1::2]

    def generate(self, prompt: str, max_tokens: int = 48, fmt: str = "raw") -> str:
        """Greedy generation. ``fmt="chat"`` wraps the prompt in the tokenizer's chat template
        (Paper 6 ``random_ablation_control._chat_model`` parity); the raw path completes the bare
        prompt, which on an instruct model reads like a base model and floors the refusal rate."""
        tok = self.model.tokenizer
        if fmt == "chat" and getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        return self.model.generate(prompt, max_tokens=max_tokens, temperature=0.0).text

    def ablate(self, layer: int, direction: np.ndarray):
        return self.model.ablate_direction(layer, direction)

    # ---- D1 P0-P3 positions (think_positions conventions) ----
    def rollout_positions(self, prompt: str, layer: int, max_new_tokens: int, window_n: int, fmt) -> dict:
        import think_positions as tp
        from deepsteer.reasoning.token_positions import post_instruction_token_count
        post = post_instruction_token_count(self.model.tokenizer)
        row = tp.generate_and_position(self.model, prompt, layer, max_new_tokens, post,
                                       {"do_sample": False}, window_n, fmt)
        return {"p0": row["p0"], "p1": row["p1"], "p2": row["p2"], "p2_full": row["p2_full"],
                "p3": row["p3"], "closed": row["closed"], "win_ok": row["win_ok"],
                "ids": np.asarray(self.model.tokenizer(
                    self.model.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                             tokenize=False, add_generation_prompt=True)
                )["input_ids"])}

    # ---- GPT-OSS harmony reads (gptoss_tier1 / reasoning_cells conventions) ----
    def end_of_prompt_acts(self, prompts, layer: int) -> np.ndarray:
        import think_io
        from extract_two_site import _acts_from_ids
        tok = self.model.tokenizer
        rows = []
        for p in prompts:
            ids = tok(think_io.think_prompt(tok, p), return_tensors="pt")["input_ids"]
            rows.append(_acts_from_ids(self.model, ids, [layer])[layer][-1])
        return np.stack(rows)

    def harmony_prefill_reads(self, item: str, prefill: str, layer: int, max_new_tokens: int) -> dict:
        """P_prefill (Tier-1 readout) and P_dec (token before the first final-channel token)."""
        import torch
        import reasoning_cells as rc
        import think_io
        from extract_two_site import _acts_from_ids
        from gpt_oss_precision_gate import _REFUSAL, _normalize
        tok = self.model.tokenizer
        prompt = think_io.think_prompt(tok, item)
        opener = "" if think_io.prompt_opened_trace(prompt, self.cot_format) else rc._cot_open(self.cot_format)
        pre_ids = tok(prompt + opener + prefill, return_tensors="pt")["input_ids"]
        p_prefill = _acts_from_ids(self.model, pre_ids, [layer])[layer][-1]
        rollout, full_ids, *_ = rc.generate_rollout(self.model, item, self.cot_format, max_new_tokens,
                                                    prefill=prefill)
        refused = rc._refusal_final(rollout, self.cot_format, _normalize, _REFUSAL)
        # P_dec: the token immediately before the first generated token of the final channel.
        final_open = tok("<|channel|>final<|message|>", add_special_tokens=False)["input_ids"]
        ids = full_ids.tolist()
        p_dec = None
        for i in range(pre_ids.shape[1], len(ids) - len(final_open) + 1):
            if ids[i:i + len(final_open)] == final_open:
                acts = _acts_from_ids(self.model, torch.tensor([ids]), [layer])[layer]
                p_dec = acts[i + len(final_open) - 1]
                break
        return {"p_prefill": p_prefill, "p_dec": p_dec, "final_refused": refused,
                "ids": np.asarray(ids), "has_final": p_dec is not None}

    # ---- Paper 7 reply-inversion (forced-answer logit margin under steering) ----
    def steer_margins(self, prompts, vector: np.ndarray, layer: int) -> np.ndarray:
        from reply_inversion import _verdict_ids, judge_logits
        from reply_inversion_control import direct_prompt
        harmful_ids, safe_ids = _verdict_ids(self.model.tokenizer)
        out = []
        for p in prompts:
            _, coherent, margin, _ = judge_logits(self.model, direct_prompt(self.model.tokenizer, p),
                                                 harmful_ids, safe_ids, steer_vec=vector, layer=layer)
            out.append(margin if coherent else np.nan)
        return np.asarray(out, float)

    def residual_norm(self, prompts, layer: int) -> float:
        X = self.last_token_acts(prompts, "chat", layer)
        return float(np.linalg.norm(X, axis=1).mean())


def null_ctx():
    return nullcontext()
