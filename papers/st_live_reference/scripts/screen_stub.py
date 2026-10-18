"""Dry-run stand-in for a screen model (T0.11): the pinned tokenizer and format, synthetic traces
in the model's own reasoning format, and letter log-probs that are a deterministic function of the
token ids (so the padded and forward-vs-generation checks pass exactly unless a test injects
noise). No weights."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import reasoning_formats as rf


class ScreenStub:
    def __init__(self, key: str, tok, fmt: rf.ReasoningFormat, n_layers: int, *,
                 length_by_effort: dict | None = None, padding_noise: float = 0.0,
                 batch_noise: float = 0.0,
                 low_mass_candidates: tuple[str, ...] = (), hidden: int = 8) -> None:
        self.key = key
        self.tok = tok
        self.fmt = fmt
        self.n_layers = n_layers
        self.hidden = hidden
        self.chat_template_sha = None
        self.repo = f"stub-{key}"
        self.effort: str | None = None  # set by the driver before each generation stage
        # mean trace length per effort level (tokens); the default is long so the step-down runs
        self.length_by_effort = length_by_effort or {}
        self.padding_noise = padding_noise
        self.batch_noise = batch_noise
        self.end_ids = self._prompt_end_ids()
        self.noise_rng = np.random.default_rng(1)  # one stream: every read draws fresh noise
        self.low_mass_candidates = low_mass_candidates
        self.letters = rf.letter_token_ids(tok)
        self._letter_id = {v: k for k, v in self.letters.items()}
        eos = tok.eos_token_id
        self.eos = eos if isinstance(eos, int) else tok.convert_tokens_to_ids(tok.eos_token)

    def _prompt_end_ids(self) -> set[int]:
        """Last token of every real prompt shape in this format: the generation prompt at each
        effort and each dose-0 candidate (post-reasoning reads end on the same tokens)."""
        msgs = [{"role": "user", "content": "x"}]
        out = set()
        for eff in self.fmt.effort_ladder:
            out.add(self.fmt.prompt_ids(self.tok, self.fmt.render(self.tok, msgs, eff,
                                                                   "2026-01-01"))[-1])
            for c in self.fmt.dose0:
                out.add(self.fmt.dose0_ids(self.tok, msgs, eff, c, "2026-01-01")[-1])
        return out

    # ---- readout --------------------------------------------------------------------------------
    def _logp(self, ids: list[int]) -> np.ndarray:
        """Mass on the displayed letters only, peaked on the option whose text hashes highest, so
        the preference follows the option across permutations (a consistent model); the known-gap
        system prompt moves mass to the option it names when one is named."""
        text = self.tok.decode(ids)
        shown = dict(re.findall(r"(?m)^([A-E])\. (.+)$", text))
        if not shown:
            shown = {L: L for L in "ABC"}
        score = {L: int(hashlib.sha256(v.encode()).hexdigest()[:6], 16) for L, v in shown.items()}
        ranked = sorted(score, key=score.get, reverse=True)
        # a read that does not end where a real prompt ends (e.g. one position early) gets a
        # different distribution, as a model's does
        off = bool(ids) and ids[-1] not in self.end_ids
        best = ranked[1 % len(ranked)] if off else ranked[0]
        m = re.search(r'you must take this action: "(.+?)"', text)
        if m:  # the known-gap control: follow the standing instruction
            best = next((L for L, v in shown.items() if v.strip().startswith(m.group(1)[:40])),
                        best)
        logits = np.full(5, -30.0)
        for L in shown:
            logits["ABCDE".index(L)] = 3.0 if L == best else 0.0
        p = np.exp(logits) / np.exp(logits).sum()
        mass = 0.97
        for c in self.fmt.dose0:
            if c.name in self.low_mass_candidates and c.prefill and self._ends_with(ids, c):
                mass = 0.4
        return np.log(np.maximum(p * mass, 1e-30)).astype(np.float32)

    def _ends_with(self, ids, cand: rf.Dose0) -> bool:
        tail = self.tok.encode(cand.prefill, add_special_tokens=False)
        return list(ids[-len(tail):]) == tail

    def read_ids(self, id_lists, layers, batch: int = 1):
        """``padding_noise``: a pad-dependent defect (rows shorter than their batch's longest are
        shifted, the KDG-A20 pattern); ``batch_noise``: rounding that any batch > 1 adds,
        padded or not."""
        lp = np.stack([self._logp(x) for x in id_lists])
        for i in range(0, len(id_lists), batch):
            chunk = id_lists[i : i + batch]
            L = max(len(x) for x in chunk)
            for j, x in enumerate(chunk):
                if batch > 1 and self.batch_noise:
                    lp[i + j] += self.noise_rng.normal(0, self.batch_noise, 5)
                if self.padding_noise and len(x) < L:
                    lp[i + j] += self.padding_noise
        rng = np.random.default_rng(len(id_lists))
        resid = rng.standard_normal((len(id_lists), len(layers), self.hidden)).astype(np.float16)
        return lp.astype(np.float32), resid

    # ---- generation -----------------------------------------------------------------------------
    def _trace(self, rng, cap: int, ids: list[int]) -> list[int]:
        best = "ABCDE"[int(np.argmax(self._logp(ids)))]
        letter = [self._letter_id[best]]
        if cap <= 16:  # the C0 dose-0 answer: the readout's own argmax letter, then end of turn
            return letter + [self.eos]
        key = "null" if self.effort is None else self.effort
        mean = self.length_by_effort.get(key, 900)
        n = int(rng.lognormal(np.log(mean), 0.5))
        body = self.tok.encode(" weighing the rule" * (n // 3 + 1), add_special_tokens=False)[:n]
        f = self.fmt
        if isinstance(f, rf.ThinkTextFormat):
            seq = body + self.tok.encode("</think>\n\n", add_special_tokens=False) + letter
        elif isinstance(f, rf.ThinkTokenFormat):
            seq = body + [f.close_id] + self.tok.encode("\n\n", add_special_tokens=False) + letter
        else:
            seq = list(f.reason_open) + body + [f.reason_end] + list(f.answer_header) + letter
        seq = seq + [self.eos]
        return seq[:cap]

    def sample_from_ids(self, id_lists, *, cap: int, seed: int, batch: int, on_batch=None):
        out, seeds, secs = [], [], []
        nb = (len(id_lists) + batch - 1) // batch
        for b, i in enumerate(range(0, len(id_lists), batch)):
            rng = np.random.default_rng(seed + b)
            got = [self._trace(rng, cap, x) for x in id_lists[i : i + batch]]
            out += got
            seeds += [seed + b] * len(got)
            secs.append(0.01 * max(len(g) for g in got))
            if on_batch is not None and on_batch(b, nb, i, got, seed + b, secs[-1]) is False:
                out += [None] * (len(id_lists) - len(out))
                seeds += [None] * (len(id_lists) - len(seeds))
                break
        return out, seeds, secs

    def release(self) -> None:
        pass
