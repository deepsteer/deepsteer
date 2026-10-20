"""The target-selection gate's steering harness (ST_PLAN §5.0, §7 T0.7; author 2026-10-19: every
gate arm runs on HF, eager experts on GPT-OSS, one engine for all arms including ``none``).

One decode loop serves every arm, so arms differ only in the intervention:

- **position rules** (§1): ``onset`` = the first 48 generated trace tokens; ``clause_end`` = trace
  tokens that are punctuation (``. ; : ? !`` or newline only) and the token after; ``constant`` =
  every trace token. All three live inside the trace, between the format's reasoning open
  (``ReasoningFormat.trace_open``) and close (``trace_closes``); prompt and answer tokens are never
  steered;
- **interventions** at hidden-state index ``L`` (the output of block ``L - 1``, the screen's
  banking convention), at the positions of the class, as each position is computed:
  ``Add(vector, alpha)`` (``care``, ``fill``) and ``Mix(sources, beta)`` (``self``: the residual
  becomes ``(1 - beta) h + beta s`` with ``s`` the scenario's judgment-frame residual);
- **sampling**: the KDG sampler v2 (temperature 0.7, no top-k / top-p), one seeded generator per
  rollout, so a rollout reproduces whatever its batch-mates are;
- **per-rollout saving**: each finished rollout is handed to ``on_finish`` before the loop goes on;
- **coherence gate** (§5.0): parse rate >= 0.95, median trace length within 0.7-1.5x the unsteered
  arm's, and the steered traces' per-token NLL under the unsteered model within 0.3 nats of the
  unsteered traces' own;
- **fill draws** (§4 ``V_fill``): Gaussian in whitened space (covariance-matched through centred
  residual samples at the layer), norm-matched, seeded;
- **transport control** (§5.0): the ``self`` mix at the dose-0 decision token on the letter readout.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASSES = ("onset", "clause_end", "constant")
N_ONSET = 48
PUNCT = set(".;:?!\n")
TEMPERATURE = 0.7
COHERENCE = {"parse_rate_min": 0.95, "length_ratio": (0.7, 1.5), "nll_delta_max": 0.3}


@dataclass
class Add:
    """``h += alpha * vector`` at the class positions (care, fill). ``vector`` is one vector for
    every row, or ``[n_entries, hidden]``, one per entry of ``prompts`` (per rollout), so arms that
    differ only in the vector (fill's draws, the sweep's alphas) share one batch."""

    vector: np.ndarray
    alpha: float
    name: str = "add"


@dataclass
class Mix:
    """``h = (1 - beta) h + beta s_row`` at the class positions (self); one source per prompt."""

    sources: np.ndarray  # [n_prompts, hidden]
    beta: float
    name: str = "self"


def punct_mask(tok) -> np.ndarray:
    """Per vocabulary id: the token decodes to punctuation only (after trailing spaces)."""
    out = np.zeros(len(tok), dtype=bool)
    for i in range(len(tok)):
        s = tok.decode([i]).strip(" ")
        out[i] = bool(s) and set(s) <= PUNCT
    return out


class Tracker:
    """Per-row trace state during generation: where the trace opens, whether it has closed, and
    which class the position being fed belongs to."""

    def __init__(self, fmt, tok, punct: np.ndarray):
        self.fmt, self.tok, self.punct = fmt, tok, punct
        self.start: int | None | str = "pending"
        self.closed = False

    def update(self, g: list[int]) -> None:
        """After ``g[-1]`` was appended (it is the next position to be fed)."""
        if self.start == "pending":
            self.start = self.fmt.trace_open(self.tok, g)
        if not self.closed and isinstance(self.start, int) and len(g) - 1 >= self.start:
            self.closed = self.fmt.trace_closes(self.tok, g)

    def in_class(self, g: list[int], cls: str) -> bool:
        """Whether position ``len(g) - 1`` (the token ``g[-1]`` being fed) is in ``cls``."""
        i = len(g) - 1
        if not isinstance(self.start, int) or self.closed or i < self.start:
            return False
        if cls == "constant":
            return True
        if cls == "onset":
            return i - self.start < N_ONSET
        if cls == "clause_end":
            return bool(self.punct[g[i]] or (i - 1 >= self.start and self.punct[g[i - 1]]))
        raise ValueError(cls)


def generate(model, prompts: Sequence[Sequence[int]], *, fmt, cap: int, seeds: Sequence[int],
             layer: int | None = None, cls: str | None = None, intervention=None,
             prompt_index: Sequence[int] | None = None, batch: int = 8,
             on_finish: Callable[[dict], None] | None = None,
             eos: Sequence[int] = ()) -> list[dict]:
    """Sample one rollout per entry of ``prompts`` (seeded per entry), with ``intervention``
    applied at hidden-state index ``layer`` on the positions of ``cls``. ``model`` is a
    ``kdg_pod_lib.ModelWrapper`` / ``WhiteBoxModel``-backed object with ``.model``, ``.tok`` and
    ``.wb``. Returns rows with ``gen_ids``, ``n_steered`` and ``seed``."""
    import torch

    if intervention is not None and (layer is None or cls not in CLASSES):
        raise ValueError("an intervention needs a layer and a position class")
    hf, tok = model.model, model.tok
    dev = hf.device
    eos = set(eos) or set(_eos_ids(hf, tok))
    punct = punct_mask(tok)
    pidx = list(prompt_index) if prompt_index is not None else list(range(len(prompts)))
    rows_out: list[dict] = []
    for b0 in range(0, len(prompts), batch):
        P = [list(prompts[i]) for i in range(b0, min(b0 + batch, len(prompts)))]
        B = len(P)
        gens = [torch.Generator().manual_seed(int(seeds[b0 + r])) for r in range(B)]
        trk = [Tracker(fmt, tok, punct) for _ in range(B)]
        gen: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        steered = [0] * B
        state = {"mask": torch.zeros(B, dtype=torch.bool)}
        handle = None
        if intervention is not None:
            vec = None
            if isinstance(intervention, Add):
                v = np.asarray(intervention.vector)
                if v.ndim == 2:  # one vector per entry: this batch's rows
                    v = v[b0: b0 + B]
                vec = torch.as_tensor(v, dtype=hf.dtype, device=dev)
            src = None
            if isinstance(intervention, Mix):
                src = torch.as_tensor(intervention.sources[[pidx[b0 + r] for r in range(B)]],
                                      dtype=hf.dtype, device=dev)

            def hook(_m, _i, out):
                m = state["mask"]
                if not bool(m.any()):
                    return out
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                last = h[:, -1, :]
                if vec is not None:
                    last[m] = last[m] + intervention.alpha * (vec[m] if vec.dim() == 2 else vec)
                else:
                    last[m] = (1 - intervention.beta) * last[m] + intervention.beta * src[m]
                h[:, -1, :] = last
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            handle = model.wb._get_layer_module(layer - 1).register_forward_hook(hook)  # noqa: SLF001
        try:
            L = max(len(x) for x in P)
            ids = torch.full((B, L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((B, L), dtype=torch.long)
            for r, x in enumerate(P):
                ids[r, L - len(x):] = torch.tensor(x)
                att[r, L - len(x):] = 1
            ids, att = ids.to(dev), att.to(dev)
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                o = hf(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
            past, logits = o.past_key_values, o.logits[:, -1, :]
            for _step in range(cap):
                nxt = []
                for r in range(B):
                    if done[r]:
                        nxt.append(tok.pad_token_id)
                        continue
                    p = torch.softmax(logits[r].float() / TEMPERATURE, -1).cpu()
                    t = int(torch.multinomial(p, 1, generator=gens[r]))
                    gen[r].append(t)
                    trk[r].update(gen[r])
                    if t in eos or len(gen[r]) >= cap:
                        done[r] = True
                    nxt.append(t)
                if all(done):
                    break
                m = [(not done[r]) and cls is not None and trk[r].in_class(gen[r], cls)
                     for r in range(B)]
                for r in range(B):
                    steered[r] += int(m[r] and intervention is not None)
                state["mask"] = torch.tensor(m, dtype=torch.bool, device=dev)
                att = torch.cat([att, torch.ones((B, 1), dtype=torch.long, device=dev)], 1)
                pos = pos[:, -1:] + 1
                with torch.no_grad():
                    o = hf(input_ids=torch.tensor(nxt, device=dev)[:, None], attention_mask=att,
                           position_ids=pos, past_key_values=past, use_cache=True)
                past, logits = o.past_key_values, o.logits[:, -1, :]
        finally:
            if handle is not None:
                handle.remove()
        for r in range(B):
            row = {"prompt": pidx[b0 + r], "seed": int(seeds[b0 + r]), "gen_ids": gen[r],
                   "n_tokens": len(gen[r]), "hit_cap": len(gen[r]) >= cap and gen[r][-1] not in eos,
                   "n_steered": steered[r], "trace_start": trk[r].start
                   if isinstance(trk[r].start, int) else None}
            rows_out.append(row)
            if on_finish is not None:
                on_finish(row)
    return rows_out


def _eos_ids(hf, tok) -> list[int]:
    e = getattr(hf.generation_config, "eos_token_id", None)
    ids = e if isinstance(e, list) else ([e] if e is not None else [])
    return ids or [tok.eos_token_id]


def jsonl_saver(path: Path) -> Callable[[dict], None]:
    """Per-rollout saving (the KDG-A26 lesson): one line per finished rollout, flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def save(row: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return save


def fill_draws(samples: np.ndarray, norm: float, k: int, seed: int) -> np.ndarray:
    """``k`` fill vectors: Gaussian in whitened space mapped back (covariance-matched to the
    centred residual ``samples`` [n, hidden] at the layer), each scaled to ``norm``."""
    x = samples - samples.mean(0, keepdims=True)
    z = np.random.default_rng(seed).standard_normal((k, x.shape[0]))
    v = z @ x / np.sqrt(max(x.shape[0] - 1, 1))
    return (v / np.linalg.norm(v, axis=1, keepdims=True) * norm).astype(np.float32)


def trace_nll(model, prompt: Sequence[int], gen: Sequence[int]) -> float | None:
    """Mean per-token negative log-likelihood of ``gen`` after ``prompt`` under ``model`` (the
    unsteered model: the coherence gate's third bar)."""
    import torch

    if not gen:
        return None
    ids = torch.tensor([list(prompt) + list(gen)], device=model.model.device)
    with torch.no_grad():
        lp = torch.log_softmax(model.model(input_ids=ids).logits[0].float(), -1)
    n = len(prompt)
    tgt = ids[0, n:]
    return float(-lp[n - 1: n - 1 + len(gen)].gather(1, tgt[:, None]).mean())


def coherence(steered: list[dict], base: list[dict], parsed_steered: Sequence[bool],
              nll_steered: Sequence[float | None], nll_base: Sequence[float | None]) -> dict:
    """§5.0's coherence gate for one steered arm against the unsteered arm."""
    pr = float(np.mean(parsed_steered)) if len(parsed_steered) else 0.0
    ls = np.median([r["n_tokens"] for r in steered]) if steered else 0.0
    lb = np.median([r["n_tokens"] for r in base]) if base else 0.0
    ratio = float(ls / lb) if lb else float("inf")
    ns = [x for x in nll_steered if x is not None]
    nb = [x for x in nll_base if x is not None]
    dn = float(np.mean(ns) - np.mean(nb)) if ns and nb else float("inf")
    lo, hi = COHERENCE["length_ratio"]
    ok = (pr >= COHERENCE["parse_rate_min"] and lo <= ratio <= hi
          and dn <= COHERENCE["nll_delta_max"])
    return {"parse_rate": pr, "length_ratio": ratio, "nll_delta": dn, "passes": bool(ok),
            "bars": COHERENCE}


def mix_readout(model, ids: Sequence[Sequence[int]], sources: np.ndarray, beta: float,
                layer: int, letter_ids: Sequence[int]) -> np.ndarray:
    """§5.0's transport control: the dose-0 letter readout (last position, one row at a time,
    the readout mode of record) with the ``self`` mix at the decision token, block ``layer - 1``'s
    output. Returns letter log-probs [n, letters]."""
    import torch

    hf = model.model
    out = []
    for x, s in zip(ids, sources):
        st = torch.as_tensor(s, dtype=hf.dtype, device=hf.device)

        def hook(_m, _i, o, st=st):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone()
            h[:, -1, :] = (1 - beta) * h[:, -1, :] + beta * st
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hd = model.wb._get_layer_module(layer - 1).register_forward_hook(hook)  # noqa: SLF001
        try:
            with torch.no_grad():
                lg = hf(input_ids=torch.tensor([list(x)], device=hf.device),
                        logits_to_keep=1).logits[0, -1]
        finally:
            hd.remove()
        out.append(torch.log_softmax(lg.float(), -1)[list(letter_ids)].cpu().numpy())
    return np.stack(out)
