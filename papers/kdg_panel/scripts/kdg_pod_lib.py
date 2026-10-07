"""KDG pilot pod library: model wrappers, readout cells, per-rollout saving, manifest.

Every cell saves per-unit artifacts (KDG_PANEL_SPEC §9): a JSONL of rollouts (full text, parsed
option, harness label, option order, seeds) and an .npz with the full next-token log-prob vector
at the decision position (float16) plus the option-token ids, so the base floor and any
re-analysis are recomputable without a pod. Nothing here judges: verdict rules live in
``analyze_pilot.py`` and run zero-GPU from the saved arrays.

Decision position conventions (Amendment 1, 2026-09-13):
  * dose-0 D_chat and F2 final turn: the FIRST generated token (letter-only reply requested);
  * J_stated and the dose arms: the token generated immediately after the ``Answer:`` anchor;
  * raw frames: the next token after the fixed prefix ``Answer:``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from deepsteer.kdg.harness import KDG_HARNESS_VERSION, parse_response  # noqa: E402
from deepsteer.kdg.phase1_frames import (  # noqa: E402
    PHASE1_TEMPLATE_VERSION,
    framing_prefix,
    norm_salience_sentence,
    render_letter_user_message,
)
from deepsteer.kdg.phase2 import (  # noqa: E402
    PHASE2_TEMPLATE_VERSION,
    TSN_DISTANCES,
    filler_path,
    load_filler_turns,
    render_eval_letter_paraphrase,
    rotate_filler,
    tsn_messages,
)
from deepsteer.kdg.schema import (  # noqa: E402
    DOSE_CAPS,
    MAX_OPTIONS,
    TEMPLATE_VERSION,
    Scenario,
    assign_letters,
    letter_map,
    render_agent_user_message,
    render_eval_user_message,
    render_f2_final_user_message,
    render_known_gap_system_prompt,
    render_raw_prompt,
)

SEED = 0
_ANSWER_ANCHOR = re.compile(r"answer\s*[:=\-]\s*\**\s*\(?$", re.I)


# ---------------------------------------------------------------------------------------------
# small helpers (kept local so the KDG driver has no W4 coupling)
# ---------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pod syncs have no .git
        return "unknown"


def versions() -> dict:
    out = {"python": platform.python_version()}
    for m in ("numpy", "torch", "transformers"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


# ---------------------------------------------------------------------------------------------
# generation outputs
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass
class GenOut:
    text: str
    token_ids: list[int]
    logp_first: np.ndarray  # log-softmax over vocab at the first generated token
    logp_decision: np.ndarray | None  # at the answer-anchor token (None when no anchor found)
    decision_step: int  # index into token_ids of the decision token (-1 = none)


class ModelWrapper:
    """Real HF model: batched chat generation with per-step logits, raw next-token log-probs."""

    def __init__(self, repo: str, revision: str | None = None, harmony=None) -> None:
        import torch

        from deepsteer.directions.extraction import load_whitebox

        self.repo = repo
        # KDG_GPTOSS_SPEC: a kdg_harmony.HarmonyConfig switches render_chat to the pinned harmony
        # render and the letter cells to the registered prefill; None leaves every other model as is
        self.harmony = harmony
        self.reasoning_level = harmony.reasoning_level if harmony else None
        self.letter_prefill = harmony.prefill_primary if harmony else ""
        self.wb = load_whitebox(repo) if revision is None else None
        if self.wb is None:
            from deepsteer.core.model_interface import WhiteBoxModel
            from deepsteer.core.types import AccessTier

            self.wb = WhiteBoxModel(repo, access_tier=AccessTier.WEIGHTS, revision=revision)
        self.model = self.wb.model
        self.tok = self.wb.tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.device = next(self.model.parameters()).device
        self.torch = torch
        self.vocab = int(self.model.get_output_embeddings().weight.shape[0])
        self.has_chat_template = bool(getattr(self.tok, "chat_template", None))
        self.chat_template_sha = (
            sha256_text(self.tok.chat_template) if self.has_chat_template else None
        )

    # ---- tokens -----------------------------------------------------------------------------
    def token_id(self, surface: str) -> int:
        ids = self.tok.encode(surface, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"{self.repo}: option surface {surface!r} is not a single token: {ids}"
            )
        return int(ids[0])

    def render_chat(self, messages: list[dict[str, str]]) -> str:
        if not self.has_chat_template:
            raise RuntimeError(f"{self.repo} has no chat template; chat cells are instruct-only")
        if self.harmony is not None:
            import kdg_harmony

            return kdg_harmony.render(self.tok, messages, self.harmony, self.reasoning_level)
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ---- generation ---------------------------------------------------------------------------
    def generate(
        self,
        rendered: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int,
        find_anchor: bool,
        batch_size: int = 16,
    ) -> list[GenOut]:
        outs: list[GenOut] = []
        for i in range(0, len(rendered), batch_size):
            outs += self._generate_batch(
                rendered[i : i + batch_size], max_new_tokens, temperature, seed + i, find_anchor
            )
        return outs

    def _generate_batch(self, rendered, max_new_tokens, temperature, seed, find_anchor):
        torch = self.torch
        enc = self.tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(
            self.device
        )
        torch.manual_seed(seed)
        kw: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self.tok.pad_token_id,
        )
        if temperature > 0:
            kw["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        plen = enc["input_ids"].shape[1]
        new = out.sequences[:, plen:].cpu()
        # out.logits: tuple(steps) of [batch, vocab] raw (pre-warper) logits
        logps = [torch.log_softmax(step.float(), dim=-1).cpu().numpy() for step in out.logits]
        res = []
        for b in range(new.shape[0]):
            ids = [int(t) for t in new[b] if int(t) != self.tok.pad_token_id]
            text = self.tok.decode(ids, skip_special_tokens=True)
            dstep, dlogp = 0, logps[0][b]
            if find_anchor:
                dstep, dlogp = self._anchor_step(ids, logps, b)
            res.append(
                GenOut(
                    text,
                    ids,
                    logps[0][b].astype(np.float16),
                    None if dlogp is None else dlogp.astype(np.float16),
                    dstep,
                )
            )
        return res

    def _anchor_step(self, ids, logps, b):
        """Step whose preceding decoded prefix ends with 'Answer:' (the decision token)."""
        last = -1
        for k in range(1, len(ids)):
            prefix = self.tok.decode(ids[:k], skip_special_tokens=True)
            if _ANSWER_ANCHOR.search(prefix.rstrip()[-12:] if prefix else ""):
                last = k
        if last < 0 or last >= len(logps):
            return -1, None
        return last, logps[last][b]

    def raw_next_logprobs(
        self, prompts: list[str], batch_size: int = 16, add_special_tokens: bool = True
    ) -> np.ndarray:
        """Next-token log-probs after each prompt. Raw frames keep the tokenizer's special tokens
        (BOS); chat-rendered prompts pass ``add_special_tokens=False`` so the tokens match the
        generation path (``_generate_batch``), whose first-step distribution this reproduces."""
        torch = self.torch
        rows = []
        for i in range(0, len(prompts), batch_size):
            enc = self.tok(
                prompts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=add_special_tokens,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            rows.append(torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(rows, axis=0)

    def next_logprobs_and_residuals(
        self, prompts: list[str], prefills: list[str], batch_size: int = 16
    ) -> tuple[np.ndarray, np.ndarray]:
        """Next-token log-probs after ``prompt + prefill`` and the residual stream at the
        decision token (the prompt's last token) at every layer, from the same forward pass.

        Returns ``(logp [n, vocab] fp16, resid [n, n_layers + 1, hidden] fp16)``; ``resid[:, h]``
        is HF ``hidden_states[h]`` (h = 0 the embeddings, h = l + 1 the output of block l; the
        last entry is after the model's final norm where the HF implementation applies it there).
        Left padding puts every sequence's end at the last column, so the decision index is
        ``-1 - len(prefill tokens)``; it is asserted against the prompt's own last token id."""
        import kdg_harmony

        torch = self.torch
        lps, res = [], []
        for i in range(0, len(prompts), batch_size):
            P, F = prompts[i : i + batch_size], prefills[i : i + batch_size]
            splits = [kdg_harmony.split_prefill(self.tok, p, f) for p, f in zip(P, F)]
            enc = self.tok(
                [p + f for p, f in zip(P, F)],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            lps.append(
                torch.log_softmax(out.logits[:, -1, :].float(), dim=-1).cpu().numpy()
                .astype(np.float16)
            )
            for b, (a, f) in enumerate(splits):
                pos = enc["input_ids"].shape[1] - 1 - len(f)
                if int(enc["input_ids"][b, pos]) != a[-1]:
                    raise RuntimeError("decision-token index does not hold the prompt's last token")
                at = torch.stack([h[b, pos, :] for h in out.hidden_states])  # [n_layers + 1, d]
                res.append(at.float().cpu().numpy().astype(np.float16))
        return np.concatenate(lps, axis=0), np.stack(res)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def release(self) -> None:
        self.wb.release()


class StubModel:
    """Dry-run stand-in: random letters and random log-probs over a 64-token vocab, no weights."""

    def __init__(self, kind: str = "instruct", seed: int = 0) -> None:
        self.repo = f"stub-{kind}"
        self.rng = np.random.default_rng(seed)
        self.vocab = 64
        self.has_chat_template = kind != "base"
        self.chat_template_sha = "stub"
        self._ids = {L: i for i, L in enumerate("ABCDE")}
        self._ids.update({f" {L}": 10 + i for i, L in enumerate("ABCDE")})
        self.harmony = None
        self.reasoning_level = None
        self.letter_prefill = ""

    def token_id(self, surface: str) -> int:
        return self._ids[surface]

    def render_chat(self, messages):
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "\n<assistant>"

    def generate(self, rendered, *, max_new_tokens, temperature, seed, find_anchor, batch_size=16):
        outs = []
        for r in rendered:
            L = self.rng.choice(list("ABC"))
            text = (
                f"Some reasoning here.\nAnswer: {L}\nBecause of the stakes." if find_anchor else L
            )
            lp = np.log(self.rng.dirichlet(np.ones(self.vocab))).astype(np.float16)
            outs.append(
                GenOut(
                    text, [self._ids[L]], lp, lp if find_anchor else None, 3 if find_anchor else 0
                )
            )
        return outs

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)

    def encode(self, text):
        return list(range(len(text.split())))

    def raw_next_logprobs(self, prompts, batch_size=16, add_special_tokens=True):
        self.last_add_special_tokens = add_special_tokens
        return np.log(self.rng.dirichlet(np.ones(self.vocab), size=len(prompts))).astype(np.float16)

    def next_logprobs_and_residuals(self, prompts, prefills, batch_size=16):
        logp = self.raw_next_logprobs([p + f for p, f in zip(prompts, prefills)], batch_size, False)
        return logp, self.rng.standard_normal((len(prompts), 3, 8)).astype(np.float16)

    def release(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------
# manifest + context
# ---------------------------------------------------------------------------------------------


class Manifest:
    def __init__(self, out_root: Path, dry: bool, scenario_meta: list[dict]) -> None:
        self.out_root = out_root
        self.data = {
            "run_id": time.strftime("kdg_%Y%m%dT%H%M%S"),
            "dry_run": dry,
            "git_commit": git_commit(),
            "versions": versions(),
            "seed": SEED,
            "preregistration": "papers/KDG_PANEL_SPEC.md v0.4 (2026-09-13)",
            "template_version": TEMPLATE_VERSION,
            "harness_version": KDG_HARNESS_VERSION,
            "scenario_sets": scenario_meta,
            "loads": [],
            "artifacts": [],
            "unit_status": {},
        }

    def add(self, path: Path, unit: str, model_key: str) -> None:
        path = Path(path).resolve()
        self.data["artifacts"].append(
            {
                "path": str(path.relative_to(self.out_root.resolve())),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "unit": unit,
                "model": model_key,
            }
        )

    def status(self, unit: str, status: str, note: str = "") -> None:
        self.data["unit_status"][unit] = {"status": status, "note": note}

    def write(self) -> Path:
        self.out_root.mkdir(parents=True, exist_ok=True)
        p = self.out_root / "manifest_kdg.json"
        p.write_text(json.dumps(_jsonable(self.data), indent=2))
        return p


def verify_manifest(path: Path) -> list[str]:
    d = json.loads(Path(path).read_text())
    root = Path(path).resolve().parent
    bad = []
    for a in d["artifacts"]:
        p = root / a["path"]
        if not p.exists():
            bad.append(f"missing {a['path']}")
        elif sha256_file(p) != a["sha256"]:
            bad.append(f"sha mismatch {a['path']}")
    return bad


@dataclasses.dataclass
class Ctx:
    key: str
    kind: str  # instruct | base
    model: ModelWrapper | StubModel
    out: Path
    manifest: Manifest
    dry: bool
    n_d: int = 32
    n_j: int = 8
    n_dose: int = 16
    n_raw_perm: int = 8
    temperature: float = 0.7
    # KDG_GPTOSS_SPEC G-A4: decision-token residuals at every layer in the letter cells, and the
    # harmony row fields (date pin, reasoning level, reasoning_trace) on every letter row
    save_residuals: bool = False
    row_fields: dict = dataclasses.field(default_factory=dict)

    def n(self, real: int) -> int:
        return min(real, 2) if self.dry else real

    def save_cell(
        self, cell: str, rows: list[dict], logp: np.ndarray, extra: dict[str, np.ndarray]
    ) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        jl = self.out / f"{cell}.jsonl"
        with open(jl, "w") as f:
            for r in rows:
                f.write(json.dumps(_jsonable(r), ensure_ascii=False) + "\n")
        npz = self.out / f"{cell}.npz"
        np.savez_compressed(npz, logp_decision=logp.astype(np.float16), **extra)
        self.manifest.add(jl, cell, self.key)
        self.manifest.add(npz, cell, self.key)


# ---------------------------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------------------------


def _row(
    s: Scenario,
    cell: str,
    arm: str,
    i: int,
    seed: int,
    order,
    g: GenOut | None,
    parse,
    prompt: str,
    **extra,
) -> dict:
    return {
        "scenario_id": s.id,
        "role": s.role,
        "family": s.family,
        "generator": s.generator,
        "cell": cell,
        "arm": arm,
        "rollout": i,
        "seed": seed,
        "order": letter_map(order),
        "prompt_sha256": sha256_text(prompt),
        "text": None if g is None else g.text,
        "n_gen_tokens": None if g is None else len(g.token_ids),
        "decision_step": None if g is None else g.decision_step,
        "option_id": parse.option_id,
        "letter": parse.letter,
        "parse_method": parse.method,
        "norm_status": parse.norm_status,
        "harness_version": KDG_HARNESS_VERSION,
        "template_version": TEMPLATE_VERSION,
        **extra,
    }


def _option_ids(ctx: Ctx, order, chat: bool) -> list[int]:
    ids = [ctx.model.token_id(L if chat else f" {L}") for L, _ in order]
    return ids + [-1] * (MAX_OPTIONS - len(ids))


def cell_d_chat(
    ctx: Ctx,
    scenarios: list[Scenario],
    *,
    arm: str = "dose0",
    variant: str = "primary",
    n_roll: int | None = None,
) -> None:
    """D_chat (spec §4.3): agent frame, chat template, n rollouts at T=0.7, per-rollout order.

    ``variant``: primary | pressure_removed (matched null) | known_gap (positive band).
    F2 scenarios go through the multi-turn path (turn 1 free reply, fixed pushback, letter).
    """
    cell = f"d_chat_{arm}" + ("" if variant == "primary" else f"_{variant}")
    n = ctx.n(n_roll or (ctx.n_d if arm == "dose0" else ctx.n_dose))
    rows, logps, opt_ids = [], [], []
    for s in scenarios:
        orders = [assign_letters(s, seed) for seed in range(n)]
        turn1: list[str | None] = [None] * n
        if s.family == "F2":
            gens, prompts, turn1 = _f2_rollouts(ctx, s, orders, variant)
        else:
            prompts = []
            for order in orders:
                user = render_agent_user_message(
                    s, order, arm, pressure_removed=(variant == "pressure_removed")
                )
                msgs = [{"role": "user", "content": user}]
                if variant == "known_gap":
                    msgs.insert(0, {"role": "system", "content": render_known_gap_system_prompt(s)})
                prompts.append(ctx.model.render_chat(msgs))
            cap = 6 if arm == "dose0" else DOSE_CAPS[arm] + 24
            gens = ctx.model.generate(
                prompts,
                max_new_tokens=cap,
                temperature=ctx.temperature,
                seed=SEED,
                find_anchor=(arm != "dose0"),
            )
        for i, (order, g, p) in enumerate(zip(orders, gens, prompts)):
            parse = parse_response(g.text, s, order)
            rows.append(
                _row(
                    s,
                    cell,
                    arm,
                    i,
                    i,
                    order,
                    g,
                    parse,
                    p,
                    variant=variant,
                    chat_template_sha256=ctx.model.chat_template_sha,
                    turn1_text=turn1[i],
                )
            )
            lp = (
                g.logp_decision
                if (arm != "dose0" and g.logp_decision is not None)
                else g.logp_first
            )
            logps.append(lp)
            opt_ids.append(_option_ids(ctx, order, chat=True))
    ctx.save_cell(
        cell, rows, np.stack(logps), {"option_token_ids": np.array(opt_ids, dtype=np.int64)}
    )


def _f2_rollouts(ctx: Ctx, s: Scenario, orders, variant: str):
    """F2 appendix family: turn-1 free reply (T=0.7), fixed pushback, letter-only final turn.

    Returns (final-turn GenOuts, rendered final prompts, turn-1 texts)."""
    body = s.agent_text_pressure_removed if variant == "pressure_removed" else s.agent_text
    turn1 = [{"role": "user", "content": f"{body}\n\nReply to the user now."}]
    if variant == "known_gap":
        turn1.insert(0, {"role": "system", "content": render_known_gap_system_prompt(s)})
    r1 = ctx.model.render_chat(turn1)
    g1 = ctx.model.generate(
        [r1] * len(orders),
        max_new_tokens=160,
        temperature=ctx.temperature,
        seed=SEED,
        find_anchor=False,
    )
    prompts = []
    for order, g in zip(orders, g1):
        msgs = turn1 + [
            {"role": "assistant", "content": g.text},
            {"role": "user", "content": render_f2_final_user_message(s, order)},
        ]
        prompts.append(ctx.model.render_chat(msgs))
    g2 = ctx.model.generate(
        prompts, max_new_tokens=6, temperature=ctx.temperature, seed=SEED + 1, find_anchor=False
    )
    return g2, prompts, [g.text for g in g1]


def cell_j_stated(
    ctx: Ctx,
    scenarios: list[Scenario],
    *,
    variant: str = "primary",
    paraphrase_index: int | None = None,
) -> None:
    """J_stated (spec §4.1; A13 frames): third-person frame, greedy + n sampled, free text saved.

    ``variant``: primary | paraphrase (pilot single paraphrase) | pressure_removed.
    ``paraphrase_index`` (A13, 0..2) selects a paraphrase of the chosen frame; the cell is then
    named ``j_stated_p<k>`` or ``j_stated_pressure_removed_p<k>``.
    Prompts are batched ACROSS scenarios (KDG-2 driver change): all greedy prompts in one pass,
    then all sampled prompts, instead of two generate calls per scenario.
    """
    if paraphrase_index is not None:
        cell = (
            "j_stated_pressure_removed" if variant == "pressure_removed" else "j_stated"
        ) + f"_p{paraphrase_index}"
    else:
        cell = "j_stated" + ("" if variant == "primary" else f"_{variant}")
    n = ctx.n(ctx.n_j)
    kw = {
        "paraphrase": variant == "paraphrase" and paraphrase_index is None,
        "pressure_removed": variant == "pressure_removed",
        "paraphrase_index": paraphrase_index,
    }
    per_scen = []
    for s in scenarios:
        orders = [assign_letters(s, seed) for seed in range(n + 1)]
        prompts = [
            ctx.model.render_chat(
                [{"role": "user", "content": render_eval_user_message(s, o, **kw)}]
            )
            for o in orders
        ]
        per_scen.append((s, orders, prompts))
    greedy_prompts = [pr[0] for _, _, pr in per_scen]
    sampled_prompts = [q for _, _, pr in per_scen for q in pr[1:]]
    g_greedy = ctx.model.generate(
        greedy_prompts,
        max_new_tokens=220,
        temperature=0.0,
        seed=SEED,
        find_anchor=True,
        batch_size=32,
    )
    g_samp = ctx.model.generate(
        sampled_prompts,
        max_new_tokens=220,
        temperature=ctx.temperature,
        seed=SEED,
        find_anchor=True,
        batch_size=32,
    )
    rows, logps, opt_ids = [], [], []
    k = 0
    for si, (s, orders, prompts) in enumerate(per_scen):
        gens = [g_greedy[si]] + g_samp[k : k + n]
        k += n
        for i, (order, g, p) in enumerate(zip(orders, gens, prompts)):
            parse = parse_response(g.text, s, order)
            rows.append(
                _row(
                    s,
                    cell,
                    "greedy" if i == 0 else "sampled",
                    i,
                    i,
                    order,
                    g,
                    parse,
                    p,
                    variant=variant,
                    paraphrase_index=paraphrase_index,
                    chat_template_sha256=ctx.model.chat_template_sha,
                )
            )
            logps.append(
                g.logp_decision
                if g.logp_decision is not None
                else np.full_like(g.logp_first, np.nan)
            )
            opt_ids.append(_option_ids(ctx, order, chat=True))
    ctx.save_cell(
        cell, rows, np.stack(logps), {"option_token_ids": np.array(opt_ids, dtype=np.int64)}
    )


def cell_raw(ctx: Ctx, scenarios: list[Scenario], *, frame: str, variant: str = "primary") -> None:
    """Raw-frame readout (spec §4.6): next-token log-probs after the fixed ``Answer:`` prefix.

    ``frame``: agent (D_raw) | eval (J_raw). Saves the full vector per permutation plus the
    option-token ids and the option-token mass (the base-specific floor is applied at analysis).
    """
    cell = f"{'d' if frame == 'agent' else 'j'}_raw" + (
        "" if variant == "primary" else f"_{variant}"
    )
    nperm = ctx.n(ctx.n_raw_perm)
    rows, prompts, orders_all, scen_all = [], [], [], []
    for s in scenarios:
        for seed in range(nperm):
            order = assign_letters(s, seed)
            prompts.append(
                render_raw_prompt(s, order, frame, pressure_removed=(variant == "pressure_removed"))
            )
            orders_all.append(order)
            scen_all.append(s)
    logp = ctx.model.raw_next_logprobs(prompts)
    opt_ids = []
    for k, (s, order, p) in enumerate(zip(scen_all, orders_all, prompts)):
        ids = _option_ids(ctx, order, chat=False)
        opt_ids.append(ids)
        valid = [i for i in ids if i >= 0]
        lp = logp[k].astype(np.float32)
        mass = float(np.exp(lp[valid]).sum())
        # argmax over option tokens under THIS permutation, mapped back to option id
        best = int(np.argmax(lp[valid]))
        L, o = order[best]
        rows.append(
            {
                "scenario_id": s.id,
                "role": s.role,
                "family": s.family,
                "generator": s.generator,
                "cell": cell,
                "arm": "raw",
                "rollout": k % nperm,
                "seed": k % nperm,
                "order": letter_map(order),
                "prompt_sha256": sha256_text(p),
                "option_mass": mass,
                "option_logps": {LL: float(lp[i]) for (LL, _), i in zip(order, valid)},
                "option_id": o.option_id,
                "letter": L,
                "norm_status": o.norm_status,
                "parse_method": "raw_argmax",
                "harness_version": KDG_HARNESS_VERSION,
                "template_version": TEMPLATE_VERSION,
                "variant": variant,
                "frame": frame,
            }
        )
    ctx.save_cell(
        cell,
        rows,
        logp,
        {
            "option_token_ids": np.array(opt_ids, dtype=np.int64),
            "option_mass": np.array([r["option_mass"] for r in rows]),
        },
    )


def letter_chat_messages(
    s: Scenario, order, frame: str, prefix: str, variant: str = "primary"
) -> list[dict[str, str]]:
    """The exact message list a letter-only chat unit renders (also what the stage identity check
    renders, so the check covers every scheduled unit's shapes; KDG_F6_F8_SPEC §8 G7).

    variant: primary | pressure_removed | known_gap (system + user) | pressure_removed_p{0,1,2}
    (Phase 2 G2: letter-only J on a paraphrase of the no-pressure third-person text)."""
    if variant.startswith("pressure_removed_p"):
        assert frame == "eval", "paraphrase frames are judgment (eval) frames"
        idx = int(variant[-1])
        user = f"{framing_prefix(prefix)}\n\n{render_eval_letter_paraphrase(s, order, idx)}"
        return [{"role": "user", "content": user}]
    user = render_letter_user_message(
        s, order, frame, prefix, pressure_removed=(variant == "pressure_removed")
    )
    msgs = [{"role": "user", "content": user}]
    if variant == "known_gap":
        msgs.insert(0, {"role": "system", "content": render_known_gap_system_prompt(s)})
    return msgs


def _letter_cell(
    ctx: Ctx,
    scenarios: list[Scenario],
    cell: str,
    build,
    row_extra: dict,
    template_version: str = PHASE1_TEMPLATE_VERSION,
) -> None:
    """Shared letter-only readout: one forward pass per option permutation, next-token log-probs
    at the first assistant token; full vector, option ids and mass, rendered-prompt sha saved."""
    nperm = ctx.n(ctx.n_raw_perm)
    prompts, orders_all, scen_all = [], [], []
    for s in scenarios:
        for seed in range(nperm):
            order = assign_letters(s, seed)
            prompts.append(ctx.model.render_chat(build(s, order)))
            orders_all.append(order)
            scen_all.append(s)
    prefill = ctx.model.letter_prefill
    extra: dict[str, np.ndarray] = {}
    if ctx.save_residuals:
        logp, extra["resid_decision"] = ctx.model.next_logprobs_and_residuals(
            prompts, [prefill] * len(prompts)
        )
    else:
        assert not prefill, "a prefilled readout goes through next_logprobs_and_residuals"
        logp = ctx.model.raw_next_logprobs(prompts, add_special_tokens=False)
    rows, opt_ids = [], []
    for k, (s, order, p) in enumerate(zip(scen_all, orders_all, prompts)):
        ids = _option_ids(ctx, order, chat=True)
        opt_ids.append(ids)
        valid = [i for i in ids if i >= 0]
        lp = logp[k].astype(np.float32)
        best = int(np.argmax(lp[valid]))
        L, o = order[best]
        rows.append(
            {
                "scenario_id": s.id,
                "role": s.role,
                "family": s.family,
                "generator": s.generator,
                "cell": cell,
                "arm": "letter_chat",
                "rollout": k % nperm,
                "seed": k % nperm,
                "order": letter_map(order),
                "prompt_sha256": sha256_text(p + prefill),
                "option_mass": float(np.exp(lp[valid]).sum()),
                "option_logps": {LL: float(lp[i]) for (LL, _), i in zip(order, valid)},
                "option_id": o.option_id,
                "letter": L,
                "norm_status": o.norm_status,
                "action_type": o.action_type,
                "level": s.level,
                "nudge": s.nudge,
                "item_id": s.item_id,
                "parse_method": "letter_chat_argmax",
                "harness_version": KDG_HARNESS_VERSION,
                "template_version": template_version,
                "chat_template_sha256": ctx.model.chat_template_sha,
                **ctx.row_fields,
                **row_extra,
            }
        )
    ctx.save_cell(
        cell,
        rows,
        logp,
        {
            "option_token_ids": np.array(opt_ids, dtype=np.int64),
            "option_mass": np.array([r["option_mass"] for r in rows]),
            **extra,
        },
    )


def cell_letter_chat(
    ctx: Ctx,
    scenarios: list[Scenario],
    *,
    frame: str,
    prefix: str,
    variant: str = "primary",
) -> None:
    """Phase 1 C1 / C3-secondary readout: letter-only J or D under the chat template, one forward
    pass per option permutation (no sampling), next-token log-probs at the first assistant token.

    Cell name ``{dl|jl}_chat_{prefix}[_variant]``. Same permutation seeds as the raw cells
    (0..n_raw_perm-1), so orders match the raw frame and the first eight D_chat rollouts. Saves the
    full next-token vector per permutation, option-token ids and mass, and the rendered-prompt
    sha (the stage-identity check compares these across checkpoints).
    """
    known_gap = variant == "known_gap"  # P1-A9 positive control: operator orders the violation
    if known_gap:
        assert frame == "agent", "the known-gap control is an acting-frame cell"
        cell = "dl_chat_known_gap"
        scenarios = [s for s in scenarios if s.role == "primary"]
    else:
        cell = f"{'dl' if frame == 'agent' else 'jl'}_chat_{prefix}" + (
            "" if variant == "primary" else f"_{variant}"
        )
    if variant.startswith("pressure_removed_p"):
        scenarios = [s for s in scenarios if len(s.eval_text_pressure_removed_paraphrases) == 3]
    _letter_cell(
        ctx,
        scenarios,
        cell,
        lambda s, order: letter_chat_messages(s, order, frame, prefix, variant),
        {"variant": variant, "frame": frame, "prefix": prefix},
        PHASE2_TEMPLATE_VERSION
        if variant.startswith("pressure_removed_p")
        else PHASE1_TEMPLATE_VERSION,
    )


def tsn_cell_name(
    kind: str, k: int, pressure_removed: bool, length: int = 138, rotate: bool = False
) -> str:
    tag = "dl_tsn" + ("" if length == 138 else str(length)) + ("rot" if rotate else "")
    return f"{tag}_{kind}_k{k}" + ("_pressure_removed" if pressure_removed else "")


def tsn_build(kind: str, k: int, pressure_removed: bool, length: int = 138, rotate: bool = False):
    """Message builder for a turns-since-norm unit (P2-A4: filler length and rotation)."""
    filler = load_filler_turns(filler_path(length))

    def build(s: Scenario, order) -> list[dict[str, str]]:
        f = rotate_filler(filler, s.id) if rotate else filler
        return tsn_messages(s, order, k, kind, f, pressure_removed=pressure_removed)

    return build


def cell_tsn(
    ctx: Ctx,
    scenarios: list[Scenario],
    *,
    kind: str,
    k: int,
    pressure_removed: bool,
    length: int = 138,
    rotate: bool = False,
) -> None:
    """KDG_F6_F8_SPEC §4 turns-since-norm arm (+ P2-A4 follow-ups): letter-only D after a fixed
    multi-turn conversation with the norm reminder (``kind='reminder'``) or its matched neutral
    sentence (``'neutral'``) k user turns before the decision. Assistant turns are prefilled, never
    generated. ``length`` picks the filler set (138 / 600 / 2,000 tokens); ``rotate`` applies the
    per-scenario cyclic filler order."""
    _letter_cell(
        ctx,
        scenarios,
        tsn_cell_name(kind, k, pressure_removed, length, rotate),
        tsn_build(kind, k, pressure_removed, length, rotate),
        {
            "variant": "tsn",
            "frame": "agent",
            "tsn_kind": kind,
            "tsn_k": k,
            "pressure_removed": pressure_removed,
            "tsn_filler_tokens": length,
            "tsn_rotated": rotate,
        },
        PHASE2_TEMPLATE_VERSION,
    )


def forced_reasoning_text(model, g: GenOut, budget: int) -> tuple[str, bool, int]:
    """P1-A5: reasoning R before the natural ``Answer:`` anchor if it occurs within ``budget``
    generated tokens, else the first ``budget`` tokens. Returns (R, natural_anchor, n_tokens)."""
    if g.decision_step is not None and 0 < g.decision_step <= budget:
        text = model.decode(g.token_ids[: g.decision_step])
        text = _ANSWER_ANCHOR.sub("", text.rstrip())
        return text.rstrip(), True, g.decision_step
    n = min(budget, len(g.token_ids))
    return model.decode(g.token_ids[:n]).rstrip(), False, n


def cell_dose_forced(
    ctx: Ctx,
    scenarios: list[Scenario],
    *,
    arm: str,
    budget: int | None = None,
    n_roll: int | None = None,
    suffix: str = "_bf",
    variant: str = "primary",
) -> None:
    """P1-A5 budget-forced dose arm. Generation exactly as ``cell_d_chat`` (same instruction, T,
    seeds, per-rollout order) with generation cap ``budget + 24``; then, per rollout, a forced
    forward pass on prompt + R + "\n\nAnswer:" (``forced_reasoning_text``). Saves the natural
    cell ``d_chat_{arm}{suffix}`` (as ``cell_d_chat``) and ``d_chat_{arm}{suffix}_forced`` (full
    next-token vector at the forced anchor, per rollout). F2 is skipped (no gate family).
    ``variant="pressure_removed"`` (P1-A12) runs the same arm on the twins: same seeds and orders,
    cells ``d_chat_{arm}{suffix}_pressure_removed`` and ``..._pressure_removed_forced``."""
    budget = budget or DOSE_CAPS[arm]
    n = ctx.n(n_roll or ctx.n_dose)
    pr = variant == "pressure_removed"
    cell = f"d_chat_{arm}{suffix}" + ("_pressure_removed" if pr else "")
    rows, logp_nat, fr_rows, fr_prompts, opt_ids = [], [], [], [], []
    for s in [x for x in scenarios if x.family != "F2"]:
        orders = [assign_letters(s, seed) for seed in range(n)]
        prompts = [
            ctx.model.render_chat(
                [
                    {
                        "role": "user",
                        "content": render_agent_user_message(s, o, arm, pressure_removed=pr),
                    }
                ]
            )
            for o in orders
        ]
        gens = ctx.model.generate(
            prompts,
            max_new_tokens=budget + 24,
            temperature=ctx.temperature,
            seed=SEED,
            find_anchor=True,
        )
        for i, (order, g, p) in enumerate(zip(orders, gens, prompts)):
            parse = parse_response(g.text, s, order)
            base = _row(
                s,
                cell,
                arm,
                i,
                i,
                order,
                g,
                parse,
                p,
                variant=variant,
                chat_template_sha256=ctx.model.chat_template_sha,
                budget=budget,
            )
            rows.append(base)
            logp_nat.append(g.logp_decision if g.logp_decision is not None else g.logp_first)
            opt_ids.append(_option_ids(ctx, order, chat=True))
            text, natural, n_tok = forced_reasoning_text(ctx.model, g, budget)
            fp = p + text + "\n\nAnswer:"
            fr_prompts.append(fp)
            fr_rows.append(
                {
                    **base,
                    "cell": cell + "_forced",
                    "prompt_sha256": sha256_text(fp),
                    "forced_natural_anchor": natural,
                    "forced_reasoning_tokens": n_tok,
                    "forced_truncated": not natural,
                    "parse_method": "forced_anchor",
                }
            )
    ids = np.array(opt_ids, dtype=np.int64)
    ctx.save_cell(cell, rows, np.stack(logp_nat), {"option_token_ids": ids})
    forced = ctx.model.raw_next_logprobs(fr_prompts, add_special_tokens=False)
    ctx.save_cell(cell + "_forced", fr_rows, forced, {"option_token_ids": ids})


DOSE_TEXTS = REPO / "papers" / "kdg_panel" / "data" / "dose_bf_rollout_texts.jsonl.gz"
_ANSWER_ANYWHERE = re.compile(r"answer\s*[:=\-]", re.I)


def reasoning_before_answer(text: str) -> tuple[str, bool]:
    """P1-A8: the saved rollout text before its last 'Answer:' line (natural anchor), else all."""
    hits = list(_ANSWER_ANYWHERE.finditer(text))
    if not hits:
        return text.rstrip(), False
    return text[: hits[-1].start()].rstrip(), True


def cell_dose_control(
    ctx: Ctx,
    scenarios: list[Scenario],
    *,
    kind: str,
    source: str = "committed",
    variant: str = "primary",
) -> None:
    """P1-A8 forward-pass controls on the committed P1-A5 filler rollouts.

    kind "tf": the filler reasoning cut mid-text at floor(0.75 n) of its own tokens, forced
    "\n\nAnswer:". kind "ns": a fixed norm-naming sentence, then the filler reasoning, forced.
    Prompts are the original filler prompts (order re-derived from the rollout seed and asserted
    equal to the saved order). Cell ``d_chat_dose2_filler_{kind}_forced``. ``variant=
    "pressure_removed"`` (P1-A12, ``source="own"`` only) reads the twin filler rollouts and renders
    the twin prompts; cell ``d_chat_dose2_filler_{kind}_pressure_removed_forced``.
    """
    pr = variant == "pressure_removed"
    if pr and source != "own":
        raise ValueError("pressure_removed controls read the model's own twin filler rollouts")
    sfx = "_pressure_removed" if pr else ""
    import gzip

    by_id = {s.id: s for s in scenarios if s.family != "F2"}
    if ctx.dry:  # synthetic rows: stub scenarios reuse real ids with different options
        rows_in = [
            {
                "scenario_id": s.id,
                "rollout": i,
                "order": letter_map(assign_letters(s, i)),
                "text": "restated situation words here and more words\nAnswer: A",
                "variant": variant,
            }
            for s in list(by_id.values())[:2]
            for i in range(2)
        ]
    elif source == "own":  # P1-A9: this model's own filler rollouts, written earlier in the step
        src = ctx.out / f"d_chat_dose2_filler_bf{sfx}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"{src}: run d_chat_dose2_filler_bf before the own-rollout TF")
        rows_in = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        rows_in = [r for r in rows_in if r["scenario_id"] in by_id]
    else:
        rows_in = []
        for line in gzip.open(DOSE_TEXTS, "rt"):
            r = json.loads(line)
            if r["arm"] == "dose2_filler" and r["scenario_id"] in by_id:
                rows_in.append(r)
    cell = f"d_chat_dose2_filler_{kind}{sfx}_forced"
    prompts, rows, opt_ids = [], [], []
    for r in rows_in:
        s = by_id[r["scenario_id"]]
        order = assign_letters(s, int(r["rollout"]))
        # "the control reads the rollout under its own option order": fail loudly on drift
        assert letter_map(order) == r["order"], (s.id, r["rollout"], "order drift")
        user = render_agent_user_message(s, order, "dose2_filler", pressure_removed=pr)
        base = ctx.model.render_chat([{"role": "user", "content": user}])
        # "the twin control reads twin rollouts": the source row must carry the same variant
        assert not pr or r.get("variant") == "pressure_removed", (s.id, "primary rollout read")
        text, natural = reasoning_before_answer(r["text"])
        if kind == "tf":
            ids = ctx.model.encode(text)
            n_cut = int(np.floor(0.75 * len(ids)))
            body = ctx.model.decode(ids[:n_cut]).rstrip()
        elif kind == "ns":
            n_cut = None
            body = f"{norm_salience_sentence(s.norm_class)}\n\n{text}"
        else:
            raise ValueError(kind)
        fp = base + body + "\n\nAnswer:"
        prompts.append(fp)
        opt_ids.append(_option_ids(ctx, order, chat=True))
        rows.append(
            {
                "scenario_id": s.id,
                "role": s.role,
                "family": s.family,
                "generator": s.generator,
                "cell": cell,
                "arm": "dose2_filler",
                "control": kind,
                "variant": variant,
                "rollout": int(r["rollout"]),
                "seed": int(r["rollout"]),
                "order": letter_map(order),
                "prompt_sha256": sha256_text(fp),
                "natural_anchor_in_source": natural,
                "tf_tokens": n_cut,
                "norm_class": s.norm_class,
                "harness_version": KDG_HARNESS_VERSION,
                "template_version": PHASE1_TEMPLATE_VERSION,
                "chat_template_sha256": ctx.model.chat_template_sha,
            }
        )
    logp = ctx.model.raw_next_logprobs(prompts, add_special_tokens=False)
    ctx.save_cell(cell, rows, logp, {"option_token_ids": np.array(opt_ids, dtype=np.int64)})


def rendered_identity_mismatches(
    model: ModelWrapper | StubModel,
    reference_render,
    scenarios: list[Scenario],
    n: int = 16,
    *,
    units: list[str] | None = None,
) -> list[str]:
    """Stage-checkpoint fork check (models.yaml phase1 header; KDG_F6_F8_SPEC §8 G7).

    ``units=None`` keeps the Phase 1 check (single user turn, agent and eval frames; ids
    ``{scenario}/{frame}``). With ``units``, every scheduled chat unit's own message lists are
    rendered by ``model`` and by ``reference_render`` (system turns, prefilled multi-turn
    conversations included) and mismatches are returned as ``{unit}/{scenario}``. A scheduled chat
    unit with no registered message builder raises: the check never skips a shape silently."""
    bad = []
    if units is None:
        for s in scenarios[:n]:
            order = assign_letters(s, 0)
            for frame in ("agent", "eval"):
                user = render_letter_user_message(s, order, frame, "neutral")
                msgs = [{"role": "user", "content": user}]
                if model.render_chat(msgs) != reference_render(msgs):
                    bad.append(f"{s.id}/{frame}")
        return bad
    for u in units:
        if u in RAW_UNITS or u == "validate_forward_matches_generate":
            continue
        if u not in UNIT_MESSAGES:
            raise KeyError(
                f"no message builder for chat unit {u!r}; the identity check cannot cover it"
            )
        build, keep = UNIT_MESSAGES[u]
        for s in [s for s in scenarios if keep(s)][:n]:
            msgs = build(s, assign_letters(s, 0))
            if model.render_chat(msgs) != reference_render(msgs):
                bad.append(f"{u}/{s.id}")
    return bad


def reference_renderer(repo: str, revision: str | None):
    """Tokenizer-only chat renderer for the reference checkpoint (no weights loaded)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    return lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def cell_forward_matches_generate(ctx: Ctx, scenarios: list[Scenario], n: int = 8) -> None:
    """VALIDATE-stage check for the letter-chat readout: the forward-pass next-token log-probs on
    the dose-0 agent prompt must equal the generation path's first-step log-probs (same prompt,
    greedy). Raises on a mismatch > 0.05 nats on the option tokens; saves both for the record."""
    from deepsteer.kdg.schema import render_agent_user_message as _agent

    prompts, orders = [], []
    for s in scenarios[:n]:
        order = assign_letters(s, 0)
        prompts.append(ctx.model.render_chat([{"role": "user", "content": _agent(s, order)}]))
        orders.append(order)
    # KDG_F6_F8_SPEC §8 G6: one F6 message-board prompt, one F8 five-option prompt and one
    # turns-since-norm k = 6 conversation, when the loaded set carries them
    extra = [x for x in scenarios if x.family == "F6" and x.level == "peer"][:1]
    extra += [x for x in scenarios if x.family == "F8"][:1]
    for s in extra:
        order = assign_letters(s, 0)
        prompts.append(ctx.model.render_chat([{"role": "user", "content": _agent(s, order)}]))
        orders.append(order)
    if scenarios:
        s = scenarios[0]
        order = assign_letters(s, 0)
        for length in (138, 2000):  # G6 also on the longest P2-A4b context
            msgs = tsn_messages(s, order, 6, "reminder", load_filler_turns(filler_path(length)))
            prompts.append(ctx.model.render_chat(msgs))
            orders.append(order)
    fwd = ctx.model.raw_next_logprobs(prompts, add_special_tokens=False).astype(np.float32)
    gen = ctx.model.generate(
        prompts, max_new_tokens=1, temperature=0.0, seed=SEED, find_anchor=False
    )
    worst = 0.0
    for k, (order, g) in enumerate(zip(orders, gen)):
        ids = [i for i in _option_ids(ctx, order, chat=True) if i >= 0]
        first = g.logp_first.astype(np.float32)[ids]
        worst = max(worst, float(np.max(np.abs(fwd[k][ids] - first))))
    ctx.out.mkdir(parents=True, exist_ok=True)
    rec = {"max_abs_nats": worst, "n": len(prompts)}
    (ctx.out / "forward_matches_generate.json").write_text(json.dumps(rec))
    if not ctx.dry and worst > 0.05:
        raise RuntimeError(
            f"forward-pass readout differs from generation's first step by {worst:.3f} nats"
        )


# unit registry: name -> (kinds it runs on, callable(ctx, scenarios))
UNITS: dict[str, tuple[tuple[str, ...], Any]] = {
    "d_chat_dose0": (("instruct",), lambda c, S: cell_d_chat(c, S)),
    "d_chat_dose0_pressure_removed": (
        ("instruct",),
        lambda c, S: cell_d_chat(c, S, variant="pressure_removed"),
    ),
    "d_chat_dose0_known_gap": (
        ("instruct",),
        lambda c, S: cell_d_chat(c, [s for s in S if s.role == "primary"], variant="known_gap"),
    ),
    "j_stated": (("instruct",), lambda c, S: cell_j_stated(c, S)),
    "j_stated_paraphrase": (("instruct",), lambda c, S: cell_j_stated(c, S, variant="paraphrase")),
    "j_stated_pressure_removed": (
        ("instruct",),
        lambda c, S: cell_j_stated(c, S, variant="pressure_removed"),
    ),
    # A13 four-frame reference: p0 == the pilot's single paraphrase, p1/p2 new; null frame too
    "j_stated_p0": (("instruct",), lambda c, S: cell_j_stated(c, S, paraphrase_index=0)),
    "j_stated_p1": (("instruct",), lambda c, S: cell_j_stated(c, S, paraphrase_index=1)),
    "j_stated_p2": (("instruct",), lambda c, S: cell_j_stated(c, S, paraphrase_index=2)),
    "j_stated_pressure_removed_p0": (
        ("instruct",),
        lambda c, S: cell_j_stated(c, S, variant="pressure_removed", paraphrase_index=0),
    ),
    "j_stated_pressure_removed_p1": (
        ("instruct",),
        lambda c, S: cell_j_stated(c, S, variant="pressure_removed", paraphrase_index=1),
    ),
    "j_stated_pressure_removed_p2": (
        ("instruct",),
        lambda c, S: cell_j_stated(c, S, variant="pressure_removed", paraphrase_index=2),
    ),
    "d_raw": (("instruct", "base"), lambda c, S: cell_raw(c, S, frame="agent")),
    "j_raw": (("instruct", "base"), lambda c, S: cell_raw(c, S, frame="eval")),
    "d_raw_pressure_removed": (
        ("instruct", "base"),
        lambda c, S: cell_raw(c, S, frame="agent", variant="pressure_removed"),
    ),
    "j_raw_pressure_removed": (
        ("instruct", "base"),
        lambda c, S: cell_raw(c, S, frame="eval", variant="pressure_removed"),
    ),
    # dose arms (spec §4.5): screened scenarios only, panel stage; caps fixed in DOSE_CAPS
    "d_chat_dose1": (("instruct",), lambda c, S: cell_d_chat(c, S, arm="dose1")),
    "d_chat_dose2": (("instruct",), lambda c, S: cell_d_chat(c, S, arm="dose2")),
    "d_chat_dose2_filler": (("instruct",), lambda c, S: cell_d_chat(c, S, arm="dose2_filler")),
    # Phase 1 (KDG_PHASE1_SPEC.md): VALIDATE check for the forward-pass chat readout
    "validate_forward_matches_generate": (
        ("instruct",),
        lambda c, S: cell_forward_matches_generate(c, S),
    ),
}


def _letter_unit(frame: str, prefix: str, variant: str):
    return lambda c, S: cell_letter_chat(c, S, frame=frame, prefix=prefix, variant=variant)


# Phase 1 P1-A5 budget-forced dose arms and the 2,048-token descriptive rider
UNITS["d_chat_dose1_bf"] = (("instruct",), lambda c, S: cell_dose_forced(c, S, arm="dose1"))
UNITS["d_chat_dose2_bf"] = (("instruct",), lambda c, S: cell_dose_forced(c, S, arm="dose2"))
UNITS["d_chat_dose2_filler_bf"] = (
    ("instruct",),
    lambda c, S: cell_dose_forced(c, S, arm="dose2_filler"),
)
UNITS["d_chat_dose2_long"] = (
    ("instruct",),
    lambda c, S: cell_dose_forced(c, S, arm="dose2", budget=2048, n_roll=8, suffix="_long8"),
)
UNITS["d_chat_dose2_filler_long"] = (
    ("instruct",),
    lambda c, S: cell_dose_forced(c, S, arm="dose2_filler", budget=2048, n_roll=8, suffix="_long8"),
)
DOSE_BF_UNITS = ("d_chat_dose1_bf", "d_chat_dose2_bf", "d_chat_dose2_filler_bf")
UNITS["dose_ctrl_tf"] = (("instruct",), lambda c, S: cell_dose_control(c, S, kind="tf"))
UNITS["dose_ctrl_ns"] = (("instruct",), lambda c, S: cell_dose_control(c, S, kind="ns"))
DOSE_CTRL_UNITS = ("dose_ctrl_tf", "dose_ctrl_ns")
UNITS["dose_ctrl_tf_own"] = (
    ("instruct",),
    lambda c, S: cell_dose_control(c, S, kind="tf", source="own"),
)
# P1-A12 (GPU-1): the dose2 / filler forced arms and the own truncated filler on the twins
UNITS["d_chat_dose2_bf_pressure_removed"] = (
    ("instruct",),
    lambda c, S: cell_dose_forced(c, S, arm="dose2", variant="pressure_removed"),
)
UNITS["d_chat_dose2_filler_bf_pressure_removed"] = (
    ("instruct",),
    lambda c, S: cell_dose_forced(c, S, arm="dose2_filler", variant="pressure_removed"),
)
UNITS["dose_ctrl_tf_own_pressure_removed"] = (
    ("instruct",),
    lambda c, S: cell_dose_control(c, S, kind="tf", source="own", variant="pressure_removed"),
)
DOSE_TWIN_UNITS = (
    "d_chat_dose2_bf_pressure_removed",
    "d_chat_dose2_filler_bf_pressure_removed",
    "dose_ctrl_tf_own_pressure_removed",
)
UNITS["dl_chat_known_gap"] = (
    ("instruct",),
    lambda c, S: cell_letter_chat(c, S, frame="agent", prefix="neutral", variant="known_gap"),
)
DOSE_LONG_UNITS = ("d_chat_dose2_long", "d_chat_dose2_filler_long")

# Phase 1 C1 / C3-secondary letter-only chat cells:
#   {dl|jl}_chat_{eval|deploy|neutral}[_pressure_removed]
for _pre in ("eval", "deploy", "neutral"):
    for _frame, _tag in (("agent", "dl"), ("eval", "jl")):
        for _var in ("primary", "pressure_removed"):
            _name = f"{_tag}_chat_{_pre}" + ("" if _var == "primary" else "_pressure_removed")
            UNITS[_name] = (
                ("instruct",),
                _letter_unit(_frame, _pre, _var),
            )
C1_UNITS = tuple(
    f"{t}_chat_{p}{v}"
    for p in ("eval", "deploy", "neutral")
    for t in ("dl", "jl")
    for v in ("", "_pressure_removed")
)
C3_CHAT_UNITS = (
    "dl_chat_neutral",
    "jl_chat_neutral",
    "dl_chat_neutral_pressure_removed",
    "jl_chat_neutral_pressure_removed",
)
RAW_UNITS = ("d_raw", "j_raw", "d_raw_pressure_removed", "j_raw_pressure_removed")
DOSE_UNITS = ("d_chat_dose1", "d_chat_dose2", "d_chat_dose2_filler")
PILOT_UNITS_INSTRUCT = (
    "d_chat_dose0",
    "j_stated",
    "j_stated_paraphrase",
    "d_chat_dose0_pressure_removed",
    "j_stated_pressure_removed",
    "d_chat_dose0_known_gap",
    "d_raw",
    "j_raw",
    "d_raw_pressure_removed",
    "j_raw_pressure_removed",
)
PILOT_UNITS_BASE = ("d_raw", "j_raw", "d_raw_pressure_removed", "j_raw_pressure_removed")
# KDG-2 (full panel, A13): the pilot cells minus the single-paraphrase cell, plus the six A13 frames
KDG2_UNITS_INSTRUCT = (
    "d_chat_dose0",
    "j_stated",
    "j_stated_p0",
    "j_stated_p1",
    "j_stated_p2",
    "d_chat_dose0_pressure_removed",
    "j_stated_pressure_removed",
    "j_stated_pressure_removed_p0",
    "j_stated_pressure_removed_p1",
    "j_stated_pressure_removed_p2",
    "d_chat_dose0_known_gap",
    "d_raw",
    "j_raw",
    "d_raw_pressure_removed",
    "j_raw_pressure_removed",
)


# ---------------------------------------------------------------------------------------------
# Phase 2 (KDG_F6_F8_SPEC.md): G2 paraphrase-J units, turns-since-norm units, message builders
# ---------------------------------------------------------------------------------------------
P2_PARA_UNITS = tuple(f"jl_chat_neutral_pressure_removed_p{i}" for i in range(3))
for _i in range(3):
    UNITS[P2_PARA_UNITS[_i]] = (
        ("instruct",),
        _letter_unit("eval", "neutral", f"pressure_removed_p{_i}"),
    )


def _tsn_unit(kind: str, k: int, pr: bool):
    return lambda c, S: cell_tsn(c, S, kind=kind, k=k, pressure_removed=pr)


TSN_UNITS = tuple(
    tsn_cell_name(kind, k, pr)
    for kind in ("reminder", "neutral")
    for k in TSN_DISTANCES
    for pr in (False, True)
)
for _kind in ("reminder", "neutral"):
    for _k in TSN_DISTANCES:
        for _pr in (False, True):
            UNITS[tsn_cell_name(_kind, _k, _pr)] = (("instruct",), _tsn_unit(_kind, _k, _pr))


# P2-A4a: counterbalanced filler order (Llama, k 0/3/6) and P2-A4b: token-distance ladder (k 0/6)
def _tsn_unit2(kind: str, k: int, pr: bool, length: int, rotate: bool):
    return lambda c, S: cell_tsn(
        c, S, kind=kind, k=k, pressure_removed=pr, length=length, rotate=rotate
    )


TSN_ROT_UNITS = tuple(
    tsn_cell_name(kd, k, pr, 138, True)
    for kd in ("reminder", "neutral")
    for k in (0, 3, 6)
    for pr in (False, True)
)
TSN_LEN_UNITS = tuple(
    tsn_cell_name(kd, k, pr, L, False)
    for L in (600, 2000)
    for kd in ("reminder", "neutral")
    for k in (0, 6)
    for pr in (False, True)
)
for _kd in ("reminder", "neutral"):
    for _pr in (False, True):
        for _k in (0, 3, 6):
            UNITS[tsn_cell_name(_kd, _k, _pr, 138, True)] = (
                ("instruct",),
                _tsn_unit2(_kd, _k, _pr, 138, True),
            )
        for _L in (600, 2000):
            for _k in (0, 6):
                UNITS[tsn_cell_name(_kd, _k, _pr, _L, False)] = (
                    ("instruct",),
                    _tsn_unit2(_kd, _k, _pr, _L, False),
                )

# P2 pilot keystone (§10): J and D on every level, the null twins, G2 paraphrase frames, known-gap
P2_PILOT_UNITS = C3_CHAT_UNITS + P2_PARA_UNITS + ("dl_chat_known_gap",)


def _all(s: Scenario) -> bool:
    return True


# unit -> (message builder(scenario, order), scenario filter); the G7 identity check renders these
UNIT_MESSAGES: dict[str, tuple[Any, Any]] = {}
for _pre in ("eval", "deploy", "neutral"):
    for _frame, _tag in (("agent", "dl"), ("eval", "jl")):
        for _var in ("primary", "pressure_removed"):
            _name = f"{_tag}_chat_{_pre}" + ("" if _var == "primary" else "_pressure_removed")
            UNIT_MESSAGES[_name] = (
                (lambda f, p, v: lambda s, o: letter_chat_messages(s, o, f, p, v))(
                    _frame, _pre, _var
                ),
                _all,
            )
UNIT_MESSAGES["dl_chat_known_gap"] = (
    lambda s, o: letter_chat_messages(s, o, "agent", "neutral", "known_gap"),
    lambda s: s.role == "primary",
)
for _i in range(3):
    UNIT_MESSAGES[P2_PARA_UNITS[_i]] = (
        (lambda v: lambda s, o: letter_chat_messages(s, o, "eval", "neutral", v))(
            f"pressure_removed_p{_i}"
        ),
        lambda s: len(s.eval_text_pressure_removed_paraphrases) == 3,
    )
for _kd in ("reminder", "neutral"):
    for _pr in (False, True):
        for _k in (0, 3, 6):
            UNIT_MESSAGES[tsn_cell_name(_kd, _k, _pr, 138, True)] = (
                tsn_build(_kd, _k, _pr, 138, True),
                _all,
            )
        for _L in (600, 2000):
            for _k in (0, 6):
                UNIT_MESSAGES[tsn_cell_name(_kd, _k, _pr, _L, False)] = (
                    tsn_build(_kd, _k, _pr, _L, False),
                    _all,
                )
for _kind in ("reminder", "neutral"):
    for _k in TSN_DISTANCES:
        for _pr in (False, True):
            UNIT_MESSAGES[tsn_cell_name(_kind, _k, _pr)] = (
                (
                    lambda kd, kk, pr: (
                        lambda s, o: tsn_messages(
                            s, o, kk, kd, load_filler_turns(), pressure_removed=pr
                        )
                    )
                )(_kind, _k, _pr),
                _all,
            )
