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
from deepsteer.kdg.schema import (  # noqa: E402
    DOSE_CAPS,
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

    def __init__(self, repo: str, revision: str | None = None) -> None:
        import torch

        from deepsteer.directions.extraction import load_whitebox

        self.repo = repo
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

    def raw_next_logprobs(self, prompts: list[str], batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        rows = []
        for i in range(0, len(prompts), batch_size):
            enc = self.tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            rows.append(torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float16))
        return np.concatenate(rows, axis=0)

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
        self._ids = {L: i for i, L in enumerate("ABCD")}
        self._ids.update({f" {L}": 10 + i for i, L in enumerate("ABCD")})

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

    def raw_next_logprobs(self, prompts, batch_size=16):
        return np.log(self.rng.dirichlet(np.ones(self.vocab), size=len(prompts))).astype(np.float16)

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
    return ids + [-1] * (4 - len(ids))


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
}
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
