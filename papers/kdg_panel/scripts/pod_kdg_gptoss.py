#!/usr/bin/env python3
"""KDG Tier 2 GPT-OSS-20B session (KDG_GPTOSS_SPEC v0.2): one model load, gates first.

Order (spec §3, G-A4):
  1. GATES (bail on any failure, exit 2, no cells): resolved revision and template sha equal the
     registry; option letters A..E single tokens; the date pin replaces the template's line exactly
     once and leaves the ``Knowledge cutoff:`` line byte-identical (G-A5); mxfp4 experts dequantized
     to bf16; forward pass equals generation's first step on both prefilled final channels
     (<= 0.05 nats on the option tokens).
  2. ``--validate``: also time one full letter-readout unit and one C0 generation batch, project
     the whole run, write ``timing.json`` and exit; exit 3 (stop and report before launch) if the
     projection exceeds ``--max-hours`` (2.6 A100-h = 2x the spec's 1.3 h).
  3. C0 readout construct check on 64 union scenarios (seed 0): both forced readouts and
     low-effort generation (4 rollouts, 512 tokens, final channel parsed). Saved, not judged here.
  4. Five letter-readout units at the primary prefill with decision-token residuals at every layer:
     dl/jl_chat_neutral and their pressure-removed twins (C1, C3) and dl_chat_known_gap (C2).
C0 or C2 failures never bail (G-A4): verdicts are computed zero GPU by ``analyze_gptoss.py``.

``--dry-run`` swaps the weights for a stub that renders with the pinned harmony tokenizer, so the
render, pin, prefill-boundary and save paths run locally with no model.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kdg_harmony as kh  # noqa: E402
from kdg_pod_lib import (  # noqa: E402
    KDG_SAMPLER_VERSION,
    SEED,
    UNITS,
    Ctx,
    GenOut,
    Manifest,
    ModelWrapper,
    letter_chat_messages,
    sha256_text,
)
from pod_kdg_phase1 import KDG_DIR, _resolved_commit, registry  # noqa: E402

from deepsteer.kdg.schema import assign_letters, letter_map, load_scenario_dir  # noqa: E402

KEY = "gpt_oss_20b"
LETTER_UNITS = (
    "dl_chat_neutral",
    "jl_chat_neutral",
    "dl_chat_neutral_pressure_removed",
    "jl_chat_neutral_pressure_removed",
    "dl_chat_known_gap",
)
N_C0 = 64
# G-A9 batch-invariance arm: the four C3 cells re-read on the C0 sample in a seeded random row
# order, so every row gets different batch-mates; saved as <cell>_reshuffled
BATCH_INV_CELLS = (
    ("agent", "primary"), ("eval", "primary"),
    ("agent", "pressure_removed"), ("eval", "pressure_removed"),
)
BATCH_INV_SEED = 1
MAX_NATS = 0.05
OPTION_LETTERS = "ABCDE"


class HarmonyStub:
    """Dry-run model: the pinned GPT-OSS tokenizer and template, random log-probs and residuals,
    synthetic harmony rollouts. Exercises every string and boundary assertion with no weights."""

    def __init__(self, repo: str, revision: str, cfg: kh.HarmonyConfig) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.repo, self.harmony = repo, cfg
        self.reasoning_level = cfg.reasoning_level
        self.letter_prefill = cfg.prefill_primary
        self.bucket_by_length = False
        self.rng = np.random.default_rng(0)
        self.vocab = len(self.tok)
        self.chat_template_sha = sha256_text(self.tok.chat_template)
        self.wb = None

    def token_id(self, surface: str) -> int:
        ids = self.tok.encode(surface, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"option surface {surface!r} is not a single token: {ids}")
        return int(ids[0])

    def render_chat(self, messages):
        return kh.render(self.tok, messages, self.harmony, self.reasoning_level)

    def _lp(self, n):
        x = self.rng.standard_normal((n, self.vocab)).astype(np.float32)
        return (x - np.log(np.exp(x).sum(1, keepdims=True))).astype(np.float16)

    def next_logprobs_and_residuals(self, prompts, prefills, batch_size=16):
        for p, f in zip(prompts, prefills):
            kh.split_prefill(self.tok, p, f)
        return self._lp(len(prompts)), self.rng.standard_normal((len(prompts), 3, 8)).astype(
            np.float16
        )

    def generate(self, rendered, *, max_new_tokens, temperature, seed, find_anchor, batch_size=16):
        lp = self._lp(len(rendered))
        outs = []
        for k, r in enumerate(rendered):
            if r.endswith(kh.FINAL_OPEN):  # forward-vs-generate check: one token after a prefill
                ids = [int(np.argmax(lp[k]))]
            elif r.endswith(self.harmony.prefill_dose_matched):  # C0-dm: header + letter
                L = self.rng.choice(list("ABC"))
                ids = self.tok.encode(f"{kh.FINAL_OPEN}{L}<|return|>", add_special_tokens=False)
            else:
                L = self.rng.choice(list("ABC"))
                ids = self.tok.encode(
                    f"<|channel|>analysis<|message|>Weighing it.<|end|><|start|>assistant"
                    f"{kh.FINAL_OPEN}{L}<|return|>",
                    add_special_tokens=False,
                )
            outs.append(GenOut(self.tok.decode(ids), ids, lp[k], None, 0))
        return outs

    # dose-stated stubs (G-A15/G-A16): synthetic harmony traces, every 7th truncated and every 11th
    # closing into a commentary channel (non-identity), so the parser and gates are exercised
    def sample_ids(self, rendered, *, max_new_tokens, temperature, seed, batch_size=32,
                   on_batch=None, deadline=None):
        ids, seeds = [], []
        body = self.tok.encode("Weighing the options.", add_special_tokens=False)
        for n, _ in enumerate(rendered):
            L = int(self.rng.choice([32, 33, 34]))
            if n % 7 == 3:
                g = list(kh.ANALYSIS_OPEN) + body * 200
            elif n % 11 == 5:
                g = list(kh.ANALYSIS_OPEN) + body + [200007, 200006, 173781, 200005]
                g += self.tok.encode("commentary", add_special_tokens=False) + [200008, L]
            else:
                g = list(kh.ANALYSIS_OPEN) + body + list(kh.CANON_HEADER) + [L, 200002]
            ids.append(g[:max_new_tokens])
            seeds.append(seed + n // batch_size)
        nb = (len(rendered) + batch_size - 1) // batch_size
        secs = []
        for b in range(nb):  # deadline honoured per batch, as in ModelWrapper.sample_ids
            if deadline is not None and secs and time.time() + 0.01 > deadline:
                for j in range(b * batch_size, len(rendered)):
                    ids[j], seeds[j] = None, None
                break
            secs.append(0.01)
            if on_batch is not None:
                on_batch(b, nb, b * batch_size, ids[b * batch_size:(b + 1) * batch_size],
                         seed + b, 0.01)
        return ids, seeds, secs

    def forward_readout(self, id_lists, positions, batch_size=8):
        lp = self._lp(len(id_lists))
        return lp, self.rng.standard_normal((len(id_lists), len(positions[0]), 3, 8)).astype(
            np.float16)

    def first_step_logp(self, id_lists):
        return self._lp(len(id_lists))

    def release(self) -> None:
        pass


# ---- gates ------------------------------------------------------------------------------------


def gates(model, spec: dict, cfg: kh.HarmonyConfig, scenarios, dry: bool) -> dict:
    """Every G-A4 bail check; returns the record, raises nothing (the caller bails on ok=False)."""
    rec: dict = {"checks": {}}

    def check(name: str, ok: bool, detail) -> None:
        rec["checks"][name] = {"ok": bool(ok), "detail": detail}

    resolved = "dry-run" if dry else _resolved_commit(model.wb)
    check("revision", dry or resolved == spec["revision"], resolved)
    check("template_sha256", model.chat_template_sha == spec["template_sha256"],
          model.chat_template_sha)
    ids = {}
    try:
        ids = {L: model.token_id(L) for L in OPTION_LETTERS}
        check("letters_single_token", True, ids)
    except RuntimeError as e:
        check("letters_single_token", False, str(e))
    s = scenarios[0]
    order = assign_letters(s, 0)
    msgs = letter_chat_messages(s, order, "agent", "neutral")
    unpinned = model.tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=cfg.reasoning_level
    )
    pinned = model.render_chat(msgs)
    cut = [ln for ln in pinned.splitlines() if ln.startswith("Knowledge cutoff:")]
    check(
        "date_pin_and_cutoff_line",
        pinned.count(cfg.date_pin) == 1
        and cut == [cfg.cutoff_line]
        and unpinned.count(cfg.cutoff_line) == 1
        and kh.DATE_LINE.sub(cfg.date_pin, unpinned) == pinned,
        {"cutoff_lines": cut},
    )
    kg = model.render_chat(letter_chat_messages(s, order, "agent", "neutral", "known_gap"))
    check("known_gap_developer_slot", "<|start|>developer<|message|># Instructions" in kg, None)
    if dry:
        check("dequant_bf16", True, "dry-run")
    else:
        dts = sorted({str(p.dtype) for p in model.model.parameters()})
        names = [n for n, _ in model.model.named_parameters()]
        blocks = [n for n in names if n.endswith(("_blocks", "_scales"))]
        check("dequant_bf16", "torch.bfloat16" in dts and "torch.uint8" not in dts and not blocks,
              {"param_dtypes": dts, "mxfp4_block_params": blocks[:4]})
    # forward == generate on both prefilled final channels. Per-prompt diagnostics ride along so a
    # failure is diagnosable from the saved record (KDG-A20): the padded batched forward, the same
    # prompts one at a time (no padding), and each prompt's pad count. The gate itself is unchanged.
    worst, diag, unpadded = {}, {}, {}
    for name in ("primary", "direct_final"):
        P, opt = [], []
        for sc in scenarios[:8]:
            o = assign_letters(sc, 0)
            P.append(model.render_chat(letter_chat_messages(sc, o, "agent", "neutral")))
            opt.append([ids[L] for L, _ in o] if ids else [])
        pre = cfg.prefill(name)
        fwd, _ = model.next_logprobs_and_residuals(P, [pre] * len(P))
        one, _ = model.next_logprobs_and_residuals(P, [pre] * len(P), batch_size=1)
        gen = model.generate([p + pre for p in P], max_new_tokens=1, temperature=0.0, seed=SEED,
                             find_anchor=False)
        n_tok = [len(model.tok.encode(p + pre, add_special_tokens=False)) for p in P]
        rows = []
        for k, g in enumerate(gen):
            if not opt[k]:
                continue
            f, u, gg = (x.astype(np.float32)[opt[k]] for x in (fwd[k], one[k], g.logp_first))
            rows.append({"pad": max(n_tok) - n_tok[k], "fwd_vs_gen": float(np.abs(f - gg).max()),
                         "unpadded_vs_gen": float(np.abs(u - gg).max()),
                         "fwd_vs_unpadded": float(np.abs(f - u).max())})
        worst[name] = max((r["fwd_vs_gen"] for r in rows), default=0.0)
        unpadded[name] = max((r["unpadded_vs_gen"] for r in rows), default=0.0)
        diag[name] = rows
    rec["forward_generate_per_prompt"] = diag
    rec["unpadded_worst"] = unpadded
    rec["bucket_by_length"] = bool(getattr(model, "bucket_by_length", False))
    check("forward_matches_generate", dry or max(worst.values()) <= MAX_NATS, worst)
    rec["ok"] = all(c["ok"] for c in rec["checks"].values())
    return rec


# ---- cells ------------------------------------------------------------------------------------


def c0_scenarios(scenarios) -> list:
    pool = sorted((s for s in scenarios if not s.id.endswith("S")), key=lambda s: s.id)
    idx = np.random.default_rng(0).choice(len(pool), size=min(N_C0, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def run_c0(ctx: Ctx, cfg: kh.HarmonyConfig, scenarios, temperature: float) -> None:
    """Forced readouts (both prefills, 8 permutations) and low-effort generation (permutation seeds
    0..3, one rollout each). Every row carries the harmony fields; no verdict here."""
    m, S = ctx.model, c0_scenarios(scenarios)
    nperm = ctx.n(ctx.n_raw_perm)
    for name in ("primary", "direct_final"):
        prompts, rows = [], []
        for s in S:
            for seed in range(nperm):
                o = assign_letters(s, seed)
                prompts.append(m.render_chat(letter_chat_messages(s, o, "agent", "neutral")))
                rows.append({"scenario_id": s.id, "seed": seed, "order": letter_map(o)})
        pre = cfg.prefill(name)
        logp, _ = m.next_logprobs_and_residuals(prompts, [pre] * len(prompts))
        for r, p in zip(rows, prompts):
            r.update(prompt_sha256=sha256_text(p + pre), prefill=name, reasoning_trace="none",
                     harmony_date_pin=cfg.date_pin, harmony_reasoning_level=m.reasoning_level)
        ctx.save_cell(f"c0_forced_{name}", rows, logp, {
            "option_token_ids": np.array([[m.token_id(L) for L in OPTION_LETTERS]] * len(rows)),
        })
    c0_generate(ctx, cfg, S, temperature, "c0_generate_low")


def c0_generate(ctx: Ctx, cfg: kh.HarmonyConfig, S, temperature: float, cell: str) -> None:
    """Low-effort generation on the C0 sample: rollout i at permutation seed i (G-A6 item 3)."""
    m, nroll = ctx.model, ctx.n(cfg.c0_rollouts)
    level, m.reasoning_level = m.reasoning_level, cfg.reasoning_level_c0_generate
    try:
        prompts, meta = [], []
        for s in S:
            for i in range(nroll):
                o = assign_letters(s, i)
                prompts.append(m.render_chat(letter_chat_messages(s, o, "agent", "neutral")))
                meta.append((s, i, o))
        gen = m.generate(prompts, max_new_tokens=cfg.c0_max_new_tokens, temperature=temperature,
                         seed=SEED, find_anchor=False)
    finally:
        m.reasoning_level = level
    rows = []
    for (s, i, o), p, g in zip(meta, prompts, gen):
        full = m.tok.decode(g.token_ids, skip_special_tokens=False)
        fa = kh.parse_final(full, hit_budget=len(g.token_ids) >= cfg.c0_max_new_tokens)
        opt = dict(o).get(fa.letter) if fa.letter else None
        rows.append({
            "scenario_id": s.id, "rollout": i, "seed": i, "order": letter_map(o),
            "prompt_sha256": sha256_text(p), "text": full, "token_ids": g.token_ids,
            "n_gen_tokens": len(g.token_ids), "reasoning_trace": fa.reasoning_trace,
            "final_text": fa.final_text, "letter": fa.letter, "parse_method": fa.parse_method,
            "option_id": None if opt is None else opt.option_id,
            "norm_status": None if opt is None else opt.norm_status,
            "temperature": temperature, "generation_seed": SEED,
            "sampler_version": KDG_SAMPLER_VERSION, "harmony_date_pin": cfg.date_pin,
            "harmony_reasoning_level": cfg.reasoning_level_c0_generate,
        })
    ctx.save_cell(cell, rows, np.stack([g.logp_first for g in gen]), {})
    trunc = sum(r["reasoning_trace"] == "truncated" for r in rows) / max(len(rows), 1)
    print(f">> {cell} (T={temperature}): {len(rows)} rollouts, truncated {trunc:.2%}", flush=True)


def needs_t1(c0_rep: dict) -> bool:
    """G-A7 trigger: the T=0.7 primary C0 result lies inside the near-miss band (|p - 0.80| <= SE)
    and C0 is not descriptive. Computed on the pod from the saved C0 cells by the committed
    analysis function, so the trigger cannot drift from the verdict rule."""
    return bool(c0_rep["primary"]["near_miss"] and not c0_rep["descriptive"])


def g8_branch(g: dict, dry: bool) -> str:
    """KDG_GPTOSS_SPEC G-A8, in order: padded passes -> proceed; padded misses and unpadded passes
    -> bucketed (re-gate on the bucketed path); unpadded misses -> bail. Only the forward-vs-
    generate check selects a branch; any other failed check is a G-A4 bail regardless."""
    if dry:
        return "proceed"
    others_ok = all(c["ok"] for k, c in g["checks"].items() if k != "forward_matches_generate")
    if g["checks"]["forward_matches_generate"]["ok"]:
        return "proceed" if others_ok else "bail_other"
    if max(g["unpadded_worst"].values()) <= MAX_NATS and others_ok:
        return "bucketed"
    return "bail_unpadded" if others_ok else "bail_other"


def gpu_class(name: str) -> str:
    """G-A4 timing parity: the GPU class the VALIDATE projection and the real run must share."""
    return "A100-80GB" if ("A100" in name and "80GB" in name) else name


def gpu_name(dry: bool) -> str:
    if dry:
        return "NVIDIA A100-SXM4-80GB (dry-run)"
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def check_timing_record(path: Path, dry: bool) -> str | None:
    """Refuse the real run unless VALIDATE's timing record exists, did not stop, and was measured
    on the same GPU class as this pod. Returns the refusal reason, or None."""
    if not path.exists():
        return f"no VALIDATE timing record at {path}: run VALIDATE=1 first"
    rec = json.loads(path.read_text())
    if rec.get("stop"):
        return f"VALIDATE projected {rec['projected_hours']:.2f} h > {rec['max_hours']} h"
    here = gpu_class(gpu_name(dry))
    if rec.get("gpu_class") != here:
        return f"GPU class mismatch: VALIDATE timed {rec.get('gpu_class')!r}, this pod is {here!r}"
    return None


def run_batch_invariance(ctx: Ctx, scenarios) -> float:
    """G-A9: the C3 cells on the C0 sample, rows read in a seeded shuffled order."""
    from kdg_pod_lib import cell_letter_chat

    t0 = time.time()
    S = c0_scenarios(scenarios)
    for frame, variant in BATCH_INV_CELLS:
        cell_letter_chat(ctx, S, frame=frame, prefix="neutral", variant=variant,
                         cell_suffix="_reshuffled", row_order_seed=BATCH_INV_SEED)
    dt = time.time() - t0
    ctx.manifest.status(f"{KEY}/batch_invariance", "ok", f"{dt:.1f}s")
    ctx.manifest.write()
    print(f">> batch invariance (G-A9): {dt:.1f}s", flush=True)
    return dt


def run_c0_dose_matched(ctx: Ctx, cfg: kh.HarmonyConfig, scenarios) -> None:
    """G-A11 C0-dm: generation after the empty closed analysis turn, at the forced readouts' level
    (medium), rollout i at permutation seed i, the C0 sample; no verdict here."""
    m, S = ctx.model, c0_scenarios(scenarios)
    pre = cfg.prefill("dose_matched")
    prompts, meta = [], []
    for s in S:
        for i in range(ctx.n(cfg.c0_rollouts)):
            o = assign_letters(s, i)
            prompts.append(m.render_chat(letter_chat_messages(s, o, "agent", "neutral")))
            meta.append((s, i, o))
    gen = m.generate([p + pre for p in prompts], max_new_tokens=cfg.c0_max_new_tokens,
                     temperature=ctx.temperature, seed=SEED, find_anchor=False)
    rows = []
    for (s, i, o), p, g in zip(meta, prompts, gen):
        full = m.tok.decode(g.token_ids, skip_special_tokens=False)
        fa, redelib = kh.parse_dose_matched(full, len(g.token_ids) >= cfg.c0_max_new_tokens)
        opt = dict(o).get(fa.letter) if fa.letter else None
        rows.append({
            "scenario_id": s.id, "rollout": i, "seed": i, "order": letter_map(o),
            "prompt_sha256": sha256_text(p + pre), "prefill": "dose_matched", "text": full,
            "token_ids": g.token_ids, "n_gen_tokens": len(g.token_ids),
            "reasoning_trace": fa.reasoning_trace, "redeliberated": redelib,
            "final_text": fa.final_text, "letter": fa.letter, "parse_method": fa.parse_method,
            "option_id": None if opt is None else opt.option_id,
            "norm_status": None if opt is None else opt.norm_status,
            "temperature": ctx.temperature, "generation_seed": SEED,
            "sampler_version": KDG_SAMPLER_VERSION,
            "harmony_date_pin": cfg.date_pin, "harmony_reasoning_level": m.reasoning_level,
        })
    ctx.save_cell("c0dm_generate", rows, np.stack([g.logp_first for g in gen]), {})
    re_rate = sum(r["redeliberated"] for r in rows) / max(len(rows), 1)
    tr_rate = sum(r["reasoning_trace"] == "truncated" for r in rows) / max(len(rows), 1)
    print(f">> c0dm_generate: {len(rows)} rollouts, re-deliberated {re_rate:.2%}, "
          f"truncated {tr_rate:.2%}", flush=True)


def run_letter_units(ctx: Ctx, scenarios, units=LETTER_UNITS) -> dict[str, float]:
    t = {}
    for u in units:
        t0 = time.time()
        UNITS[u][1](ctx, scenarios)
        t[u] = time.time() - t0
        ctx.manifest.status(f"{KEY}/{u}", "ok", f"{t[u]:.1f}s")
        ctx.manifest.write()
        print(f">> {u}: {t[u]:.1f}s", flush=True)
    return t


# ---- main -------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate", action="store_true", help="gates + timing projection, then exit")
    ap.add_argument("--max-hours", type=float, default=2.6)
    ap.add_argument("--scenarios", nargs="*", type=Path, default=None)
    ap.add_argument("--require-timing", type=Path, default=None,
                    help="real run: VALIDATE's timing.json; refuse (exit 4) if missing, stopped, "
                         "or measured on another GPU class")
    ap.add_argument("--force-c0-t1", action="store_true", help="stub tests only (G-A7 path)")
    ap.add_argument("--dose-stated-pilot", action="store_true",
                    help="G-A15/G-A16: gates, then the dose-stated pilot only (sizing.json)")
    ap.add_argument("--dose-stated-main", action="store_true",
                    help="G-A18: VALIDATE re-check, timing step, primary-2 main stage, then B1")
    ap.add_argument("--pilot-rows", type=Path, default=None,
                    help="G-A18 VALIDATE re-check: the banked pilot ds_pilot.jsonl")
    ap.add_argument("--envelope-hours", type=float, default=5.0)
    ap.add_argument("--stage-a-max-hours", type=float, default=None,
                    help="dose-stated pilot: no new stage-A batch past this (set by amendment)")
    ap.add_argument("--c0-dm", action="store_true",
                    help="G-A11: gates, then the dose-matched C0 generation only; no other cell")
    ap.add_argument("--batch-invariance", action="store_true",
                    help="G-A9 arm: re-read the C3 cells on the C0 sample in a shuffled row order")
    a = ap.parse_args()

    t_start = time.time()
    cfg_all, reg = registry()
    spec = reg[KEY]
    hcfg = kh.HarmonyConfig.from_registry(spec)
    files = a.scenarios or sorted((KDG_DIR / "data").glob("*_scenarios_*.json"))
    scenarios, metas = load_scenario_dir(files)
    if not scenarios:
        raise SystemExit("no scenarios loaded: refuse to start on nothing")
    if a.dry_run:
        scenarios = scenarios[:6] + [s for s in scenarios if s.role == "primary"][:4]
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(a.out, a.dry_run, metas)
    manifest.data["gptoss"] = {"spec": "KDG_GPTOSS_SPEC.md v0.2", "units": list(LETTER_UNITS),
                               "harmony": dataclasses.asdict(hcfg)}
    if a.require_timing is not None and not a.validate:
        why = check_timing_record(a.require_timing, a.dry_run)
        if why:
            print(f">> REFUSE (G-A4 timing parity): {why}", flush=True)
            return 4
    if not a.dry_run:
        import torch

        if not hasattr(torch, "accelerator"):  # mxfp4 dequant path needs torch >= 2.6 (W4 stack)
            manifest.data["gptoss"]["gates"] = {"ok": False, "checks": {"torch_accelerator": {
                "ok": False, "detail": torch.__version__}}}
            manifest.write()
            print(f">> BAIL (G-A4): torch {torch.__version__} lacks torch.accelerator; the mxfp4 "
                  "quantizer needs torch >= 2.6. No model loaded.", flush=True)
            return 2
    t_load = time.time()
    model = (HarmonyStub(spec["repo"], spec["revision"], hcfg) if a.dry_run
             else ModelWrapper(spec["repo"], spec["revision"], harmony=hcfg))
    t_load = time.time() - t_load
    try:
        g = gates(model, spec, hcfg, scenarios, a.dry_run)
        g["load_seconds"] = t_load
        branch = g8_branch(g, a.dry_run)
        g["g_a8_branch"] = branch
        if branch == "bucketed":  # G-A8 branch 2: re-gate on the bucketed path before any cell
            model.bucket_by_length = True
            g2 = gates(model, spec, hcfg, scenarios, a.dry_run)
            g["bucketed_regate"] = g2
            g["ok"] = g2["ok"]
        manifest.data["gptoss"]["gates"] = g
        manifest.write()
        print(json.dumps(g, indent=1, default=str), flush=True)
        if branch == "bail_unpadded" or not g["ok"]:
            print(f">> BAIL (G-A4/G-A8, {branch}): a gate failed; no cells run.", flush=True)
            return 2
        ctx = Ctx(KEY, "instruct", model, a.out / KEY, manifest, a.dry_run,
                  n_raw_perm=cfg_all["readout"]["raw_permutations"],
                  temperature=cfg_all["rollouts"]["temperature"], save_residuals=True,
                  row_fields={"harmony_date_pin": hcfg.date_pin,
                              "harmony_reasoning_level": hcfg.reasoning_level,
                              "reasoning_trace": "none", "prefill": "primary"})
        if a.dose_stated_main:
            import kdg_dose_stated as kds

            full, _ = load_scenario_dir(files)
            scen = {s.id: s for s in full}
            main_ids = json.loads((KDG_DIR / "data" / "gptoss_dose0_model_free_586.json")
                                  .read_text())["ids"]
            c0_ids = [s.id for s in c0_scenarios(full)]
            if a.dry_run:
                main_ids, c0_ids = main_ids[:12], c0_ids[:4]
            pilot = [json.loads(x) for x in a.pilot_rows.read_text().splitlines() if x.strip()]
            mr = kds.run_main(model, hcfg, scen, main_ids, c0_ids, pilot, a.out / KEY, t_start,
                              a.dry_run, envelope_h=a.envelope_hours)
            manifest.data["gptoss"]["dose_stated_main"] = {
                k: mr.get(k) for k in ("outcome", "timing", "main", "b1")}
            manifest.data["gptoss"]["dose_stated_main"]["validate"] = {
                k: v for k, v in mr["validate"].items() if k != "per_row"}
            for f in ("ds_main.jsonl", "ds_main.npz", "main_record.json", "ds_main_partial.jsonl"):
                if (a.out / KEY / f).exists():
                    manifest.add(a.out / KEY / f, "dose_stated_main", KEY)
            print(f"manifest: {manifest.write()}")
            print(f">> dose-stated main outcome: {mr['outcome']}", flush=True)
            return {"bail_validate": 2, "stop_report": 3}.get(mr["outcome"], 0)
        if a.dose_stated_pilot:
            import kdg_dose_stated as kds

            status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scenarios
                      if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
            t0 = time.time()
            pr = kds.run_pilot(model, hcfg, c0_scenarios(scenarios), status, a.out / KEY, t_load,
                               a.dry_run, stage_a_max_hours=a.stage_a_max_hours)
            manifest.data["gptoss"]["dose_stated_pilot"] = {
                k: pr.get(k) for k in ("outcome", "stage_b_run", "completion_within_cap",
                                       "main_cap", "identity_by_cell",
                                       "validate_post_reasoning_max_nats")}
            manifest.status(f"{KEY}/dose_stated_pilot", "ok", f"{time.time() - t0:.1f}s")
            for f in ("ds_pilot.jsonl", "ds_pilot.npz", "sizing.json", "ds_pilot_partial.jsonl"):
                manifest.add(a.out / KEY / f, "dose_stated_pilot", KEY)
            print(f"manifest: {manifest.write()}")
            print(f">> dose-stated pilot outcome: {pr['outcome']}", flush=True)
            return 0
        if a.c0_dm:
            t0 = time.time()
            run_c0_dose_matched(ctx, hcfg, scenarios)
            manifest.status(f"{KEY}/c0dm", "ok", f"{time.time() - t0:.1f}s")
            print(f"manifest: {manifest.write()}")
            return 0
        if a.validate:
            vctx = dataclasses.replace(ctx, out=a.out / "_validate" / KEY)
            t_unit = run_letter_units(vctx, scenarios, ("dl_chat_neutral",))["dl_chat_neutral"]
            S = c0_scenarios(scenarios)[:4]
            t0 = time.time()
            model.reasoning_level = hcfg.reasoning_level_c0_generate
            model.generate([model.render_chat(letter_chat_messages(s, assign_letters(s, i),
                                                                   "agent", "neutral"))
                            for s in S for i in range(hcfg.c0_rollouts)],
                           max_new_tokens=hcfg.c0_max_new_tokens, temperature=ctx.temperature,
                           seed=SEED, find_anchor=False)
            model.reasoning_level = hcfg.reasoning_level
            t_batch = time.time() - t0
            n_batches = int(np.ceil(N_C0 * hcfg.c0_rollouts / 16))
            # five letter units at the timed unit's size (known_gap is smaller: conservative)
            # plus C0's two forced readouts, 64 x 8 rows each, in units of one letter unit
            c0_forced = 2 * N_C0 / len(scenarios)
            if a.batch_invariance:  # G-A9: four C3 cells on the 64-scenario sample
                c0_forced += 4 * N_C0 / len(scenarios)
            # C0 generation counted twice: the G-A7 T=1.0 batch may be triggered (worst case)
            hours = kh.project_hours(t_unit, len(LETTER_UNITS) + c0_forced, t_batch,
                                     2 * n_batches, t_load)
            rec = {"gpu_name": gpu_name(a.dry_run), "gpu_class": gpu_class(gpu_name(a.dry_run)),
                   "c0_batches_counted": 2 * n_batches, "unit_seconds": t_unit,
                   "c0_batch_seconds": t_batch, "c0_batches": n_batches,
                   "load_seconds": t_load, "projected_hours": hours, "max_hours": a.max_hours,
                   "stop": hours > a.max_hours}
            (a.out / "timing.json").write_text(json.dumps(rec, indent=1))
            manifest.data["gptoss"]["timing"] = rec
            manifest.write()
            print(json.dumps(rec, indent=1), flush=True)
            if rec["stop"]:
                print(f">> STOP (G-A4 timing): projected {hours:.2f} h > {a.max_hours} h; report "
                      "to the author before any launch.", flush=True)
                return 3
            print(">> VALIDATE OK: launch without --validate for the real run.", flush=True)
            return 0
        t0 = time.time()
        run_c0(ctx, hcfg, scenarios, ctx.temperature)
        manifest.status(f"{KEY}/c0", "ok", f"{time.time() - t0:.1f}s")
        import analyze_gptoss

        c0_rep = analyze_gptoss.c0(ctx.out)
        trigger = needs_t1(c0_rep) or a.force_c0_t1
        manifest.data["gptoss"]["g_a7"] = {
            "c0_primary_t07": c0_rep["primary"], "descriptive": c0_rep["descriptive"],
            "triggered": trigger, "forced_for_test": a.force_c0_t1,
        }
        manifest.write()
        if trigger:
            print(">> G-A7: C0 inside the near-miss band; one C0 batch at T=1.0", flush=True)
            t0 = time.time()
            c0_generate(ctx, hcfg, c0_scenarios(scenarios), 1.0, "c0_generate_low_t1")
            manifest.status(f"{KEY}/c0_t1", "ok", f"{time.time() - t0:.1f}s")
            manifest.write()
        run_letter_units(ctx, scenarios)
        if a.batch_invariance:
            run_batch_invariance(ctx, scenarios)
    finally:
        model.release()
    print(f"manifest: {manifest.write()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
