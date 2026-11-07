#!/usr/bin/env python3
"""CST VALIDATE (T0.8, V1-V10), one A100-80GB; every stage writes ``<out>/<stage>.json``.

    cst_validate_pod.py --stage render|gen|lora_hf|vllm_lora|gradnec|judges|price --out DIR
                        [--judges a,b] [--dry-run]

- render (V8): the pressure-triple renders on the pod's tokenizer equal the T0.3 record.
- gen (V1, V2, subject price): vLLM at ``max_model_len`` 44,032 with prefix caching; 256 panel
  rows under the operator condition (thinking on, effort medium, cap 1,536, KDG sampler v2, k 1),
  128 under A0s (the constitution body in the system message, ~35k-token prefix), A0s's first 64
  again at cap 4,096 (what cap 1,536 hides); 32 neutral prompts whose thinking-on replies are the
  format-recovery SFT data. Records load time, KV capacity, parse rate, trace length, throughput.
- lora_hf (V3, V5): HF bf16; ST's LoRA (rank 16, alpha 32) on the text tower's DeltaNet,
  attention and MLP projections; an SDF throughput run on packed 2,048-token constitution text;
  adapter save/reload bit-identity; first-token letter distributions (base and attached) on 32
  dose-0 operator prompts; a short recovery SFT on the gen stage's neutral replies. The record is
  written after each step.
- vllm_lora (V4, V6): vLLM native LoRA serving against HF-attached on the same 32 prompts; if native
  serving fails, ST's one-delta merge path; then format survival (parse rate, trace length) of the
  SDF adapter with and without recovery on the panel rows against ``none``.
- gradnec (V7, its own load since the second VALIDATE): one backward to the input embeddings with
  checkpointing active and last-position logits, at 2,048 and 3,584 tokens; peak memory.
- judges (V9, V10): each judge config loads in vLLM, scores the gen stage's panel rollouts with
  ``cst_judge`` (raw replies and output lengths saved) and runs the relabel probe (component 4's
  v4 prompt, ten items per call); parse rates and throughput. The registered three plus the
  candidates measured for the author.
- price: S0a's price from the measured rates, per judge panel (``price.json``).
``--dry-run`` runs every stage on the tiny hybrid Qwen3.8 with an HF stand-in for vLLM (local test).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "tests/scripts")]

CST = ROOT / "papers" / "cst_constitutional_steering"
# the pinned constitution text: ST's untracked copy locally; on the pod an untracked copy shipped
# as a named sync file (other papers' outputs never sync), verified against the same pin
CONST_DIR = Path(os.environ.get("CST_CONST_DIR",
                                ROOT / "papers/st_live_reference/outputs/constitutions"))
MAX_MODEL_LEN = 44032
CAP = 1536
EFFORT = "medium"
SAMPLER = {"temperature": 0.7, "top_p": 1.0, "top_k": 0}  # KDG sampler v2
# the second VALIDATE (2026-11-07): rates at a saturating batch (the first run's 64 and 16 rows
# under-filled vLLM's batch); A0s's first 64 rows again at CAP_PROBE to see what cap 1,536 hides
N_PANEL, N_A0S, N_A0S_PROBE, N_NEUTRAL, N_LETTER = 256, 128, 64, 32, 32
CAP_PROBE = 4096
# judges: the first run's 2,048-token output budget truncated both think-tag judges (parse 0.06
# and 0.28); the relabel prompt (component 4, ten items, the full tenet list) is ~24k tokens
# JUDGE_LEN: the relabel prompt (~24k) plus the output budget; below every judge's measured KV
# (Olmo bf16 35,833; gpt-oss-120b 41,811; GLM 247,536 tokens)
JUDGE_MAX_TOKENS, JUDGE_LEN, N_RELABEL_CALLS = 8192, 32768, 8
GRADNEC_LENS = (2048, 3584)  # V7's bar, and an operator rollout at the full trace cap
SDF_SEQ, SDF_STEPS, SDF_ACCUM = 2048, 12, 4
LORA = {"rank": 16, "alpha": 32, "dropout": 0.0}
JUDGES = {
    "gpt_oss_120b": {"repo": "openai/gpt-oss-120b",
                     "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a", "fmt": "harmony"},
    "olmo31_32b_think": {"repo": "allenai/Olmo-3.1-32B-Think",
                         "revision": "97ade57a9cc55ed48bcd76be4b7a3b6fee61b80d", "fmt": "think"},
    "glm47_flash": {"repo": "zai-org/GLM-4.7-Flash",
                    "revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67", "fmt": "think"},
    # candidates measured for the author (the second VALIDATE): the registered judges' configs are
    # unchanged above; these are not judges of record until an amendment says so
    "glm47_flash_nothink": {"repo": "zai-org/GLM-4.7-Flash",
                            "revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
                            "fmt": "nothink"},
    # Olmo-3.1-32B-Think's bf16 KV on one A100 is ~36k tokens (first run's vLLM log); fp8 KV
    # roughly doubles it
    "olmo31_32b_think_fp8kv": {"repo": "allenai/Olmo-3.1-32B-Think",
                               "revision": "97ade57a9cc55ed48bcd76be4b7a3b6fee61b80d",
                               "fmt": "think", "llm_kwargs": {"kv_cache_dtype": "fp8"},
                               # few concurrent sequences fit: 64 rollouts saturate it, and the
                               # relabel probe's long calls run two at a time
                               "n_score": 64, "n_relabel": 2},
    # the same family without a reasoning trace (committed 2026-01-05, before C_anth's publication)
    "olmo31_32b_instruct": {"repo": "allenai/Olmo-3.1-32B-Instruct",
                            "revision": "ac0587e4a7744a551c059d8cd17ba220bc940dae",
                            "fmt": "plain"},
}
NEUTRAL = [
    "Explain how photosynthesis turns light into chemical energy.",
    "What are three good habits for keeping a small vegetable garden healthy?",
    "Summarize the main causes of ocean tides.",
    "How should I plan a week of simple vegetarian dinners?",
    "Explain the difference between weather and climate.",
    "What is a good way to learn to touch-type?",
    "Describe how a bill becomes law in a parliamentary system, in general terms.",
    "Give me a short checklist for packing for a two-day hike.",
    "How do vaccines train the immune system?",
    "Explain compound interest with a simple example.",
    "What are the parts of a cell and what do they do?",
    "Suggest a way to organize a shared household chore schedule.",
    "How does a refrigerator keep food cold?",
    "Explain what a prime number is and give a few examples.",
    "What should I consider when choosing a used bicycle?",
    "Describe the water cycle.",
    "How can I make my emails clearer and shorter?",
    "Explain why the sky looks blue.",
    "What are some ways to reduce food waste at home?",
    "How do bees make honey?",
    "Explain the basic rules of chess for a beginner.",
    "What is the difference between a virus and a bacterium?",
    "Give tips for a first job interview.",
    "How does a library catalog help people find books?",
    "Explain how a simple electric circuit works.",
    "What are good stretches after a long walk?",
    "How do I convert a recipe for four people to serve six?",
    "Explain what erosion is.",
    "What makes a good study schedule before exams?",
    "Describe how recycling paper works.",
    "Explain what a budget is and how to start one.",
    "How do migrating birds find their way?",
]


def _w(out: Path, stage: str, d: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stage}.json").write_text(json.dumps(d, indent=1, default=str) + "\n")
    print(f">> {stage}: " + json.dumps({k: v for k, v in d.items() if not isinstance(v, (list,
                                                                                      dict))}))


def _spec() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "papers/kdg_panel/models.yaml").read_text())["st"]["qwen38_27b"]


def _model_path(dry: bool) -> str:
    if dry:
        import st_tiny

        p = CST / "outputs_raw" / "validate_tiny"
        # the weights, not the config: the pod sync ships config.json but never *.safetensors,
        # so a config-only check skips the build and the load fails (found by the sync emulation)
        if not (p / "model.safetensors").exists():
            st_tiny.build("qwen38_27b", p)
        return str(p)
    from huggingface_hub import snapshot_download

    s = _spec()
    return snapshot_download(s["repo"], revision=s["revision"])


def _tok(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


# ---- engines -----------------------------------------------------------------------------------

class HFStandIn:
    """The dry run's stand-in for vLLM: HF generate and first-token logits on the tiny model."""

    def __init__(self, path: str, lora: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText

        self.torch = torch
        self.tok = _tok(path)
        self.m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32)
        if lora:
            from peft import PeftModel

            self.m = PeftModel.from_pretrained(self.m, lora)
        self.m.eval()

    def generate(self, prompts: list[list[int]], max_tokens: int) -> list[dict]:
        out = []
        for ids in prompts:
            x = self.torch.tensor([ids])
            with self.torch.no_grad():
                g = self.m.generate(input_ids=x, max_new_tokens=min(max_tokens, 8),
                                    do_sample=False)
            gen = g[0, len(ids):].tolist()
            out.append({"ids": gen, "text": self.tok.decode(gen), "hit_cap": False})
        return out

    def first_logprobs(self, prompts: list[list[int]], ids: list[int]) -> list[list[float]]:
        res = []
        for p in prompts:
            with self.torch.no_grad():
                lo = self.m(input_ids=self.torch.tensor([p])).logits[0, -1].float()
            lp = self.torch.log_softmax(lo, -1)
            res.append([float(lp[i]) for i in ids])
        return res


class VLLMEngine:
    """vLLM on one model; generation and first-token log-probs over a fixed id set."""

    def __init__(self, model: str, revision: str | None, **kw):
        from vllm import LLM

        t = time.time()
        self.llm = LLM(model=model, revision=revision, **kw)
        self.load_s = time.time() - t

    def generate(self, prompts: list[list[int]], max_tokens: int, lora=None) -> list[dict]:
        from vllm import SamplingParams, TokensPrompt

        sp = SamplingParams(max_tokens=max_tokens, seed=0, temperature=SAMPLER["temperature"],
                            top_p=SAMPLER["top_p"], top_k=SAMPLER["top_k"] or -1)
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                                lora_request=lora)
        return [{"ids": list(r.outputs[0].token_ids), "text": r.outputs[0].text,
                 "hit_cap": r.outputs[0].finish_reason == "length"} for r in res]

    def first_logprobs(self, prompts: list[list[int]], ids: list[int], lora=None):
        from vllm import SamplingParams, TokensPrompt

        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        res = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp,
                                lora_request=lora)
        out = []
        for r in res:
            lp = r.outputs[0].logprobs[0]
            out.append([float(lp[i].logprob) if i in lp else -1e9 for i in ids])
        return out

    def kv_tokens(self) -> int | None:
        try:
            cc = self.llm.llm_engine.vllm_config.cache_config
            return int(cc.num_gpu_blocks * cc.block_size)
        except Exception:  # noqa: BLE001 - recorded as unknown, not a failure
            return None


def _subject_engine(path: str, dry: bool, *, lora: bool = False, spec_on: bool = True):
    if dry:
        return HFStandIn(path)
    import vllm_phase as VP

    cfg = VP.engine_json(_spec(), spec_on=spec_on and not lora, label="validate")
    kw = dict(cfg["llm_kwargs"], max_model_len=MAX_MODEL_LEN)
    if lora:
        kw.update(enable_lora=True, max_lora_rank=LORA["rank"], max_loras=2)
    return VLLMEngine(path, None, **kw)


# ---- shared inputs -----------------------------------------------------------------------------

def _rows(tok, fmt, kind: str, n: int, body: str | None = None) -> list[dict]:
    """``n`` panel items of record (clear directed-away first, seeded order) rendered under the
    operator condition (or A0s), effort medium."""
    import random

    import cst_pressure as P
    import scenario_labels as L

    from deepsteer.kdg.schema import assign_letters

    rec = json.loads((CST / "data/scenario_labels_v4_record.json").read_text())["labels"]
    ids = [r["scenario_id"] for r in rec if r["set"] == "panel"]
    random.Random(20261106).shuffle(ids)
    _, by, ordered = L._ordered_and_opts()
    rows = []
    for sid in ids[:n]:
        s, _ = by[sid]
        order = assign_letters(s, 0)
        msgs = P.triple_messages(s, order, "operator", "panel")
        if kind == "a0s":
            msgs = P.a0s_messages(msgs, body)
        text = fmt.render(tok, msgs, EFFORT)
        rows.append({"scenario_id": sid, "order": [(L_, o.option_id) for L_, o in order],
                     "ids": tok.encode(text, add_special_tokens=False),
                     "dose0_ids": fmt.dose0_ids(tok, msgs, EFFORT, fmt.dose0[0])})
    return rows


def _parse_rows(tok, fmt, rows, gens) -> dict:
    import reasoning_formats as rf

    letters = rf.letter_token_ids(tok)
    st, lens, parsed = [], [], 0
    for r, g in zip(rows, gens):
        p = fmt.parse(tok, g["ids"], g["hit_cap"], letters)
        st.append(p.status)
        lens.append(p.trace_len)
        parsed += p.letter_record is not None
        g["parse"] = {"status": p.status, "trace_len": p.trace_len, "letter": p.letter_record}
    return {"n": len(rows), "parse_rate": parsed / len(rows) if rows else None,
            "trace_len_median": statistics.median(lens) if lens else None,
            "status": {k: st.count(k) for k in set(st)}}


# ---- stages ------------------------------------------------------------------------------------

def stage_render(out: Path, dry: bool) -> None:
    import cst_pressure as P

    want = {(r["set"], r["scenario_id"], r["condition"], r["effort"]): r["sha256"]
            for r in json.loads((CST / "outputs/t03_pressure_triple.json").read_text())["rows"]}
    if dry:
        _w(out, "render", {"skipped": "dry run (the pinned tokenizer renders in the local tests)",
                           "passes": True})
        return
    got = P.render_samples()
    rows = got["rows"]
    bad = [r for r in rows if want.get((r["set"], r["scenario_id"], r["condition"],
                                        r["effort"])) != r["sha256"]]
    _w(out, "render", {"n": len(rows), "mismatched": len(bad), "passes": not bad,
                       "template_sha256": got["template_sha256"]})


def stage_gen(out: Path, dry: bool) -> None:
    import reasoning_formats as rf

    from deepsteer.constitutions import load_document

    path = _model_path(dry)
    tok = _tok(path)
    fmt = rf.build_format("qwen38_27b", tok)
    ST = ROOT / "papers/st_live_reference"
    body = load_document("C_anth", ST / "constitutions.yaml", CONST_DIR).body \
        if not dry else "## Overview\nClaude should be honest."
    n_p, n_a, n_q, n_n = (4, 2, 2, 4) if dry else (N_PANEL, N_A0S, N_A0S_PROBE, N_NEUTRAL)
    panel = _rows(tok, fmt, "operator", n_p)
    a0s = _rows(tok, fmt, "a0s", n_a, body)
    neutral = [{"prompt": q, "ids": tok.encode(fmt.render(tok, [{"role": "user", "content": q}],
                                                          EFFORT), add_special_tokens=False)}
               for q in NEUTRAL[:n_n]]
    eng = _subject_engine(path, dry)
    rec: dict = {"max_model_len": MAX_MODEL_LEN, "load_s": getattr(eng, "load_s", None),
                 "kv_capacity_tokens": eng.kv_tokens() if hasattr(eng, "kv_tokens") else None,
                 "a0s_prompt_tokens": max(len(r["ids"]) for r in a0s)}
    for name, rows, cap in (("panel", panel, CAP), ("a0s", a0s, CAP),
                            ("a0s_probe", a0s[:n_q], CAP_PROBE), ("neutral", neutral, CAP)):
        t = time.time()
        gens = eng.generate([r["ids"] for r in rows], cap)
        dt = time.time() - t
        rec[name] = {"seconds": dt, "rollouts_per_hour": len(rows) / dt * 3600 if dt else None,
                     "gen_tokens": sum(len(g["ids"]) for g in gens)}
        if name != "neutral":
            rec[name].update(_parse_rows(tok, fmt, rows, gens))
            lens = sorted(g["parse"]["trace_len"] or 0 for g in gens)
            rec[name].update({"cap": cap, "trace_len_p90": lens[int(0.9 * (len(lens) - 1))],
                              "share_trace_over_1536": sum(x > CAP for x in lens) / len(lens)})
        (out / f"gen_{name}.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with open(out / f"gen_{name}.jsonl", "w") as f:
            for r, g in zip(rows, gens):
                # the A0s prompts' ids (~35k each) are not kept; their count is
                keep = {k: v for k, v in r.items() if k != "dose0_ids"
                        and not (k == "ids" and name.startswith("a0s"))}
                keep["prompt_tokens"] = len(r["ids"])
                f.write(json.dumps({**keep,
                                    "gen_ids": g["ids"], "text": g["text"],
                                    "hit_cap": g["hit_cap"], "parse": g.get("parse")}) + "\n")
        json.dump([r["dose0_ids"] for r in rows if "dose0_ids" in r][:N_LETTER],
                  open(out / f"dose0_{name}.json", "w")) if name == "panel" else None
    need = rec["a0s_prompt_tokens"] + CAP
    rec["v1_passes"] = rec["a0s"]["parse_rate"] is not None or dry
    rec["v2_kv_ok"] = (rec["kv_capacity_tokens"] or 0) >= need if not dry else True
    _w(out, "gen", rec)


def _sdf_batches(tok, n_seq: int, seq: int) -> list[list[int]]:
    from deepsteer.constitutions import load_document

    ST = ROOT / "papers/st_live_reference"
    try:
        body = load_document("C_anth", ST / "constitutions.yaml", CONST_DIR).body
    except FileNotFoundError:
        body = "Claude should be honest and helpful. " * 400
    ids = tok.encode(body, add_special_tokens=False)
    while len(ids) < n_seq * seq:
        ids = ids + ids
    return [ids[i * seq:(i + 1) * seq] for i in range(n_seq)]


def _lora_targets(model, allowed: tuple[str, ...]) -> list[str]:
    """ST's LoRA targets (``objective_lora.lora_targets``' rule) on a bare HF model: every Linear
    inside the text decoder layers (the registry's ``layers_path``) whose name ends in an allowed
    projection; the vision tower and the MTP head are outside by construction."""
    import torch

    layers = model
    for part in _spec()["layers_path"].split("."):
        layers = getattr(layers, part)
    names = {id(m): n for n, m in model.named_modules()}
    prefix = names[id(layers)]
    return sorted(n for n, m in model.named_modules()
                  if n.startswith(prefix + ".") and isinstance(m, torch.nn.Linear)
                  and n.rsplit(".", 1)[-1] in allowed)


def stage_lora_hf(out: Path, dry: bool) -> None:
    import reasoning_formats as rf
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText

    from deepsteer.steering.objective_lora import ATTN, LINEAR_ATTN, MLP

    path = _model_path(dry)
    tok = _tok(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    base = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(dev)
    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    mods = _lora_targets(base, ATTN + MLP + LINEAR_ATTN)
    kinds = sorted({m.split(".")[-1] for m in mods})
    cfg = LoraConfig(r=LORA["rank"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
                     bias="none", target_modules=mods)
    model = get_peft_model(base, cfg)
    seq = 64 if dry else SDF_SEQ
    steps = 2 if dry else SDF_STEPS
    batches = _sdf_batches(tok, steps * SDF_ACCUM, seq)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    times = []
    for s in range(steps):
        t = time.time()
        for a in range(SDF_ACCUM):
            x = torch.tensor([batches[s * SDF_ACCUM + a]], device=dev)
            loss = model(input_ids=x, labels=x).loss / SDF_ACCUM
            loss.backward()
        opt.step()
        opt.zero_grad()
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t)
    tok_s = (SDF_ACCUM * seq) / statistics.mean(times[1:] or times)
    rec: dict = {"lora_target_kinds": kinds, "n_target_modules": len(mods),
                 "deltanet_targets": any(k.startswith("in_proj") for k in kinds),
                 "sdf_tokens_per_s": tok_s, "sdf_10M_hours": 1e7 / tok_s / 3600,
                 "sdf_final_loss": float(loss.detach()) * SDF_ACCUM}
    # written after each step: the first run lost V5's measured rate to the OOM in a later step
    _w(out, "lora_hf", rec)
    del opt, loss
    # V3: save, reload, bit-identical logits on two prompts
    model.eval()
    probe = torch.tensor([tok.encode("The constitution says", add_special_tokens=False)],
                         device=dev)
    with torch.no_grad():
        before = model(input_ids=probe).logits.float().cpu()
    adir = out / "adapter_sdf"
    model.save_pretrained(adir)
    base2 = model.unload()
    re_m = PeftModel.from_pretrained(base2, adir).to(dev)
    re_m.eval()
    with torch.no_grad():
        after = re_m(input_ids=probe).logits.float().cpu()
    rec["v3_reload_max_abs_diff"] = float((before - after).abs().max())
    rec["v3_passes"] = rec["v3_reload_max_abs_diff"] == 0.0
    _w(out, "lora_hf", rec)
    # first-token letter distributions on the dose-0 panel prompts, base and attached (for V4)
    letter_ids = sorted(rf.letter_token_ids(tok))
    d0 = json.loads((out / "dose0_panel.json").read_text())
    def lp(m):
        res = []
        for p in d0:
            with torch.no_grad():
                lo = m(input_ids=torch.tensor([p], device=dev)).logits[0, -1].float()
            v = torch.log_softmax(lo, -1)
            res.append([float(v[i]) for i in letter_ids])
        return res
    attached = lp(re_m)
    with re_m.disable_adapter():
        base_lp = lp(re_m)
    json.dump({"letter_ids": letter_ids, "base": base_lp, "attached": attached},
              open(out / "hf_letters.json", "w"))
    rec["letters_written"] = len(attached)
    _w(out, "lora_hf", rec)
    # recovery SFT on the gen stage's neutral replies (loss on the reply tokens)
    neu = [json.loads(x) for x in open(out / "gen_neutral.jsonl")]
    re_m.train()
    for p in re_m.parameters():
        p.requires_grad_(False)
    for n_, p in re_m.named_parameters():
        if "lora_" in n_:
            p.requires_grad_(True)
    opt2 = torch.optim.AdamW([p for p in re_m.parameters() if p.requires_grad], lr=2e-4)
    for r in neu:
        ids = r["ids"] + r["gen_ids"]
        lab = [-100] * len(r["ids"]) + r["gen_ids"]
        loss2 = re_m(input_ids=torch.tensor([ids], device=dev),
                     labels=torch.tensor([lab], device=dev)).loss
        loss2.backward()
    opt2.step()
    opt2.zero_grad()
    re_m.save_pretrained(out / "adapter_sdf_recovery")
    rec["recovery_examples"] = len(neu)
    rec["passes"] = rec["v3_passes"] and rec["deltanet_targets"]
    _w(out, "lora_hf", rec)


def gradnec_backward(model, x) -> dict:
    """One Grad-Nec backward (D8's memory read): the gradient of the last position's logsumexp to
    the input embeddings, with checkpointing ACTIVE (train mode; HF checkpoints only when
    ``model.training``; the first run called it in eval mode and stored every layer's activations)
    and the last position's logits only (``logits_to_keep=1``; full logits at 152k vocab are
    1.2 GB at 2,048 tokens in fp32). No weight gets a gradient."""
    import torch

    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()(x).detach().requires_grad_(True)
    lo = model(inputs_embeds=emb, logits_to_keep=1).logits
    lo[0, -1].float().logsumexp(-1).backward()
    g = emb.grad
    return {"checkpointing_active": bool(model.is_gradient_checkpointing and model.training),
            "logits_positions": int(lo.shape[1]),
            "grad_finite": bool(torch.isfinite(g).all()), "grad_norm": float(g.norm())}


def stage_gradnec(out: Path, dry: bool) -> None:
    """V7 on its own model load (A0: no adapter), at 2,048 tokens (the bar) and 3,584 (an operator
    rollout's prompt plus the full trace cap)."""
    import torch
    from transformers import AutoModelForImageTextToText

    path = _model_path(dry)
    tok = _tok(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(dev)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.config.use_cache = False
    rec: dict = {"dtype": str(dtype)}
    for n in ((64, 96) if dry else GRADNEC_LENS):
        ids = _sdf_batches(tok, 2, n)
        x = torch.tensor([ids[0][:n]], device=dev)
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        r = gradnec_backward(m, x)
        if dev == "cuda":
            torch.cuda.synchronize()
        r["seconds"] = time.time() - t
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
        rec[f"v7_{n}"] = r
        _w(out, "gradnec", rec)
    rec["v7_passes"] = all(rec[k]["grad_finite"] for k in rec if k.startswith("v7_"))
    _w(out, "gradnec", rec)


def stage_vllm_lora(out: Path, dry: bool) -> None:
    import reasoning_formats as rf

    path = _model_path(dry)
    tok = _tok(path)
    fmt = rf.build_format("qwen38_27b", tok)
    hf = json.loads((out / "hf_letters.json").read_text())
    d0 = json.loads((out / "dose0_panel.json").read_text())
    rec: dict = {}

    def tv(a, b):
        import math

        tot = 0.0
        for x, y in zip(a, b):
            px = [math.exp(v) for v in x]
            py = [math.exp(v) for v in y]
            sx, sy = sum(px) or 1, sum(py) or 1
            tot += 0.5 * sum(abs(p / sx - q / sy) for p, q in zip(px, py))
        return tot / len(a)

    if dry:
        eng = HFStandIn(path, lora=str(out / "adapter_sdf"))
        served = eng.first_logprobs(d0, hf["letter_ids"])
        rec["path"] = "dry run (HF stand-in)"
    else:
        from vllm.lora.request import LoRARequest

        try:
            eng = _subject_engine(path, dry, lora=True)
            req = LoRARequest("sdf", 1, str(out / "adapter_sdf"))
            served = eng.first_logprobs(d0, hf["letter_ids"], lora=req)
            rec["path"] = "vllm native LoRA"
        except Exception as e:  # noqa: BLE001 - the fallback is the registered branch
            rec["native_lora_error"] = repr(e)[:400]
            import t16_driver as T16

            merged = out / "_merged_sdf"
            rec["merge"] = T16.write_merged(Path(path), out / "adapter_sdf", merged)
            eng = _subject_engine(str(merged), dry, spec_on=False)
            served = eng.first_logprobs(d0, hf["letter_ids"])
            rec["path"] = "merged (ST's one-delta merge writer)"
            req = None
    rec["tv_served_vs_attached"] = tv(served, hf["attached"])
    rec["tv_base_vs_attached"] = tv(hf["base"], hf["attached"])
    rec["v4_passes"] = rec["tv_served_vs_attached"] < rec["tv_base_vs_attached"]
    # V6: format survival of the SDF adapter, with and without recovery, against none (gen stage)
    if not dry:
        rows = [json.loads(x) for x in open(out / "gen_panel.jsonl")]
        base_stats = json.loads((out / "gen.json").read_text())["panel"]
        for name in ("adapter_sdf", "adapter_sdf_recovery"):
            if rec["path"].startswith("vllm"):
                r2 = LoRARequest(name, 2 if "recovery" in name else 1, str(out / name))
                gens = eng.generate([r["ids"] for r in rows], CAP, lora=r2)
                st = _parse_rows(tok, fmt, rows, gens)
                rec[f"v6_{name}"] = st
        def within(a, b):
            return b and abs(a - b) <= 0.1 * b
        rec["v6_passes"] = bool(rec.get("v6_adapter_sdf_recovery")) and within(
            rec["v6_adapter_sdf_recovery"]["parse_rate"], base_stats["parse_rate"]) and within(
            rec["v6_adapter_sdf_recovery"]["trace_len_median"], base_stats["trace_len_median"])
    _w(out, "vllm_lora", rec)


def stage_judges(out: Path, dry: bool, which: list[str]) -> None:
    import cst_judge as J
    import mrb_g1_pod as M
    import scenario_labels as L
    from generate_scenarios import extract_json

    from deepsteer.constitutions import TenetSet
    from deepsteer.reasoning.think_io import CoTFormat

    all_rows = [json.loads(x) for x in open(out / "gen_panel.jsonl")]
    tset = TenetSet.load(CST / "data/tenets_C_anth_v2.json")
    tenets = tset.by_id()
    # the relabel probe (component 4): the first N_RELABEL_CALLS ten-item batches, v4 prompt
    items = [(s, k) for k, v in L.scored_sets().items() for s in v]
    all_relabel = [items[i * 10:(i + 1) * 10] for i in range(N_RELABEL_CALLS)]
    labs = {f: {r["scenario_id"]: r for r in json.loads(
        (CST / f"data/scenario_labels_v4_{f}.json").read_text())["labels"]}
        for f in ("claude", "codex")}
    _, by, ordered = L._ordered_and_opts()
    rec: dict = {}
    for name in which:
        j = JUDGES[name]
        rows = all_rows[:j.get("n_score", len(all_rows))]
        relabel = all_relabel[:j.get("n_relabel", N_RELABEL_CALLS)]
        fmt = CoTFormat.HARMONY_ANALYSIS if j["fmt"] == "harmony" else CoTFormat.THINK_TAGS
        if dry:  # every step but the vLLM call: the subject's tokenizer, a canned reply
            eng = None
            jtok = _tok(_model_path(True))
            fmt = CoTFormat.THINK_TAGS
            cfg = None
        else:
            from huggingface_hub import snapshot_download

            jpath = snapshot_download(j["repo"], revision=j["revision"],
                                      ignore_patterns=["original/*", "metal/*"])
            eng = VLLMEngine(jpath, None, max_model_len=JUDGE_LEN, gpu_memory_utilization=0.9,
                             enable_prefix_caching=True, seed=0, **j.get("llm_kwargs", {}))
            jtok = eng.llm.get_tokenizer()
            cfg = M.harmony_cfg() if j["fmt"] == "harmony" else None
        prompts, meta = [], []
        for r in rows:
            sid = r["scenario_id"]
            s, _ = by[sid]
            short = sorted(set(labs["claude"][sid]["shortlist"]) | set(
                labs["codex"][sid]["shortlist"]) | set(L.DEFERENCE))
            trace, answer = r["text"].split("</think>")[0], r["text"].split("</think>")[-1]
            letter_to_opt = dict(r["order"])
            p = J.judge_prompt(s.agent_text, [(o, s.option(o).text) for _, o in r["order"]],
                               trace, answer, [(t, tenets[t].claim) for t in short])
            prompts.append(_judge_ids(jtok, J.SYS + "\n\n" + p, j["fmt"], fmt, cfg))
            meta.append((set(letter_to_opt.values()), set(short)))
        rl_prompts = [_judge_ids(jtok, L.SYS + "\n\n" + L._prompt_v4(b, tset), j["fmt"], fmt, cfg)
                      for b in relabel]
        t = time.time()
        if dry:
            texts = [f"<think>reading</think>{{\"concluded_option\": \"{sorted(o)[0]}\", "
                     f"\"applied_tenets\": [\"{sorted(sh)[0]}\"], \"evidence\": \"e\"}}"
                     for o, sh in meta]
            toks = [len(jtok.encode(x)) for x in texts]
            caps = [False] * len(texts)
        else:
            texts, toks, caps = _judge_generate(eng, prompts)
        dt = max(time.time() - t, 1e-6)
        ok, raw = 0, []
        for (opts, short), text, n_tok, cap, r in zip(meta, texts, toks, caps, rows):
            _, final = M.split_trace(text, fmt)
            pj = J.parse_judgment(final or text, opts, short)
            ok += pj["ok"]
            raw.append({"scenario_id": r["scenario_id"], "out_tokens": n_tok, "hit_cap": cap,
                        "ok": pj["ok"], "text": text})
        # raw replies saved: the first run's GLM and Olmo failures could not be diagnosed from
        # the parse rate alone
        with open(out / f"judges_raw_{name}.jsonl", "w") as f:
            for x in raw:
                f.write(json.dumps(x) + "\n")
        lens = sorted(toks)
        n_ok_free = sum(x["ok"] for x in raw if not x["hit_cap"])
        n_free = sum(not x["hit_cap"] for x in raw)
        rec[name] = {"load_s": getattr(eng, "load_s", None), "n": len(rows),
                     "fmt": j["fmt"], "llm_kwargs": j.get("llm_kwargs", {}),
                     "kv_capacity_tokens": eng.kv_tokens() if eng else None,
                     "max_tokens": JUDGE_MAX_TOKENS, "max_model_len": JUDGE_LEN,
                     "parse_rate": ok / len(rows),
                     "parse_rate_uncapped": n_ok_free / n_free if n_free else None,
                     "hit_cap_share": sum(caps) / len(caps),
                     "out_tokens_median": statistics.median(lens),
                     "out_tokens_p90": lens[int(0.9 * (len(lens) - 1))],
                     "calls_per_hour": len(rows) / dt * 3600, "output_tok_s": sum(toks) / dt,
                     "mean_prompt_tokens": statistics.mean(len(p) for p in prompts),
                     "passes": ok / len(rows) >= 0.9}
        # the relabel probe: one batch per call, parse = every item with a valid deference value
        t = time.time()
        if dry:
            rl_texts = ["<think>r</think>" + json.dumps({"labels": [
                {"scenario_id": s_.id, "deference": {"value": "ordered",
                                                     "ground": "business_reason_no_harm"},
                 "shortlist": [], "directions": {}} for s_, _ in b]}) for b in relabel]
            rl_toks, rl_caps = [len(jtok.encode(x)) for x in rl_texts], [False] * len(relabel)
        else:
            rl_texts, rl_toks, rl_caps = _judge_generate(eng, rl_prompts)
        rl_dt = max(time.time() - t, 1e-6)
        got = 0
        for b, text in zip(relabel, rl_texts):
            _, final = M.split_trace(text, fmt)
            try:
                lab = extract_json(L.repair_evidence(final or text)[0])["labels"]
            except (ValueError, KeyError):
                lab = []
            want = {s_.id for s_, _ in b}
            got += sum(1 for x in lab if x.get("scenario_id") in want and
                       (x.get("deference") or {}).get("value") in L.GROUNDS)
        with open(out / f"judges_relabel_raw_{name}.jsonl", "w") as f:
            for text, n_tok, cap in zip(rl_texts, rl_toks, rl_caps):
                f.write(json.dumps({"out_tokens": n_tok, "hit_cap": cap, "text": text}) + "\n")
        rec[name]["relabel"] = {
            "calls": len(relabel), "items": sum(len(b) for b in relabel), "items_parsed": got,
            "prompt_tokens": [len(p) for p in rl_prompts], "out_tokens": rl_toks,
            "hit_cap": rl_caps, "seconds": rl_dt}
        del eng
        if not dry:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
    _w(out, "judges", rec)


def _judge_ids(tok, text: str, mode: str, fmt, cfg) -> list[int]:
    """A judge prompt's token ids: harmony at effort low (MRB's judge of record), the model's own
    template otherwise; ``nothink`` passes ``enable_thinking=False`` (GLM's template closes the
    think block in the prompt)."""
    import mrb_g1_pod as M

    if mode != "nothink":
        return M.render_ids(tok, text, fmt, "low", cfg)
    r = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok(r, add_special_tokens=False)["input_ids"]


def _judge_generate(eng, prompts: list[list[int]]) -> tuple[list[str], list[int], list[bool]]:
    from vllm import SamplingParams, TokensPrompt

    # per-request budget: a long relabel prompt keeps prompt + output inside JUDGE_LEN
    sps = [SamplingParams(max_tokens=min(JUDGE_MAX_TOKENS, JUDGE_LEN - len(p) - 16),
                          temperature=0.0, seed=0) for p in prompts]
    res = eng.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sps)
    return ([o.outputs[0].text for o in res], [len(o.outputs[0].token_ids) for o in res],
            [o.outputs[0].finish_reason == "length" for o in res])


PER_ARM = 586 * 16 + 132 * 8 * 2 + 102 * 8  # panel (plain 4, user 2, operator 4, operator_pr 6)
RELABEL_CALLS = -(-718 // 10)  # component 4: the 718 items at ten per call, every judge
VOTE_CALLS = -(-(207 + 13) // 10)  # component 5: T_core's 207 splits and T_conf's 13, third judge
LAYER_A_CALLS = 15  # 150 Scruples pairs at ten per call, every judge (prompt like the relabel's)
RETEST = 100  # Amendment 7 item 4: re-scored rollouts per judge


def _judge_hours(j: dict, third: bool) -> dict:
    """One judge's S0a hours from its measured rates: load, scoring both arms plus the retest,
    and the long-prompt calls (relabel, Layer A, and the third judge's votes) at the relabel
    probe's measured seconds per call (its calls ran concurrently, so this is an upper bound)."""
    rl = j.get("relabel") or {}
    s_call = rl["seconds"] / rl["calls"] if rl.get("calls") else None
    long_calls = RELABEL_CALLS + LAYER_A_CALLS + (VOTE_CALLS if third else 0)
    h = {"load": (j.get("load_s") or 0) / 3600,
         "scoring": (2 * PER_ARM + RETEST) / j["calls_per_hour"],
         "long_prompt_calls": long_calls * s_call / 3600 if s_call else None}
    h["total"] = sum(v for v in h.values() if v is not None)
    return h


def stage_price(out: Path) -> None:
    """S0a's price (Amendment 7 item 2: components 1-8 and 12) from VALIDATE's measured rates; S0b
    is not priced until ST's A0 reads on Qwen3.8 ``none`` are listed (same item). Every judge
    candidate measured is priced; ``panels`` sums the registered triple and each substitution."""
    g = json.loads((out / "gen.json").read_text())
    jd = json.loads((out / "judges.json").read_text())
    subject = {"load": (g.get("load_s") or 0) / 3600,
               "a0_generation": PER_ARM / g["panel"]["rollouts_per_hour"],
               "a0s_generation": PER_ARM / g["a0s"]["rollouts_per_hour"],
               # forward passes only: dose-0 letter reads (ST's E and the forced choice) on both
               # arms and the recall read (240 windows x 3 variants x 96 tokens)
               "prefill_reads_and_recall": 0.15}
    fixed = {"setup_and_downloads": 0.35,
             "capability_800_two_arms": 2 * 1.2}  # carried from ST's reader, not measured here
    judges = {k: _judge_hours(v, third=k not in ("gpt_oss_120b",))
              for k, v in jd.items() if "calls_per_hour" in v}
    base = sum(subject.values()) + sum(fixed.values())
    panels = {}
    for third in ("olmo31_32b_think", "olmo31_32b_think_fp8kv", "olmo31_32b_instruct"):
        for glm in ("glm47_flash", "glm47_flash_nothink"):
            names = ("gpt_oss_120b", glm, third)
            if all(n in judges for n in names):
                t = base + sum(judges[n]["total"] for n in names)
                panels[" + ".join(names)] = {"a100_hours": t, "with_10pct_margin": t * 1.1}
    rec = {"rollouts_per_arm": PER_ARM, "subject_hours": subject, "fixed_hours": fixed,
           "judge_hours": judges, "s0a_base_hours": base, "panels": panels,
           "a0s_cap": g["a0s"].get("cap", CAP),
           "note": ("S0a only; the A0s rate is measured on the operator condition and applied to "
                    "every condition; capability is ST's rate, carried")}
    _w(out, "price", rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("render", "gen", "lora_hf", "vllm_lora", "gradnec", "judges",
                             "price"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--judges", default=",".join(JUDGES))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"render": lambda: stage_render(a.out, a.dry_run),
     "gen": lambda: stage_gen(a.out, a.dry_run),
     "lora_hf": lambda: stage_lora_hf(a.out, a.dry_run),
     "vllm_lora": lambda: stage_vllm_lora(a.out, a.dry_run),
     "gradnec": lambda: stage_gradnec(a.out, a.dry_run),
     "judges": lambda: stage_judges(a.out, a.dry_run, a.judges.split(",")),
     "price": lambda: stage_price(a.out)}[a.stage]()


if __name__ == "__main__":
    main()
