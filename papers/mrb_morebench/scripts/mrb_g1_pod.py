"""MRB-G1 pod driver (MRB_PLAN v0.3 §2-5). Runs in the vLLM venv on the pod.

    generate --model {gpt_oss_20b,qwen3_30b} --out DIR [--validate] [--deadline EPOCH]
    judge --shard I/N --tiers T1,T2,... --out DIR [--validate] [--deadline EPOCH]
    project --out DIR --validate-hours H [--envelope-h 8]
    mb-rider --out DIR [--validate]
    --dry-run (any subcommand): a stub engine and cached tokenizers, no GPU (local tests)

Generation: k = 4 per scenario (seeded), Table 4 sampling, effort high for gpt-oss, 10,500 tokens;
the generation fork (each model's own defaults) at k = 1 on the ladder subset. Traces and responses
are split with ``deepsteer.reasoning.think_io``. Judging: gpt-oss-120b at effort high, one call per
criterion and field, in priority tiers whose order is the reverse of the author's cut order (T4 =
the ladder beyond 100 scenarios, cut a; T3 = Qwen k 2-3, cut b; T2 = GPT-OSS-20B k 2-3, cut c),
so a tier left out is exactly a cut. Outputs are written as they finish and a rerun resumes.

Tracked (``DIR``): ids, lengths, token counts, finish reasons, verdict strings of at most 20
characters, credit under the substring and exact-token tests, text hashes. Untracked
(``DIR/_raw``): generations and judge replies, which can quote scenario text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "papers" / "st_live_reference" / "scripts"),
                str(REPO / "papers" / "kdg_panel" / "scripts"),
                str(REPO / "papers" / "mb_moralbench" / "scripts")]

from deepsteer.benchmarks.moral_reasoning.morebench.prompts import (  # noqa: E402
    generation_prompt,
    judge_prompt,
)
from deepsteer.benchmarks.moral_reasoning.morebench.scoring import fulfilled  # noqa: E402
from deepsteer.reasoning.think_io import CoTFormat, decode_rollout, split_rollout  # noqa: E402

POD_VERSION = "1.0.0"
K = 4
MAX_TOKENS = 10_500
CHUNK = 2048
CANNED = REPO / "papers" / "mrb_morebench" / "data" / "canned_ladder_v1.json"
REGISTRY = REPO / "papers" / "kdg_panel" / "models.yaml"
GEN = {
    "gpt_oss_20b": {"repo": "openai/gpt-oss-20b",
                    "revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                    "format": CoTFormat.HARMONY_ANALYSIS, "effort": "high",
                    "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 0},
                    "fork_sampler": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}},
    "qwen3_30b": {"repo": "Qwen/Qwen3-30B-A3B-Thinking-2507",
                  "revision": "144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
                  "format": CoTFormat.THINK_TAGS, "effort": None,
                  "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 0},
                  "fork_sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},
}
JUDGE = {"repo": "openai/gpt-oss-120b", "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
         "effort": "high", "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 0},
         "fork_sampler": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}}
# VALIDATE: 16 scenarios x 4 generations per model (64 concurrent requests for the rate) and the
# judge on all of them; the ladder on two
VALIDATE_SCENARIOS = tuple(range(0, 500, 32))[:16]
# a judge prompt holds a whole trace (up to ~10.5k tokens) plus up to 10.5k output tokens
LLM_KWARGS = {"gen": {"max_model_len": 16384, "gpu_memory_utilization": 0.90,
                      "enable_prefix_caching": True},
              # gpt-oss-120b across both GPUs (tensor parallel): on one A100 its 66 GiB of weights
              # leave 4.5 GiB for the KV cache (the second VALIDATE), which starves concurrency
              "judge": {"max_model_len": 24576, "gpu_memory_utilization": 0.92,
                        "enable_prefix_caching": True, "tensor_parallel_size": 2}}


def in_ladder_subset(idx: int) -> bool:  # 100 scenarios (cut a keeps these)
    return idx % 5 == 0


def in_j1_subset(idx: int) -> bool:  # 50 scenarios (J1 test-retest, judge fork)
    return idx % 10 == 0


def seed_of(*parts: object) -> int:
    return zlib.crc32("/".join(map(str, parts)).encode()) % (2**31)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---- rendering --------------------------------------------------------------------------------


def harmony_cfg():
    import kdg_harmony as kh
    import yaml
    spec = yaml.safe_load(REGISTRY.read_text())["tier2"]["gpt_oss_20b"]
    return kh.HarmonyConfig.from_registry(spec)


def render_ids(tok, text: str, fmt: CoTFormat, effort: str | None, cfg=None) -> list[int]:
    """Token ids of one user turn under the model's own template (harmony date-pinned)."""
    if fmt == CoTFormat.HARMONY_ANALYSIS:
        import kdg_harmony as kh
        rendered = kh.render(tok, [{"role": "user", "content": text}], cfg, effort)
    else:
        rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True)
    return tok(rendered, add_special_tokens=False)["input_ids"]


# ---- verdicts ---------------------------------------------------------------------------------


_TOKEN = re.compile(r"[a-z]+")


def exact_verdict(final: str | None) -> str | None:
    """'yes' / 'no' when the reply's only word is yes or no (punctuation ignored), else None."""
    words = _TOKEN.findall((final or "").lower())
    return words[0] if len(words) == 1 and words[0] in ("yes", "no") else None


def credits(final: str | None, weight: int) -> dict:
    """Upstream's substring credit (of record) and the exact-token credit beside it (MRB-A3)."""
    ex = exact_verdict(final)
    return {"credit_substring": bool(final is not None and fulfilled(final, weight)),
            "credit_exact": bool((ex == "yes" and weight > 0) or (ex == "no" and weight < 0))}


# ---- work lists -------------------------------------------------------------------------------


def gen_jobs(items, model: str, tok, validate: bool, cfg=None) -> list[dict]:
    spec = GEN[model]
    keep = [it for it in items if not validate or it.idx in VALIDATE_SCENARIOS]
    jobs = []
    for it in keep:
        ids = render_ids(tok, generation_prompt(it.dilemma), spec["format"], spec["effort"], cfg)
        arms = [("default", k, spec["sampler"]) for k in range(K)]
        if not validate and in_ladder_subset(it.idx):
            arms.append(("fork", 0, spec["fork_sampler"]))
        for arm, k, sampler in arms:
            gid = f"{model}/{arm}/{it.scenario_id}/k{k}"
            jobs.append({"job_id": gid, "prompt_ids": ids, "max_tokens": MAX_TOKENS,
                         "seed": seed_of(model, arm, it.scenario_id, k), "sampler": sampler})
    return jobs


def judge_units(items, gens: dict[str, dict], validate: bool) -> list[dict]:
    """Every judge unit (one response text to judge on all of a scenario's criteria), tiered.

    Tier 1: k 0-1 of both models on both fields, the ladder on its 100-scenario subset, J1's
    second pass and the judge fork on the J1 subset, the generation fork. Tier 2: GPT-OSS-20B k
    2-3 (cut c). Tier 3: Qwen3-30B k 2-3 (cut b). Tier 4: the ladder on the other 400 (cut a)."""
    canned = json.loads(CANNED.read_text())["canned"]
    by_sid = {it.scenario_id: it for it in items}
    units = []
    for gid, g in gens.items():
        model, arm, sid, kk = gid.split("/")
        k = int(kk[1:])
        idx = by_sid[sid].idx
        if validate and k != 0:  # VALIDATE judges the first generation only (~2,000 calls)
            continue
        for field in ("trace", "response"):
            text = g.get(field)
            if text is None:
                continue
            tier = 1 if (arm == "fork" or k < 2) else (2 if model == "gpt_oss_20b" else 3)
            units.append({"unit": f"{gid}/{field}", "sid": sid, "text": text, "tier": tier,
                          "pass": 1, "judge": "default"})
            if model == "gpt_oss_20b" and arm == "default" and k == 0 and in_j1_subset(idx):
                units.append({"unit": f"{gid}/{field}/retest", "sid": sid, "text": text,
                              "tier": 1, "pass": 2, "judge": "default"})
                units.append({"unit": f"{gid}/{field}/judgefork", "sid": sid, "text": text,
                              "tier": 1, "pass": 1, "judge": "fork"})
    for it in items:
        if validate and it.idx not in VALIDATE_SCENARIOS[:2]:
            continue
        for name, c in canned.items():
            text = it.dilemma if c["kind"] == "scenario" else c["text"]
            units.append({"unit": f"canned/{name}/{it.scenario_id}", "sid": it.scenario_id,
                          "text": text, "tier": 1 if in_ladder_subset(it.idx) or validate else 4,
                          "pass": 1, "judge": "default"})
    return units


def calls_by_tier(units, items) -> dict[int, int]:
    n = {it.scenario_id: len(it.criteria) for it in items}
    out: dict[int, int] = {}
    for u in units:
        out[u["tier"]] = out.get(u["tier"], 0) + n[u["sid"]]
    return out


def judge_jobs(units, items, tok, cfg, shard: int, n_shards: int, tiers: set[int]):
    """Yield judge jobs for this shard, in tier order, each unit's criteria contiguous (prefix
    caching); the shard is chosen by the unit's crc32, so both shards see every tier."""
    by_sid = {it.scenario_id: it for it in items}
    for tier in sorted(tiers):
        for u in units:
            if u["tier"] != tier or zlib.crc32(u["unit"].encode()) % n_shards != shard:
                continue
            for c in by_sid[u["sid"]].criteria:
                jid = f"{u['unit']}/{c.criterion_id}"
                sampler = JUDGE["fork_sampler"] if u["judge"] == "fork" else JUDGE["sampler"]
                yield {"job_id": jid, "unit": u["unit"], "tier": tier, "pass": u["pass"],
                       "judge": u["judge"], "criterion_id": c.criterion_id, "weight": c.weight,
                       "prompt_ids": render_ids(tok, judge_prompt(u["text"], c.title),
                                                CoTFormat.HARMONY_ANALYSIS, JUDGE["effort"], cfg),
                       "max_tokens": MAX_TOKENS, "seed": seed_of("judge", jid, u["pass"]),
                       "sampler": sampler}


# ---- projection (VALIDATE) --------------------------------------------------------------------


CUTS = (("none", {1, 2, 3, 4}), ("a", {1, 2, 3}), ("a+b", {1, 2}), ("a+b+c", {1}))


def project(measured: dict, calls: dict[int, int], envelope_h: float, validate_h: float) -> dict:
    """Session A100-hours under each cut level, from VALIDATE's measured rates; the first level
    that fits the envelope's remainder is chosen, else STOP (MRB_PLAN v0.3 §5)."""
    remaining = envelope_h - validate_h
    gen_h = measured["gen_wall_h_full"]
    rows, chosen = [], None
    for name, tiers in CUTS:
        judge_calls = sum(calls.get(t, 0) for t in tiers)
        judge_wall_h = judge_calls / measured["judge_calls_per_s"] / 3600
        wall = measured["setup_h"] + gen_h + judge_wall_h + measured["margin_h"]
        a100_h = 2 * wall
        fits = a100_h <= remaining
        rows.append({"cut": name, "tiers": sorted(tiers), "judge_calls": judge_calls,
                     "judge_wall_h": round(judge_wall_h, 3), "session_wall_h": round(wall, 3),
                     "session_a100_h": round(a100_h, 3), "fits": fits})
        if fits and chosen is None:
            chosen = rows[-1]
    return {"envelope_h": envelope_h, "validate_h": validate_h, "remaining_h": remaining,
            "levels": rows, "chosen": chosen, "decision": chosen["cut"] if chosen else "STOP",
            "session_max_wall_h": round(chosen["session_wall_h"] * 1.15, 3) if chosen else None}


# ---- engines ----------------------------------------------------------------------------------


class StubEngine:
    """vLLM-engine stand-in for --dry-run: returns a fixed harmony or think-tag continuation."""

    def __init__(self, tok, fmt: CoTFormat, reply: str):
        trace, final = "I weigh the duties and the harms involved.", reply
        text = (f"<|channel|>analysis<|message|>{trace}<|end|><|start|>assistant"
                f"<|channel|>final<|message|>{final}<|return|>"
                if fmt == CoTFormat.HARMONY_ANALYSIS else f"{trace}\n</think>\n\n{final}")
        self.ids = tok(text, add_special_tokens=False)["input_ids"]
        self.pending: list[str] = []

    def add_request(self, rid, prompt, params):
        self.pending.append(rid)

    def has_unfinished_requests(self):
        return bool(self.pending)

    def abort_request(self, rids):
        self.pending = [r for r in self.pending if r not in set(rids)]

    def step(self):
        import types
        done, self.pending = self.pending, []
        return [types.SimpleNamespace(request_id=r, finished=True, outputs=[types.SimpleNamespace(
            token_ids=self.ids, finish_reason="stop", logprobs=None)]) for r in done]


def load_engine(repo: str, revision: str, local_dir: Path | None, dry: bool, tok, fmt, reply,
                role: str):
    if dry:
        return StubEngine(tok, fmt, reply), None
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(local_dir) if local_dir else repo, revision=None if local_dir else revision,
              tokenizer=str(local_dir) if local_dir else repo, **LLM_KWARGS[role])
    return llm.llm_engine, SamplingParams


def run_chunks(engine, SamplingParams, jobs_iter, out_path: Path, raw_path: Path, decode,
               *, deadline: float | None, meta_keys: tuple[str, ...]) -> dict:
    """Run jobs in chunks with gen_vllm's guarded loop; append finished rows; resume."""
    from gen_vllm import run_jobs, sampling_params
    done = set()
    if out_path.exists():
        done = {json.loads(x)["job_id"] for x in out_path.read_text().splitlines() if x.strip()}
    make = ((lambda j: sampling_params(SamplingParams, j)) if SamplingParams else (lambda j: None))
    t0, n, toks, stopped = time.time(), 0, 0, None
    chunk: list[dict] = []

    def flush(chunk):
        nonlocal n, toks, stopped
        rows, rec = run_jobs(engine, chunk, make, deadline=deadline)
        with open(out_path, "a") as fo, open(raw_path, "a") as fr:
            for j in chunk:
                r = rows[j["job_id"]]
                tracked, raw = decode(j, r)
                fo.write(json.dumps({**{k: j[k] for k in meta_keys if k in j}, **tracked}) + "\n")
                fr.write(json.dumps({"job_id": j["job_id"], **raw}) + "\n")
        n += len(chunk)
        toks += rec["decoded_tokens"]
        stopped = stopped or rec["stopped"]

    for j in jobs_iter:
        if j["job_id"] in done:
            continue
        chunk.append(j)
        if len(chunk) >= CHUNK:
            flush(chunk)
            chunk = []
            if stopped or (deadline and time.time() > deadline):
                stopped = stopped or "deadline"
                break
    if chunk and not stopped:
        flush(chunk)
    wall = time.time() - t0
    return {"jobs_run": n, "decoded_tokens": toks, "wall_s": round(wall, 1),
            "tokens_per_s": round(toks / max(wall, 1e-9), 1),
            "calls_per_s": round(n / max(wall, 1e-9), 3), "stopped": stopped}


# ---- subcommands ------------------------------------------------------------------------------


def _tok(repo: str, revision: str, local_dir: Path | None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(local_dir) if local_dir else repo,
                                         revision=None if local_dir else revision)


def _items():
    from deepsteer.datasets.morebench import load_morebench
    return load_morebench(subset="public")


def cmd_generate(a) -> dict:
    spec = GEN[a.model]
    tok = _tok(spec["repo"], spec["revision"], a.local_dir)
    cfg = harmony_cfg() if spec["format"] == CoTFormat.HARMONY_ANALYSIS else None
    items = _items()
    jobs = gen_jobs(items, a.model, tok, a.validate, cfg)
    engine, SP = load_engine(spec["repo"], spec["revision"], a.local_dir, a.dry_run, tok,
                             spec["format"], "I would choose the first option.", "gen")
    out = a.out / "gen" / f"{a.model}.jsonl"
    raw = a.out / "_raw" / "gen" / f"{a.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)

    def decode(j, r):
        text = decode_rollout(tok, r["gen_ids"], spec["format"]) if r["gen_ids"] else ""
        trace, resp = split_rollout(text, spec["format"]) if text else ("", "")
        final_reached = bool(resp)
        return ({"n_tokens": r["n_tokens"], "finish_reason": r["finish_reason"],
                 "status": r["status"], "final_reached": final_reached,
                 "trace_chars": len(trace), "response_chars": len(resp),
                 "trace_sha256": sha(trace), "response_sha256": sha(resp)},
                {"trace": trace, "response": resp})

    rec = run_chunks(engine, SP, iter(jobs), out, raw, decode, deadline=a.deadline,
                     meta_keys=("job_id",))
    rec.update({"model": a.model, "n_jobs": len(jobs), "validate": a.validate,
                "pod_version": POD_VERSION})
    (a.out / "gen" / f"{a.model}.summary.json").write_text(json.dumps(rec, indent=1))
    return rec


def load_generations(out_dir: Path) -> dict[str, dict]:
    gens = {}
    for f in sorted((out_dir / "_raw" / "gen").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            # an unreached final channel is an empty response: judged as written (upstream)
            gens[r["job_id"]] = {"trace": r["trace"], "response": r["response"]}
    return gens


def cmd_judge(a) -> dict:
    # gpt-oss-120b shares gpt-oss-20b's tokenizer; a dry run uses the cached 20b one
    tok = (_tok(GEN["gpt_oss_20b"]["repo"], GEN["gpt_oss_20b"]["revision"], None) if a.dry_run
           else _tok(JUDGE["repo"], JUDGE["revision"], a.local_dir))
    cfg = harmony_cfg()
    items = _items()
    units = judge_units(items, load_generations(a.out), a.validate)
    shard, n_shards = (int(x) for x in a.shard.split("/"))
    tiers = {int(t[1:]) for t in a.tiers.split(",")}
    engine, SP = load_engine(JUDGE["repo"], JUDGE["revision"], a.local_dir, a.dry_run, tok,
                             CoTFormat.HARMONY_ANALYSIS, "yes", "judge")
    out = a.out / "judge" / f"shard{shard}.jsonl"
    raw = a.out / "_raw" / "judge" / f"shard{shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)

    def decode(j, r):
        text = decode_rollout(tok, r["gen_ids"], CoTFormat.HARMONY_ANALYSIS) if r["gen_ids"] else ""
        _, final = split_rollout(text, CoTFormat.HARMONY_ANALYSIS) if text else ("", "")
        final = final if r["status"] == "finished" and final else None
        short = final.strip().lower() if final is not None and len(final) <= 20 else None
        return ({"n_tokens": r["n_tokens"], "finish_reason": r["finish_reason"],
                 "status": r["status"], "verdict": short,
                 "verdict_chars": None if final is None else len(final),
                 **credits(final, j["weight"])}, {"reply": text})

    rec = run_chunks(engine, SP, judge_jobs(units, items, tok, cfg, shard, n_shards, tiers),
                     out, raw, decode, deadline=a.deadline,
                     meta_keys=("job_id", "unit", "tier", "pass", "judge", "criterion_id",
                                "weight"))
    rec.update({"shard": a.shard, "tiers": sorted(tiers), "calls_by_tier": calls_by_tier(units, items),
                "validate": a.validate, "pod_version": POD_VERSION})
    (a.out / "judge" / f"shard{shard}.summary.json").write_text(json.dumps(rec, indent=1))
    return rec


def cmd_project(a) -> dict:
    """From VALIDATE's summaries: the full session's calls by tier and the measured rates."""
    items = _items()
    v = a.out
    gen_s = {m: json.loads((v / "gen" / f"{m}.summary.json").read_text()) for m in GEN}
    judge_s = json.loads((v / "judge" / "shard0.summary.json").read_text())
    # the full session's generation per model: (K + fork share) x 500, at the measured tok/s
    full_gens = {m: len(items) * K + sum(in_ladder_subset(it.idx) for it in items) for m in GEN}
    gen_wall_h = max(full_gens[m] * gen_s[m]["decoded_tokens"] / max(gen_s[m]["jobs_run"], 1)
                     / max(gen_s[m]["tokens_per_s"], 1e-9) / 3600 for m in GEN)
    fake = {}
    for it in items:
        for m in GEN:
            for k in range(K):
                fake[f"{m}/default/{it.scenario_id}/k{k}"] = {"trace": "", "response": ""}
            if in_ladder_subset(it.idx):
                fake[f"{m}/fork/{it.scenario_id}/k0"] = {"trace": "", "response": ""}
    calls = calls_by_tier(judge_units(items, fake, validate=False), items)
    measured = {"gen_wall_h_full": gen_wall_h, "setup_h": a.setup_hours, "margin_h": 0.25,
                "judge_calls_per_s": judge_s["calls_per_s"],
                "judge_tokens_per_call": judge_s["decoded_tokens"] / max(judge_s["jobs_run"], 1),
                "judge_tokens_per_s": judge_s["tokens_per_s"]}
    rec = {"measured": measured, "calls_by_tier": calls,
           **project(measured, calls, a.envelope_h, a.validate_hours)}
    (v / "projection.json").write_text(json.dumps(rec, indent=1))
    return rec


def cmd_mb_rider(a) -> dict:
    """MB Phase 2 (MB_PLAN Amendments 3-4): GPT-OSS-20B letter mass at the dose-0 position, both
    orders, effort of record medium, single-row readouts (ST-A3)."""
    import kdg_harmony as kh
    import mb_g1_run as mb
    from deepsteer.choice.readout import label_token_sets, letter_mass, next_token_logprobs
    from deepsteer.datasets.moralbench import load_moralbench
    spec = GEN["gpt_oss_20b"]
    cfg = harmony_cfg()
    system, items = load_moralbench()
    if a.validate:
        items = items[:2]
    tok = _tok(spec["repo"], spec["revision"], a.local_dir)
    if a.dry_run:
        model = None
    else:
        from deepsteer.core.model_interface import WhiteBoxModel
        model = WhiteBoxModel(str(a.local_dir) if a.local_dir else spec["repo"],
                              revision=None if a.local_dir else spec["revision"], device="cuda").model
    sets = label_token_sets(tok, "AB")
    rows = []
    for order in ("original", "counterbalanced"):
        for it in items:
            shown = mb.counterbalance(it) if order == "counterbalanced" else it
            rendered = kh.render(tok, [{"role": "system", "content": system},
                                       {"role": "user", "content": shown.prompt}], cfg,
                                 cfg.reasoning_level)
            ids = tok(rendered + cfg.prefill_primary, add_special_tokens=False)["input_ids"]
            if model is None:
                import numpy as np
                lp = np.log(np.full((1, len(tok)), 1.0 / len(tok)))
            else:
                lp = next_token_logprobs(model, [ids], pad_id=tok.pad_token_id or 0, batch_size=1)
            dist, mass = letter_mass(lp, sets)
            rows.append({"set": it.set_name, "item_id": it.item_id, "order": order,
                         "p_A": float(dist[0][0]), "p_B": float(dist[0][1]),
                         "label_mass": float(mass[0]), "prompt_sha256": sha(rendered)})
    out = a.out / "mb_phase2" / "gpt_oss_20b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "r2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"model": spec["repo"], "revision": spec["revision"], "effort": cfg.reasoning_level,
            "position": "dose-0 (prefill_primary)", "readout": "single-row letter mass, both orders",
            "n_rows": len(rows), "validate": a.validate}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["generate", "judge", "project", "mb-rider"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(GEN))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tiers", default="T1,T2,T3,T4")
    ap.add_argument("--local-dir", type=Path, default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--envelope-h", type=float, default=8.0)
    ap.add_argument("--validate-hours", type=float, default=0.0)
    ap.add_argument("--setup-hours", type=float, default=0.5)
    a = ap.parse_args()
    rec = {"generate": cmd_generate, "judge": cmd_judge, "project": cmd_project,
           "mb-rider": cmd_mb_rider}[a.cmd](a)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("levels",)}, default=str)[:1500])


if __name__ == "__main__":
    main()
