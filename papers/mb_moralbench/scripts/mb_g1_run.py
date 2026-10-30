"""MB-G1 generation and readout on the M4 (MB_PLAN v0.5 §2-5).

Per model: three cells on all 88 items, 32 seeded rollouts each (seed = crc32 of ``set/item``):
``t07`` (T = 0.7, of record), ``t10`` (T = 1.0, the registered fork) and ``cb07`` (the
counterbalanced cell at T = 0.7: comparative options swapped, binary labels flipped, keys swapped to
match), plus R2 (letter mass at the answer position) for both orders.

Tracked (``outputs/g1/<model>/``): parsed letters (R0 of record, R0 unstripped, R1 ladder), token
counts, prompt sha256, R2 masses, run metadata. No per-item scores (§9.2), no text. Untracked
(``outputs_raw/g1/<model>/``): replies and rendered prompts. Resumable: items already written for
a cell are skipped.

Usage:
    HF_HUB_OFFLINE=1 python papers/mb_moralbench/scripts/mb_g1_run.py --model zephyr
    python papers/mb_moralbench/scripts/mb_g1_run.py --model zephyr --dry-run --out /tmp/x
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from deepsteer.choice.parse import CHOICE_PARSER_VERSION, first_char, ladder  # noqa: E402
from deepsteer.choice.readout import (  # noqa: E402
    READOUT_VERSION,
    label_token_sets,
    letter_mass,
    next_token_logprobs,
)
from deepsteer.core.chat import (  # noqa: E402
    SystemRolePolicy,
    build_messages,
    encode_rendered,
    render_chat,
    template_sha256,
)
from deepsteer.core.sampling import SAMPLER_VERSION, item_seed, sample_rollouts  # noqa: E402
from deepsteer.datasets.moralbench import (  # noqa: E402
    MANIFEST_PATH,
    MORALBENCH_COMMIT,
    MoralBenchItem,
    load_moralbench,
)

RUN_VERSION = "1.1.0"  # 1.1: G2 pod (llama2_70b, device_map, --out-sub, --raw-dir, deadline)
LABELS = "AB"
N_ROLLOUTS = 32
MAX_NEW_TOKENS = 128
MODELS = {
    "zephyr": {"repo": "HuggingFaceH4/zephyr-7b-beta",
               "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
               "policy": SystemRolePolicy.NATIVE, "license": "MIT", "paper_row": "Zephyr"},
    "gemma": {"repo": "google/gemma-1.1-7b-it",
              "revision": "065a528791af6f57f013e8e42b7276992b45ef71",
              "policy": SystemRolePolicy.PREPEND, "license": "Gemma Terms of Use",
              "paper_row": "Gemma-1.1"},
    # MB_PLAN v0.6 Amendment 1: diagnosis cells for the Zephyr row (run with --cells t07)
    "zephyr_alpha": {"repo": "HuggingFaceH4/zephyr-7b-alpha",
                     "revision": "014792bbb59d04ced3b5a9b8b4dfc926655d958f",
                     "policy": SystemRolePolicy.NATIVE, "license": "MIT", "paper_row": "Zephyr"},
    "zephyr_prepend": {"repo": "HuggingFaceH4/zephyr-7b-beta",
                       "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
                       "policy": SystemRolePolicy.PREPEND, "license": "MIT", "paper_row": "Zephyr"},
    "zephyr_nosys": {"repo": "HuggingFaceH4/zephyr-7b-beta",
                     "revision": "892b3d7a7b1cf10c7a701c60881cd93df615734c",
                     "policy": SystemRolePolicy.DROP, "license": "MIT", "paper_row": "Zephyr"},
    "zephyr_gemma": {"repo": "HuggingFaceH4/zephyr-7b-gemma-v0.1",
                     "revision": "03b3427d0ed07d2e0f86c0a7e53d82d4beef9540",
                     "policy": SystemRolePolicy.NATIVE, "license": "Gemma Terms of Use",
                     "paper_row": "Zephyr"},
    # MB_PLAN v0.6 Amendment 2: the G2 pod row (2 x A100-80GB, HF across both GPUs)
    "llama2_70b": {"repo": "meta-llama/Llama-2-70b-chat-hf",
                   "revision": "e9149a12809580e8602995856f8098ce973d1080",
                   "policy": SystemRolePolicy.NATIVE, "license": "Llama 2 Community License",
                   "paper_row": "LLaMA-2", "device_map": "auto"},
}
CELLS = {"t07": (0.7, "original"), "t10": (1.0, "original"), "cb07": (0.7, "counterbalanced")}
BINARY_SETS = ("MFQ_30", "6_concepts")
_BINARY_LINE = re.compile(r"^(A\.\s*)(.*?)(\s+B\.\s*)(.*?)(\s*)$")
_OPTION = re.compile(r"^(\s*([AB])\.\s*)(.*)$")


def counterbalance(item: MoralBenchItem) -> MoralBenchItem:
    """The item with its two options exchanged and its key swapped to match (MB_PLAN v0.5 §4).

    Binary items: the one-line ``A. <x> B. <y>`` becomes ``A. <y> B. <x>``. Comparative items: the
    contents of the ``A.`` and ``B.`` lines are exchanged, each line keeping its own prefix.
    """
    lines = item.prompt.split("\n")
    if item.set_name in BINARY_SETS:
        hits = [i for i, ln in enumerate(lines) if _BINARY_LINE.match(ln)]
        if len(hits) != 1:
            raise ValueError(f"{item.qualified_id}: expected one option line, found {len(hits)}")
        g = _BINARY_LINE.match(lines[hits[0]]).groups()
        lines[hits[0]] = g[0] + g[3] + g[2] + g[1] + g[4]
    else:
        pos = {m.group(2): (i, m) for i, ln in enumerate(lines) for m in [_OPTION.match(ln)] if m}
        if set(pos) != {"A", "B"}:
            raise ValueError(f"{item.qualified_id}: expected one A line and one B line")
        (ia, ma), (ib, mb) = pos["A"], pos["B"]
        lines[ia], lines[ib] = ma.group(1) + mb.group(3), mb.group(1) + ma.group(3)
    return replace(item, prompt="\n".join(lines), key={"A": item.key["B"], "B": item.key["A"]})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001  (a synced copy has no .git)
        return "unknown"


def done_items(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {f"{r['set']}/{r['item_id']}" for r in map(json.loads, path.read_text().splitlines())}


def load(args) -> tuple:
    if args.dry_run:
        from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
        spec = MODELS["zephyr"]
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        cfg = LlamaConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                          num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tok))
        torch.manual_seed(0)
        return LlamaForCausalLM(cfg).eval(), tok, "cpu", torch.float32
    spec = MODELS[args.model]
    if spec.get("device_map"):  # a model that spans GPUs (Amendment 2)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["repo"], revision=spec["revision"], dtype=torch.bfloat16,
            device_map=spec["device_map"]).eval()
        return model, tok, spec["device_map"], torch.bfloat16
    from deepsteer.core.model_interface import WhiteBoxModel
    wb = WhiteBoxModel(spec["repo"], revision=spec["revision"], device=args.device,
                       torch_dtype=torch.bfloat16)
    return wb.model, wb.tokenizer, args.device, torch.bfloat16


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--cells", default="t07,t10,cb07")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dry-run", action="store_true", help="tiny random model, 2 items per set")
    ap.add_argument("--out", type=Path, default=REPO / "papers" / "mb_moralbench")
    ap.add_argument("--out-sub", default="g1", help="results folder under outputs/ (g2 on the pod)")
    ap.add_argument("--raw-dir", type=Path, help="raw replies (default outputs_raw/<sub>/<model>)")
    ap.add_argument("--max-items", type=int, default=0,
                    help="first N items only (the VALIDATE timing probe; never a cell of record)")
    ap.add_argument("--deadline-epoch", type=float, default=0.0,
                    help="skip a cell whose projected end passes this time (Amendment 2 bail)")
    args = ap.parse_args()
    spec = MODELS[args.model]
    n_roll = 2 if args.dry_run else N_ROLLOUTS
    out = args.out / "outputs" / args.out_sub / args.model
    raw = args.raw_dir or (args.out / "outputs_raw" / args.out_sub / args.model)
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    system, items = load_moralbench()
    if args.dry_run:
        items = [it for s in ("MFQ_30", "6_concepts", "MFQ_30_compare", "6_concepts_compare")
                 for it in [i for i in items if i.set_name == s][:2]]
    if args.max_items:
        items = items[: args.max_items]
    model, tok, device, dtype = load(args)
    token_sets = label_token_sets(tok, LABELS)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t_start = time.time()

    def prepared(it: MoralBenchItem) -> tuple[list[int], str, bool]:
        rendered = render_chat(tok, build_messages(it.prompt, system, spec["policy"]))
        ids, bos = encode_rendered(tok, rendered)
        return ids, rendered, bos

    timings, bos_seen, skipped, rate = {}, set(), [], None
    for cell in args.cells.split(","):
        temperature, order = CELLS[cell]
        rows_path, raw_path = out / f"{cell}.jsonl", raw / f"{cell}.jsonl"
        skip = done_items(rows_path)
        todo = sum(it.qualified_id not in skip for it in items)
        # bail (Amendment 2): with a deadline, skip a cell whose projected end passes it, using the
        # previous cell's measured seconds per item; cells run in the registered order given
        late = rate is not None and time.time() + todo * rate > args.deadline_epoch
        if args.deadline_epoch and late:
            skipped.append({"cell": cell, "items_left": todo, "s_per_item": round(rate, 2)})
            print(f"[{args.model} {cell}] skipped: projected past the deadline", flush=True)
            continue
        t0 = time.time()
        with open(rows_path, "a") as fr, open(raw_path, "a") as fw:
            for it in items:
                if it.qualified_id in skip:
                    continue
                shown = counterbalance(it) if order == "counterbalanced" else it
                ids, rendered, bos = prepared(shown)
                bos_seen.add(bos)
                seed = item_seed(it.qualified_id)
                gens = sample_rollouts(model, tok, ids, n=n_roll, temperature=temperature,
                                       max_new_tokens=MAX_NEW_TOKENS, seed=seed)
                psha = sha256_text(rendered)
                for k, g in enumerate(gens):
                    text = tok.decode(g, skip_special_tokens=True)
                    r1 = ladder(text, LABELS)
                    fr.write(json.dumps({
                        "set": it.set_name, "item_id": it.item_id, "cell": cell, "order": order,
                        "rollout": k, "seed": seed,
                        "r0": first_char(text, LABELS).label,
                        "r0_unstripped": first_char(text, LABELS, strip=False).label,
                        "r1": r1.label, "r1_method": r1.method,
                        "n_new_tokens": len(g), "hit_cap": len(g) >= MAX_NEW_TOKENS,
                        "prompt_sha256": psha,
                    }) + "\n")
                    fw.write(json.dumps({"set": it.set_name, "item_id": it.item_id,
                                         "rollout": k, "text": text}) + "\n")
                fw.write(json.dumps({"set": it.set_name, "item_id": it.item_id,
                                     "rendered_prompt": rendered}) + "\n")
                fr.flush()
                fw.flush()
                print(f"[{args.model} {cell}] {it.qualified_id} {time.time() - t0:.0f}s",
                      flush=True)
        timings[cell] = round(time.time() - t0, 1)
        if todo:
            rate = timings[cell] / todo

    # R2: letter mass at the answer position, both orders (temperature-free)
    r2_path = out / "r2.jsonl"
    if not r2_path.exists():
        rows = []
        for order in ("original", "counterbalanced"):
            shown = [counterbalance(i) if order == "counterbalanced" else i for i in items]
            preps = [prepared(s) for s in shown]
            lp = next_token_logprobs(model, [p[0] for p in preps], pad_id=pad, batch_size=8)
            dist, total = letter_mass(lp, token_sets)
            for it, p, d, m in zip(items, preps, dist, total):
                rows.append({"set": it.set_name, "item_id": it.item_id, "order": order,
                             "p_A": float(d[0]), "p_B": float(d[1]), "label_mass": float(m),
                             "prompt_sha256": sha256_text(p[1])})
        r2_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    import transformers
    meta = {
        "run_version": RUN_VERSION, "date": dt.datetime.now().isoformat(timespec="seconds"),
        "model_key": args.model, "repo": spec["repo"], "revision": spec["revision"],
        "license": spec["license"], "system_role_policy": spec["policy"].value,
        "prepend_separator": "\\n\\n" if spec["policy"] is SystemRolePolicy.PREPEND else None,
        "template_sha256": template_sha256(tok), "bos_prepended": sorted(bos_seen),
        "label_token_sets": token_sets,
        "label_token_surfaces": [[tok.convert_ids_to_tokens(i) for i in s] for s in token_sets],
        "dtype": str(dtype), "device": device, "dry_run": args.dry_run,
        "max_items": args.max_items or None,
        "n_rollouts": n_roll, "max_new_tokens": MAX_NEW_TOKENS, "cells": CELLS,
        "seed_rule": "crc32('<set>/<item_id>') mod 2**31; all rollouts in one seeded call",
        "sampler_version": SAMPLER_VERSION, "readout_version": READOUT_VERSION,
        "parser_version": CHOICE_PARSER_VERSION,
        "decode": "new tokens only, skip_special_tokens=True; R0 of record strips leading space",
        "moralbench_commit": MORALBENCH_COMMIT,
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "git_commit": git_commit(), "timings_s": timings, "skipped_cells": skipped,
        "out_sub": args.out_sub, "deadline_epoch": args.deadline_epoch or None,
        "wall_s": round(time.time() - t_start, 1),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    mpath = out / "run_meta.json"
    prev = json.loads(mpath.read_text()) if mpath.exists() else {}
    prev.setdefault("runs", []).append(meta)
    mpath.write_text(json.dumps(prev, indent=1) + "\n")


if __name__ == "__main__":
    main()
