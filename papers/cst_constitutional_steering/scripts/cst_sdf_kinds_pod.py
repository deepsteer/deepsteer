#!/usr/bin/env python3
"""The kind-split pod (CST Amendment 32 items 6-10): four SDF adapters from subsets of the existing
corpora, their screen reads, the content gate.

    cst_sdf_kinds_pod.py --stage train_kind --kind dn_stories --out DIR
    cst_sdf_kinds_pod.py --stage parity_hf --tags dn_stories,dn_nostories --out DIR
    cst_sdf_kinds_pod.py --stage read_base | read_kinds --tags a,b | read_plain --tags ... --out DIR
    cst_sdf_kinds_pod.py --stage content --arm a0|a0s --out DIR

**Adapters (item 6), without recovery,** each trained once through its subset of the kept
documents (`outputs/pilot_sdf/corpus`, copied into ``DIR/corpus``), packed as the 1M run packs
them, at the 1M run's learning rate (`pilot_pod_lib.SDF_LR`), seed 0, LoRA as the 1M run:
``dn_stories`` (D-neutral story and steadiness), ``dn_nostories`` (D-neutral essay, commentary,
FAQ), ``dd_stories``, ``dd_nostories`` (D-doc likewise). One pass is ``floor(chunks / 4)``
optimizer steps of 4 packed chunks, a seeded permutation, so no chunk repeats; the record carries
the subset's documents, kept-text tokens and trained tokens.

**Reads (item 7), one session base:** ``read_base`` reads the base on the operator and plain cells
at k 2 on the 458 clear items; ``read_kinds`` reads adapters on the operator cell at k 2 (the
screen read here, flagged in the amendment); ``read_plain`` adds their plain cells if time remains.
Parity per tag as every read (Amendment 29 item 1). **The content gate (item 9):** A0 and A0s at k 4
on `data/content_scenarios_v1.json`'s plain condition (``gen/a0``, ``gen/a0s``). The smoke pass,
parity, the guard cell and its action read come from `cst_sdf_norec_pod.py`, which this module
extends (its ``TAGS`` gain the four kinds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_pilot_sdf_pod as P  # noqa: E402
import cst_s0a_pod as S  # noqa: E402
import cst_sdf_norec_pod as Q  # noqa: E402
import pilot_pod_lib as PL  # noqa: E402

STORIES, NONSTORIES = ("story", "steadiness"), ("essay", "commentary", "faq")
KINDS = {"dn_stories": ("d_neutral", STORIES), "dn_nostories": ("d_neutral", NONSTORIES),
         "dd_stories": ("d_doc", STORIES), "dd_nostories": ("d_doc", NONSTORIES)}
ORDER = ("dn_stories", "dn_nostories", "dd_stories", "dd_nostories")  # item 7's read order
Q.TAGS.update({t: (t, "max") for t in KINDS})  # train_<tag>.json names each adapter
K2_OP, K2_PLAIN = {"panel_operator": 2}, {"panel_plain": 2}
ACCUM = 4


def subset_examples(out: Path, tag: str, tok, dry: bool) -> tuple[list, dict]:
    """The subset's kept documents packed as `cst_pilot_sdf_pod.rung_data` packs a corpus arm."""
    import pilot_train as PT

    arm, kinds = KINDS[tag]
    rows = [r for r in P.corpus_rows(out, arm, dry) if r["kept"] and r["kind"] in kinds]
    seq = 256 if dry else PT.SEQ_LEN
    chunks = PT.pack([tok.encode(r["final"], add_special_tokens=False) for r in rows],
                     tok.eos_token_id, seq)
    meta = {"corpus_arm": arm, "kinds": list(kinds), "documents": len(rows),
            "kept_text_tokens": sum(r["final_tokens"] for r in rows), "packed_chunks": len(chunks),
            "seq_len": seq}
    return [(c, None) for c in chunks], meta


def stage_train_kind(out: Path, tag: str, dry: bool) -> dict:
    """Item 6: one pass through the subset (no chunk repeats), saved at its one dose."""
    import pilot_train as PT

    f = out / f"train_{tag}.json"
    if f.exists():  # a resumed session keeps the adapter of record
        return json.loads(f.read_text())
    path, wb, cfg = PL.load_model(dry)
    tok = S.V._tok(path)  # noqa: SLF001
    ex, meta = subset_examples(out, tag, tok, dry)
    steps = len(ex) // ACCUM
    if steps < 1:
        raise SystemExit(f"BAIL: {tag}: {len(ex)} packed chunks make no optimizer step")
    per_step = meta["seq_len"] * ACCUM
    dose = steps * per_step
    r = PT.train_rung(wb, ex, lambda e: PT.lm_loss(wb, e[0], e[1]), per_step_dose=per_step,
                      doses=(dose,), out_dir=out / "adapters" / tag, lr=PL.SDF_LR, seed=0)
    rec = {"rung": tag, "rule": "Amendment 32 item 6", "lr": PL.SDF_LR, "recovery": None,
           "lora": {k: cfg[k] for k in ("rank", "alpha", "dropout")}, "subset": meta,
           "steps": steps, "trained_tokens": dose,
           "chunks_left_out": len(ex) - steps * ACCUM, "train": r,
           "adapters": dict(r["checkpoints"])}
    Q._w(out, f"train_{tag}", rec)  # noqa: SLF001
    print(f">> train_{tag}:", json.dumps({k: v for k, v in rec.items() if k != "train"}))
    return rec


def stage_content(out: Path, arm: str, dry: bool) -> None:
    """Item 9: the content gate's scenarios at k 4, plain condition, for A0 or A0s."""
    if not S.CONTENT.exists():
        raise SystemExit(f"BAIL: content: {S.CONTENT} is not on the pod")
    S.stage_gen(out, arm, ["content_plain"], dry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("train_kind", "parity_hf", "read_base", "read_kinds", "read_plain",
                             "content"))
    ap.add_argument("--kind", choices=tuple(KINDS))
    ap.add_argument("--tags", default="")
    ap.add_argument("--arm", choices=("a0", "a0s"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t]
    if a.stage == "train_kind":
        stage_train_kind(a.out, a.kind, a.dry_run)
    elif a.stage == "parity_hf":  # here, not in cst_sdf_norec_pod.py: TAGS gain the kinds here
        PL.hf_letters(a.out, Q.items(a.out, tags), a.dry_run)
    elif a.stage == "read_base":
        PL.stage_read(a.out, [], a.dry_run, parts=("panel_operator", "panel_plain"),
                      k_override=K2_OP | K2_PLAIN, base=True)
    elif a.stage == "read_kinds":
        PL.stage_read(a.out, Q.items(a.out, tags), a.dry_run, parts=("panel_operator",),
                      k_override=K2_OP, base=False)
    elif a.stage == "read_plain":
        PL.stage_read(a.out, Q.items(a.out, tags), a.dry_run, parts=("panel_plain",),
                      k_override=K2_PLAIN, base=False)
    else:
        stage_content(a.out, a.arm, a.dry_run)


if __name__ == "__main__":
    main()
