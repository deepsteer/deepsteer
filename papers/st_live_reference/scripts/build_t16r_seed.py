#!/usr/bin/env python3
"""The t16r pod's seed (ST_PLAN Amendment 49): T1.6's `none` and base-DPO reads and phases copied
into ``outputs/t16r/muse_glimmer_30b``, so care-DPO2's gate pairs with the rows T1.6's care-DPO
paired with (same engine, seeds and items) and its analysis reads care-DPO2 against the same
base-DPO; a2f's forced-choice reads under ``t16r/forced`` (the launcher pulls only
``outputs/t16r``); then ``runpod/sync_t16r.txt``, every file the pod needs. Local; no model.

    python3 papers/st_live_reference/scripts/build_t16r_seed.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ST = HERE.parent
T16 = ST / "outputs" / "t16" / "muse_glimmer_30b"
T16R = ST / "outputs" / "t16r" / "muse_glimmer_30b"
A2F = ST / "outputs" / "a2f" / "muse_glimmer_30b"
FORCED = ST / "outputs" / "t16r" / "forced" / "muse_glimmer_30b"  # under the pulled OUT
PHASES = ("rr_gen", "base_vllm", "base_hf", "train_base", "hf_arm_base", "vllm_arm_base")
FILES = (["gen/h1_none.jsonl", "gen/h1_base.jsonl", "gen/rr_rollouts.jsonl", "cap/none.jsonl",
          "cap/base.jsonl"]
         + [f"parity/{f}.npy" for f in ("hf_none", "hf_base", "hf_base_merged", "vllm_none",
                                        "vllm_base")]
         + [f"vllm/{a}.{x}" for a in ("none", "base")
            for x in ("jsonl", "jobs.jsonl", "engine.json", "jsonl.summary.json")])
DIRS = [f"{k}/{a}" for k in ("dose0", "spec", "panel") for a in ("none", "base")]
OTHER = (["outputs/st2mr/muse_glimmer_30b/gen/arm_care_03.jsonl",
          "outputs/st2mr/muse_glimmer_30b/gen/arm_none.jsonl",
          "outputs/t16/muse_glimmer_30b/t16_record.json"]
         + [f"outputs/t16/muse_glimmer_30b/parity/hf_{a}.npy" for a in ("none", "care", "base")]
         + [f"outputs/t16/muse_glimmer_30b/adapters/{a}/{f}" for a in ("care", "base")
            for f in ("adapter_config.json", "adapter_model.safetensors", "st_adapter.json")])
FORCED_FILES = (["a2g_record.json", "parity/vllm_none.npy"]
                + [f"gen/cellA2_{a}.jsonl" for a in ("none", "care", "base")])


def main() -> None:
    T16R.mkdir(parents=True, exist_ok=True)
    rec = json.loads((T16 / "t16_record.json").read_text())
    seed = {"model": rec["model"], "phases": {k: rec["phases"][k] for k in PHASES},
            "seeded_from": "outputs/t16 (T1.6's none and base-DPO, Amendment 49)"}
    (T16R / "t16_record.json").write_text(json.dumps(seed, indent=1))
    copied = ["t16_record.json"]
    for f in FILES:
        (T16R / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(T16 / f, T16R / f)
        copied.append(f)
    for d in DIRS:
        shutil.copytree(T16 / d, T16R / d, dirs_exist_ok=True)
        copied += [str(p.relative_to(T16R)) for p in sorted((T16R / d).rglob("*")) if p.is_file()]
    for f in FORCED_FILES:  # a2f's forced-choice reads, so care-DPO2's arm joins them in place
        (FORCED / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(A2F / f, FORCED / f)
    sync = ([f"outputs/t16r/muse_glimmer_30b/{f}" for f in copied]
            + [f"outputs/t16r/forced/muse_glimmer_30b/{f}" for f in FORCED_FILES] + OTHER)
    for f in sync:
        if not (ST / f).exists():
            raise SystemExit(f"missing: {f}")
    (ST / "runpod" / "sync_t16r.txt").write_text("\n".join(sync) + "\n")
    print(f">> seeded {len(copied)} files into {T16R.relative_to(ST)}; sync list {len(sync)} files")


if __name__ == "__main__":
    main()
