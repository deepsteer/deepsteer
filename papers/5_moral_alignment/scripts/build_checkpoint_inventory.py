#!/usr/bin/env python3
"""Sprint 0.2: enumerate OLMo-3 7B checkpoints and architecture metadata.

Queries HuggingFace refs for the OLMo-3 7B base + post-training repos and
writes a structured inventory to
``papers/5_moral_alignment/outputs/checkpoint_inventory.json``.

The inventory records:
  * Architecture facts per repo (hybrid sliding/full attention pattern).
  * Every available revision, categorised by training stage.
  * A proposed Sprint 2 sampling grid: a pretraining-emergence trajectory
    (base stage3 anneal), the four stage anchors (base / SFT / DPO /
    Instruct finals), and the RLVR sub-trajectory (Instruct step_NNN).

Reads config.json (small) but never downloads weights.

Usage:
    python papers/5_moral_alignment/scripts/build_checkpoint_inventory.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_refs

REPOS = {
    "base": "allenai/Olmo-3-1025-7B",
    "sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "instruct": "allenai/Olmo-3-7B-Instruct",
}

OUT = Path("papers/5_moral_alignment/outputs/checkpoint_inventory.json")

_ARCH_KEYS = [
    "model_type", "architectures", "num_hidden_layers", "hidden_size",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
    "max_position_embeddings", "sliding_window", "rope_theta",
    "intermediate_size",
]


def _step_num(name: str) -> int:
    """Extract the step number for sorting (the digits after 'step').

    Falls back to the first run of digits. Note 'stage3-step1000' must sort by
    1000, not the '3' in 'stage3', so the step-anchored match comes first.
    """
    m = re.search(r"step[_-]?(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", name)
    return int(m.group()) if m else -1


def _arch(repo: str) -> dict:
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    out = {k: cfg.get(k) for k in _ARCH_KEYS}
    lt = cfg.get("layer_types")
    if lt:
        full = [i for i, t in enumerate(lt) if t == "full_attention"]
        out["full_attention_layers"] = full
        out["n_full_attention"] = len(full)
        out["n_sliding_attention"] = len(lt) - len(full)
        out["attention_pattern"] = "".join(
            "F" if t == "full_attention" else "s" for t in lt
        )
    return out


def _refs(repo: str) -> dict:
    refs = list_repo_refs(repo)
    branches = sorted((b.name for b in refs.branches), key=_step_num)
    tags = sorted((t.name for t in refs.tags), key=_step_num)
    return {"branches": branches, "tags": tags}


def _categorise_base(branches: list[str]) -> dict:
    """Bucket base pretraining branches by stage and pull out clean steps."""
    buckets: dict[str, list[str]] = {}
    for b in branches:
        if b == "main":
            buckets.setdefault("main", []).append(b)
            continue
        m = re.match(r"(stage\d+)", b)
        key = m.group(1) if m else "other"
        buckets.setdefault(key, []).append(b)

    # Clean stageN-stepNNNN (no extra mix suffix) — usable as a trajectory.
    stage3_clean = sorted(
        (b for b in buckets.get("stage3", []) if re.fullmatch(r"stage3-step\d+", b)),
        key=_step_num,
    )
    stage2_clean = sorted(
        (b for b in buckets.get("stage2", []) if re.fullmatch(r"stage2-step\d+", b)),
        key=_step_num,
    )
    return {
        "counts": {k: len(v) for k, v in sorted(buckets.items())},
        "stage3_clean_steps": stage3_clean,
        "stage2_clean_steps": stage2_clean,
    }


def main() -> None:
    inventory: dict = {
        "generated_by": "build_checkpoint_inventory.py",
        "note": (
            "OLMo-3 7B is hybrid attention: 24 sliding-window (4096) + 8 full "
            "layers (every 4th: 3,7,11,15,19,23,27,31). Short-text probes sit "
            "well under the window, so the pattern is functionally inert for "
            "probing; full-attention layers are flagged in layer-wise plots."
        ),
        "repos": {},
    }

    for tag, repo in REPOS.items():
        entry = {"repo": repo, "architecture": _arch(repo), "refs": _refs(repo)}
        branches = entry["refs"]["branches"]
        if tag == "base":
            entry["pretraining"] = _categorise_base(branches)
        else:
            steps = sorted(
                (b for b in branches if b.lower().startswith("step")), key=_step_num
            )
            entry["step_revisions"] = steps
        inventory["repos"][tag] = entry

    # ---- Proposed Sprint 2 grid (per user decision: add pretraining traj) ----
    base = inventory["repos"]["base"]
    instruct_steps = inventory["repos"]["instruct"].get("step_revisions", [])
    grid = {
        "rationale": (
            "Stage anchors give the across-method comparison (base -> SFT -> "
            "DPO -> Instruct). RLVR substeps give within-stage granularity for "
            "the final stage. stage3 anneal gives a pretraining-emergence "
            "trajectory. No intermediate SFT/DPO checkpoints exist on HF."
        ),
        "pretraining_trajectory": [
            {"label": f"olmo3_pretrain_{s.replace('-', '_')}", "repo": REPOS["base"], "revision": s}
            for s in base["pretraining"]["stage3_clean_steps"]
        ],
        "stage_anchors": [
            {"label": "olmo3_base", "repo": REPOS["base"], "revision": "main"},
            {"label": "olmo3_sft_final", "repo": REPOS["sft"], "revision": "main"},
            {"label": "olmo3_dpo_final", "repo": REPOS["dpo"], "revision": "main"},
            {"label": "olmo3_instruct_final", "repo": REPOS["instruct"], "revision": "main"},
        ],
        "rlvr_substeps": [
            {"label": f"olmo3_instruct_{s}", "repo": REPOS["instruct"], "revision": s}
            for s in instruct_steps
        ],
    }
    inventory["proposed_sprint2_grid"] = grid

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(inventory, f, indent=2)

    # ---- console summary ----
    print(f"Wrote {OUT}")
    for tag, repo in REPOS.items():
        e = inventory["repos"][tag]
        a = e["architecture"]
        n_branches = len(e["refs"]["branches"])
        extra = ""
        if tag == "base":
            extra = f" | pretraining buckets={e['pretraining']['counts']}"
        else:
            extra = f" | step revisions={len(e.get('step_revisions', []))}"
        print(f"  {tag:9s} {repo}: {a['model_type']} {a['num_hidden_layers']}L/"
              f"{a['hidden_size']}d, {n_branches} branches{extra}")
    n_pre = len(grid["pretraining_trajectory"])
    n_rlvr = len(grid["rlvr_substeps"])
    print(f"  proposed grid: {n_pre} pretraining + 4 stage anchors + {n_rlvr} RLVR "
          f"= {n_pre + 4 + n_rlvr} model states")


if __name__ == "__main__":
    main()
