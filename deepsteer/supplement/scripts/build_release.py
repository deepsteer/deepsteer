#!/usr/bin/env python3
"""Build the Zenodo deposit for the FL + MN per-unit arrays (supplement/RELEASE_PLAN.md).

    python3 deepsteer/supplement/scripts/build_release.py [--out outputs/zenodo_v1] [--dry-run]

Stages every array named in RELEASE_PLAN §1 from the (gitignored) run outputs, applies the §2
exclusions (MORABLES-derived files, datasets, smoke/dry/pilot trees, model weights), writes one
deterministic tarball per paper (shared arrays live once, in the FL tarball; the MN manifest
references them by path + sha256), and writes MANIFEST.json / PROVENANCE.md / REGENERATE.md /
LICENSE into the deposit directory. Nothing is uploaded: the DOI is reserved and the files are
published by the author (RELEASE_PLAN §3–4).

The W4 run manifest (papers/d3_decision_anatomy/outputs/w4/manifest_w4.json) is the source of
truth for every W4 file's sha256 and for the HF commit hash of the model that produced it; this
script re-hashes every staged file and refuses to proceed on a mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
W4_MANIFEST = REPO / "papers/d3_decision_anatomy/outputs/w4/manifest_w4.json"

EXCLUDE_SUBSTR = ("morables", "/_smoke/", "/_dry/", "/w4_rerun1/", "/pilot/", "_validate_")
EXCLUDE_SUFFIX = (".png", ".pdf", ".pt", ".pth", ".ckpt", ".safetensors", ".log")

# RELEASE_PLAN §1 groups: (group id, description, source globs relative to REPO, tarball, cited_by)
GROUPS: list[dict] = [
    {"id": "w4", "tarball": "fl",
     "description": "Every per-unit array of the W4 pod (Amendments 14/15; proto-refusal reliability, Llama harm capture, GPT-OSS P_dec reads, P0-P3 rollouts, cross-ablation, PR profiles, reply-inversion margins, OLMo pooled sweep, Qwen read cell).",
     "globs": ["papers/d3_decision_anatomy/outputs/w4/**/*"],
     "cited_by": ["FL §4, §7, §8, App B, App C.8, App D", "MN §2 Table 1, §3.1", "CLAIMS W4-01..10"]},
    {"id": "decision_anatomy", "tarball": "fl",
     "description": "D3 C1 sessions on OLMo-3 and Llama-3.1 (read layer and depth-matched L12): per-head write contributions + channel-matched specificity, per-twin interchange deltas for every cell and rank, channel_act, Vbasis, harm, refusal; one-knob fits; standardized-invariance check; GPT-OSS Tier-1 session.",
     "globs": ["papers/d3_decision_anatomy/outputs/c1_*.npz", "papers/d3_decision_anatomy/outputs/c1_*.json",
               "papers/d3_decision_anatomy/outputs/one_knob_*.json", "papers/d3_decision_anatomy/outputs/standardized_invariance_olmo3.json",
               "papers/d3_decision_anatomy/outputs/patch_stimuli_manifest.json", "papers/d3_decision_anatomy/outputs/tier1_*"],
     "cited_by": ["FL §7, §8, App C, App D", "MN §2.4, §3, §5, §6", "CLAIMS D3-*"]},
    {"id": "d1_geometry", "tarball": "fl",
     "description": "D1 phase-2 per-model extractions (base, instruct, Think, GPT-OSS): act_samples, Moral-Stories per-pair diffs, moral/MFT/persona directions, V_moral, refusal directions per position, calibration ladder results.",
     "globs": ["papers/d1_moral_subspace/outputs/phase2/**/*"],
     "cited_by": ["FL §3, §4, §6, App A, App B", "MN §2.1", "CLAIMS D1-*"]},
    {"id": "d2_decision_coupling", "tarball": "fl",
     "description": "D2 in-format sessions per model (OLMo-3, Llama-3.1, Qwen2.5): decision-site act_samples, chat V_moral, axis per-pair diffs (fables, ethics), Moral-Stories diffs, judgment/refusal decision directions, persona, control contrasts (syntax/register/sentiment/fable-schema), in-format ladder results.",
     "globs": ["papers/d2_decision_coupling/outputs/**/*"],
     "cited_by": ["FL §5, §6, App D", "MN §2.1-2.3", "CLAIMS D2-*"]},
    {"id": "proto_refusal_caches", "tarball": "fl",
     "description": "Paper 5 per-checkpoint proto-refusal directions for the 14 OLMo-3 stage-3 states (the zero-GPU arm of W4 14.1).",
     "globs": ["papers/5_moral_alignment/outputs/measurement/stage3/**/*"],
     "cited_by": ["FL §4 Fig 2b", "CLAIMS W4-01/02"]},
    {"id": "reasoning_p7", "tarball": "fl",
     "description": "Paper 7 GPT-OSS position extraction, trace profile, two-site decomposition, causal ablation, and the Qwen-14B / Llama-8B reply-inversion control JSONs.",
     "globs": ["papers/7_reasoning/outputs/gpt_oss_20b/*", "papers/7_reasoning/outputs/control/*",
               "papers/7_reasoning/outputs/*.json"],
     "cited_by": ["FL §8, App G", "MN §3.1", "CLAIMS P7-*"]},
    {"id": "distilled", "tarball": "fl",
     "description": "The in-repo distilled supplement (figure_data CSVs, cell JSONs, W4 summaries) at the deposit commit.",
     "globs": ["deepsteer/supplement/figure_data/*.csv", "deepsteer/supplement/cells/**/*.json",
               "deepsteer/supplement/MANIFEST.json"],
     "cited_by": ["FL App E", "MN §9"]},
]
# Files the MN cites that are NOT in the FL groups above get their own tarball; everything MN cites
# from the FL groups is referenced by path + sha in the MN section of MANIFEST.json.
MN_ONLY_GLOBS = ["papers/d2_decision_coupling/outputs/*/informat_ladder_*.json",
                 "papers/d2_decision_coupling/outputs/standardized_recompute.json",
                 "papers/d2_decision_coupling/outputs/b3_rotation_compare.json"]
MN_REFERENCED_PREFIXES = ("papers/d3_decision_anatomy/outputs/w4/", "papers/d3_decision_anatomy/outputs/c1_",
                          "papers/d2_decision_coupling/outputs/", "papers/d1_moral_subspace/outputs/phase2/")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(rel: str) -> bool:
    r = "/" + rel
    return any(s in r.lower() for s in EXCLUDE_SUBSTR) or rel.endswith(EXCLUDE_SUFFIX) or "/full/" in r


def collect() -> list[dict]:
    seen: dict[str, dict] = {}
    for g in GROUPS:
        for pat in g["globs"]:
            for p in sorted(REPO.glob(pat)):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(REPO))
                if excluded(rel) or rel in seen:
                    continue
                seen[rel] = {"path": rel, "group": g["id"], "tarball": g["tarball"], "bytes": p.stat().st_size}
    mn_only = []
    for pat in MN_ONLY_GLOBS:
        for p in sorted(REPO.glob(pat)):
            rel = str(p.relative_to(REPO))
            if excluded(rel):
                continue
            if rel in seen:
                seen[rel]["tarball"] = "fl"          # already public via FL; MN references it
            else:
                seen[rel] = {"path": rel, "group": "mn_only", "tarball": "mn", "bytes": p.stat().st_size}
    return list(seen.values())


def w4_index() -> tuple[dict, dict]:
    m = json.loads(W4_MANIFEST.read_text())
    by_path = {a["path"]: a for a in m["artifacts"]}
    loads = {l["key"]: l for l in m["loads"]}
    return by_path, {"loads": loads, "run_id": m["run_id"], "git_commit": m.get("git_commit"), "reruns": m.get("reruns", [])}


def build_tar(entries: list[dict], out: Path, name: str) -> dict:
    """Deterministic tar (sorted, mtime 0, root owner), then zstd -19. Returns tarball record."""
    tar_path = out / f"{name}.tar"
    with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as tf:
        for e in sorted(entries, key=lambda x: x["path"]):
            src = REPO / e["path"]
            ti = tf.gettarinfo(str(src), arcname=e["path"])
            ti.mtime = 0; ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
            with open(src, "rb") as f:
                tf.addfile(ti, f)
    zst = out / f"{name}.tar.zst"
    if zst.exists():
        zst.unlink()
    subprocess.run(["zstd", "-19", "-T0", "--rm", "-q", str(tar_path), "-o", str(zst)], check=True)
    return {"file": zst.name, "bytes": zst.stat().st_size, "sha256": sha256(zst), "n_files": len(entries),
            "bytes_uncompressed": sum(e["bytes"] for e in entries)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs/zenodo_v1"))
    ap.add_argument("--dry-run", action="store_true", help="collect + hash only, no tarballs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    entries = collect()
    w4_by_path, w4_meta = w4_index()
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    mismatches = []
    for e in entries:
        e["sha256"] = sha256(REPO / e["path"])
        e["license"] = "CC-BY-4.0"
        w = w4_by_path.get(e["path"])
        if w:
            if w["sha256"] != e["sha256"]:
                mismatches.append(e["path"])
            e.update({"unit": w.get("unit"), "model": w.get("model"), "source_run": w.get("rerun_id") or w4_meta["run_id"]})
            ld = w4_meta["loads"].get(w.get("model"))
            if ld:
                e["hf_repo"] = ld["repo"]; e["hf_commit"] = ld["commit_hash"]; e["dtype"] = ld.get("dtype")
        else:
            e["source_run"] = "see PROVENANCE.md"
    if mismatches:
        print("REFUSING: staged W4 files disagree with manifest_w4.json sha256:", *mismatches, sep="\n  ")
        return 1
    groups = {g["id"]: g for g in GROUPS}
    manifest = {
        "deposit": "DeepSteer per-unit arrays: what refusal reads (flagship) and instruments before verdicts (methods note)",
        "version": "v1 (W4 run of record, 2026-09-12/13)", "repo_commit": git, "w4_run_id": w4_meta["run_id"],
        "w4_reruns": [{"run_id": r["run_id"], "reason": r["reason"], "units": r["units"]} for r in w4_meta["reruns"]],
        "license": "CC BY 4.0 (arrays and manifests); code under the repository license by pointer",
        "excluded": {"rule": "RELEASE_PLAN §2", "patterns": list(EXCLUDE_SUBSTR) + list(EXCLUDE_SUFFIX) + ["/full/ (datasets)"],
                     "note": "MORABLES-derived caches (CC-BY-NC) are excluded; the headline arrays use Moral Stories, public-domain fable retellings, and ETHICS."},
        "groups": {gid: {"description": g["description"], "cited_by": g["cited_by"], "tarball": g["tarball"]} for gid, g in groups.items()},
        "hf_models": w4_meta["loads"],
        "files": sorted(entries, key=lambda x: x["path"]),
        "mn_references": sorted(e["path"] for e in entries if e["tarball"] == "fl" and e["path"].startswith(MN_REFERENCED_PREFIXES)),
    }
    fl = [e for e in entries if e["tarball"] == "fl"]; mn = [e for e in entries if e["tarball"] == "mn"]
    print(f"staged: {len(fl)} FL files ({sum(e['bytes'] for e in fl)/1e6:.0f} MB), {len(mn)} MN-only files ({sum(e['bytes'] for e in mn)/1e6:.1f} MB)")
    by_group = {}
    for e in entries:
        by_group.setdefault(e["group"], [0, 0]); by_group[e["group"]][0] += 1; by_group[e["group"]][1] += e["bytes"]
    for gid, (n, b) in by_group.items():
        print(f"  {gid:24s} {n:4d} files {b/1e6:8.1f} MB")
    if args.dry_run:
        return 0
    manifest["tarballs"] = {"fl": build_tar(fl, out, "deepsteer_fl_arrays_v1")}
    if mn:
        manifest["tarballs"]["mn"] = build_tar(mn, out, "deepsteer_mn_arrays_v1")
    else:
        manifest["tarballs"]["mn"] = {"note": "every array the methods note cites lives in the FL tarball (shared-arrays-live-once); see mn_references for the path + sha256 list"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=False))
    shutil.copy2(REPO / "deepsteer/supplement/PROVENANCE.md", out / "PROVENANCE.md")
    (out / "LICENSE").write_text("Creative Commons Attribution 4.0 International (CC BY 4.0) applies to every array and manifest in this deposit.\nhttps://creativecommons.org/licenses/by/4.0/\nThe code that produced them is in the DeepSteer repository under its own license (commit " + git + ").\n")
    (out / "REGENERATE.md").write_text(REGENERATE.format(commit=git, run=w4_meta["run_id"]))
    print("wrote", out, "tarballs:", json.dumps(manifest["tarballs"], indent=1))
    return 0


REGENERATE = """# Regeneration recipes (deposit v1, repo commit {commit})

All recipes run from the DeepSteer repository at the commit above with `pip install -e .[all]`.
Model weights are never deposited; each recipe pulls them from Hugging Face at the commit hash
recorded in MANIFEST.json (`hf_models`). Llama-3.1 is gated: accept Meta's license and set HF_TOKEN.

| group | recipe |
|---|---|
| w4 | `scripts/remote_w4.sh` via `papers/d1_moral_subspace/runpod/run_session.sh` (header lists the SYNC_EXTRA inputs); run of record `{run}` + rerun folded with `scripts/pod_w4.py --merge-from`; verify with `scripts/pod_w4.py --verify-manifest` |
| decision_anatomy | `papers/d3_decision_anatomy/scripts/c1_session.py --model <hf repo> --key <olmo3|llama31> --layer <16|12>` (`SWEEP=1`; `STANDARDIZE=1 ROBUSTIFY=zscore` for the standardized rerun); GPT-OSS Tier 1: `papers/d3_decision_anatomy/scripts/gptoss_tier1.py` |
| d1_geometry | `papers/d1_moral_subspace/runpod/phase2_session.sh` (base + instruct chains, G2/G3), `papers/d1_moral_subspace/scripts/phase2_*.py`; reasoning tags via the think/gpt_oss remote scripts in the same directory |
| d2_decision_coupling | `papers/d2_decision_coupling/scripts/informat_ladder.py`, `b1_judgment_direction.py`, `b3_*` control extractions per model key |
| proto_refusal_caches | `papers/5_moral_alignment/scripts/coupling_measurement.py` over the 14 stage-3 checkpoints (`checkpoint_inventory.json`) |
| reasoning_p7 | `papers/7_reasoning/scripts/` position extraction, two-site decomposition, `reply_inversion_control.py` |
| distilled | `python3 deepsteer/supplement/scripts/build.py`; verify with `scripts/verify.py` |

MORABLES-derived caches (excluded, CC-BY-NC): `papers/d1_moral_subspace/scripts/generate_morables.py`
under the MORABLES license, then the D1 phase-2 extraction at layer 16.
"""

if __name__ == "__main__":
    sys.exit(main())
