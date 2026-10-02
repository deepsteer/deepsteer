#!/usr/bin/env python3
"""KDG Phase 1 pod driver (papers/KDG_PHASE1_SPEC.md §4–§5). Loads models in the order given,
runs the requested units on each, saves per-unit artifacts plus a manifest.

    # Session A, final model, C1 on the enlarged union (12 forward-pass cells)
    python3 papers/kdg_panel/scripts/pod_kdg_phase1.py --models olmo3_instruct --units C1 \
        --out papers/kdg_panel/outputs/p1a/final_c1
    # Session A, stage checkpoints: raw cells + neutral letter-chat cells
    python3 papers/kdg_panel/scripts/pod_kdg_phase1.py --models olmo3_sft,olmo3_dpo \
        --units RAW,C3CHAT --out papers/kdg_panel/outputs/p1a/stages
    python3 papers/kdg_panel/scripts/pod_kdg_phase1.py --dry-run --models olmo3_instruct,olmo3_sft \
        --units RAW,C1 --out /tmp/p1dry

Unit groups: RAW, C1, C3CHAT, DOSE, KDG2 (the KDG-2 instruct ladder), VALIDATE; or unit names.
Registry: models.yaml tier1 + tier2 + phase1. Stage checkpoints with ``rendered_equals`` get the
rendered-prompt identity check before any chat unit; a mismatch skips their chat units as a fork
(raw units still run). A resolved commit different from the registry revision is logged in the
manifest load record (spec §7), never silently accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from kdg_pod_lib import (  # noqa: E402
    C1_UNITS,
    C3_CHAT_UNITS,
    DOSE_BF_UNITS,
    DOSE_CTRL_UNITS,
    DOSE_LONG_UNITS,
    DOSE_UNITS,
    KDG2_UNITS_INSTRUCT,
    P2_PILOT_UNITS,
    RAW_UNITS,
    TSN_LEN_UNITS,
    TSN_ROT_UNITS,
    TSN_UNITS,
    UNITS,
    Ctx,
    Manifest,
    ModelWrapper,
    StubModel,
    reference_renderer,
    rendered_identity_mismatches,
    verify_manifest,
)

from deepsteer.kdg.phase2 import load_expanded  # noqa: E402
from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

KDG_DIR = REPO / "papers" / "kdg_panel"
GROUPS = {
    "RAW": RAW_UNITS,
    "C1": C1_UNITS,
    "C3CHAT": C3_CHAT_UNITS,
    "DOSE": DOSE_UNITS,
    "DOSE_BF": DOSE_BF_UNITS,
    "DOSE_CTRL": DOSE_CTRL_UNITS,
    "DOSE_LONG": DOSE_LONG_UNITS,
    "KDG2": KDG2_UNITS_INSTRUCT,
    "VALIDATE": ("validate_forward_matches_generate",),
    # Phase 2 (KDG_F6_F8_SPEC.md §10): pilot keystone on the expanded F6–F8 items; the
    # turns-since-norm rider always runs on the Phase 1 scenario set (--scenarios/--scenario-ids)
    "P2PILOT": P2_PILOT_UNITS,
    "TSN": TSN_UNITS,
    "TSN_ROT": TSN_ROT_UNITS,  # P2-A4a
    "TSN_LEN": TSN_LEN_UNITS,  # P2-A4b
}


def registry() -> tuple[dict, dict]:
    cfg = yaml.safe_load((KDG_DIR / "models.yaml").read_text())
    reg: dict[str, dict] = {}
    for section in ("tier1", "tier2", "phase1"):
        for k, v in (cfg.get(section) or {}).items():
            if k in reg:
                raise SystemExit(f"models.yaml: duplicate key {k!r} across sections")
            reg[k] = v
    return cfg, reg


def expand_units(spec: list[str]) -> list[str]:
    out: list[str] = []
    for u in spec:
        for name in GROUPS.get(u, (u,)):
            if name not in UNITS:
                raise SystemExit(f"unknown unit {name!r}")
            if name not in out:
                out.append(name)
    return out


def _resolved_commit(model) -> str | None:
    h = getattr(getattr(getattr(model, "model", None), "config", None), "_commit_hash", None)
    return str(h) if h else None


def run(
    out: Path,
    dry: bool,
    models: list[str],
    units: list[str],
    scenario_files: list[Path],
    scenario_ids: set[str] | None = None,
    item_files: list[Path] | None = None,
) -> Path:
    cfg, reg = registry()
    wanted_all = expand_units(units)
    scenarios, metas = load_scenario_dir(scenario_files)
    if scenario_ids is not None:
        scenarios = [s for s in scenarios if s.id in scenario_ids]
    expanded: list = []
    if item_files:
        expanded, p2_metas = load_expanded(item_files)
        metas = metas + p2_metas
    # which set each unit reads: TSN always the Phase 1 set; everything else the Phase 2 items
    # when given, else the Phase 1 set
    tsn_all = set(TSN_UNITS) | set(TSN_ROT_UNITS) | set(TSN_LEN_UNITS)
    needs_classic = any(u in tsn_all for u in wanted_all) or not item_files
    for name, S, needed in (
        ("Phase 1", scenarios, needs_classic),
        ("Phase 2", expanded, bool(item_files)),
    ):
        if needed and not S:
            raise SystemExit(
                f"no {name} scenarios loaded (empty scenario set: refuse to start a pod on nothing)"
            )
    if dry:
        scenarios = scenarios[:6] + [s for s in scenarios if s.family == "F3"][:1]
        first: dict[str, str | None] = {}
        for s in expanded:  # one whole item (all its levels) per family
            first.setdefault(s.family, s.item_id)
        expanded = [s for s in expanded if s.item_id == first[s.family]]

    def unit_set(u: str) -> list:
        return expanded if (item_files and u not in tsn_all) else scenarios

    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out, dry, metas)
    manifest.data["phase1"] = {"spec": "KDG_PHASE1_SPEC.md", "models": models, "units": wanted_all}
    if item_files:
        manifest.data["phase2"] = {
            "spec": "KDG_F6_F8_SPEC.md",
            "items": [str(p) for p in item_files],
        }
    ro = cfg["rollouts"]
    for key in models:
        if key not in reg:
            raise SystemExit(f"model key {key!r} not in models.yaml (tier1/tier2/phase1)")
        spec = reg[key]
        kind = spec["kind"]
        wanted = [u for u in wanted_all if kind in UNITS[u][0]]
        if spec.get("raw_only"):
            wanted = [u for u in wanted if u in RAW_UNITS]
        if not wanted:
            print(f"==== {key}: no applicable units, skipped", flush=True)
            continue
        print(
            f"==== {key} ({spec['repo']}) units={wanted} n_p1={len(scenarios)} "
            f"n_p2={len(expanded)}",
            flush=True,
        )
        model = StubModel(kind) if dry else ModelWrapper(spec["repo"], spec.get("revision"))
        try:
            resolved = "dry-run" if dry else _resolved_commit(model.wb)
            requested = spec.get("revision")
            load = {
                "key": key,
                "repo": spec["repo"],
                "revision_requested": requested or "main",
                "commit_hash": resolved,
                "revision_match": None if (dry or not requested) else (resolved == requested),
                "chat_template_sha256": model.chat_template_sha,
                "template_sha256_registry": spec.get("template_sha256"),
                "vocab": model.vocab,
                "dry_run": dry,
            }
            chat_forked = False
            ref_key = spec.get("rendered_equals")
            if ref_key and any(not u.startswith(("d_raw", "j_raw")) for u in wanted):
                ref = reg[ref_key]
                render = (
                    model.render_chat
                    if dry
                    else reference_renderer(ref["repo"], ref.get("revision"))
                )
                # G7 (KDG_F6_F8_SPEC §8): every scheduled chat unit's own message shapes,
                # system and prefilled multi-turn included, on the set that unit reads
                chat_units = [u for u in wanted if u not in RAW_UNITS]
                bad = []
                for S in (expanded, scenarios):
                    us = [u for u in chat_units if unit_set(u) is S]
                    if us:
                        bad += rendered_identity_mismatches(model, render, S, units=us)
                load["rendered_identity"] = {
                    "reference": ref_key,
                    "units": chat_units,
                    "mismatches": bad,
                }
                chat_forked = bool(bad)
            manifest.data["loads"].append(load)
            ctx = Ctx(
                key,
                kind,
                model,
                out / key,
                manifest,
                dry,
                n_d=ro["d_chat"],
                n_j=ro["j_stated_sampled"],
                n_dose=ro["dose_arms"],
                n_raw_perm=cfg["readout"]["raw_permutations"],
                temperature=ro["temperature"],
            )
            for u in wanted:
                if chat_forked and u not in RAW_UNITS:
                    manifest.status(
                        f"{key}/{u}", "skipped_fork", "rendered prompt differs from reference"
                    )
                    continue
                t0 = time.time()
                try:
                    UNITS[u][1](ctx, unit_set(u))
                    manifest.status(f"{key}/{u}", "ok", f"{time.time() - t0:.1f}s")
                except Exception as e:  # one failed unit never kills the pod; it is recorded
                    manifest.status(f"{key}/{u}", "failed", f"{type(e).__name__}: {e}")
                    traceback.print_exc()
                manifest.write()
        finally:
            model.release()
    return manifest.write()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="", help="comma list of registry keys, in load order")
    ap.add_argument("--units", default="", help="comma list of unit names or groups")
    ap.add_argument("--scenarios", nargs="*", type=Path, default=None)
    ap.add_argument("--scenario-ids-file", type=Path, default=None)
    ap.add_argument(
        "--items", nargs="*", type=Path, default=None, help="Phase 2 item files (F6–F8)"
    )
    ap.add_argument("--verify-manifest", action="store_true")
    a = ap.parse_args()
    if a.verify_manifest:
        bad = verify_manifest(a.out / "manifest_kdg.json")
        print("\n".join(bad) if bad else "manifest OK")
        return 1 if bad else 0
    if not a.models or not a.units:
        raise SystemExit("--models and --units are required")
    files = a.scenarios or sorted((KDG_DIR / "data").glob("*_scenarios_*.json"))
    if not files:
        raise SystemExit("no scenario files found under papers/kdg_panel/data")
    ids = None
    if a.scenario_ids_file:
        raw = json.loads(a.scenario_ids_file.read_text())
        ids = set(raw["ids"] if isinstance(raw, dict) else raw)
    p = run(a.out, a.dry_run, a.models.split(","), a.units.split(","), files, ids, a.items)
    print(f"manifest: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
