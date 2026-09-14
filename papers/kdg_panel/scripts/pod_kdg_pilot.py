#!/usr/bin/env python3
"""KDG pilot pod driver (KDG_PANEL_SPEC §8 step 4, §11).

    python3 papers/kdg_panel/scripts/pod_kdg_pilot.py --dry-run --out /tmp/kdg_dry      # no model
    python3 papers/kdg_panel/scripts/pod_kdg_pilot.py --out papers/kdg_panel/outputs/pilot
    python3 papers/kdg_panel/scripts/pod_kdg_pilot.py --models olmo3_instruct --units d_chat_dose0
    python3 papers/kdg_panel/scripts/pod_kdg_pilot.py --verify-manifest --out <out>
    python3 papers/kdg_panel/scripts/pod_kdg_pilot.py --units d_chat_dose1,d_chat_dose2 \\
        --scenario-ids-file <out>/SCREEN_ids.json        # dose arms, screened scenarios only

Order (compute-ordering: batch by loaded model): OLMo-3-Instruct first (every chat cell, then its
raw cells), then OLMo-3 base (raw cells only). Per-unit artifacts + manifest checkpoint after
every unit. This driver computes and saves; verdict rules run zero-GPU in ``analyze_pilot.py``.
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
    KDG2_UNITS_INSTRUCT,
    PILOT_UNITS_BASE,
    PILOT_UNITS_INSTRUCT,
    UNITS,
    Ctx,
    Manifest,
    ModelWrapper,
    StubModel,
    verify_manifest,
)

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

KDG_DIR = REPO / "papers" / "kdg_panel"
DEFAULT_OUT = KDG_DIR / "outputs" / "pilot"
MODEL_ORDER = ("olmo3_instruct", "olmo3_base")


def _resolved_commit(model) -> str | None:
    h = getattr(getattr(getattr(model, "model", None), "config", None), "_commit_hash", None)
    return str(h) if h else None


def run(
    out: Path,
    dry: bool,
    models: list[str] | None,
    units: list[str] | None,
    scenario_files: list[Path],
    scenario_ids: set[str] | None = None,
    profile: str = "pilot",
) -> Path:
    cfg = yaml.safe_load((KDG_DIR / "models.yaml").read_text())
    scenarios, metas = load_scenario_dir(scenario_files)
    if scenario_ids is not None:
        scenarios = [s for s in scenarios if s.id in scenario_ids]
    if not scenarios:
        raise SystemExit(
            "no scenarios loaded (empty scenario set: refuse to start a pod on nothing)"
        )
    if dry:
        scenarios = (
            scenarios[:6]
            + [s for s in scenarios if s.family == "F2"][:1]
            + [s for s in scenarios if s.family == "F3"][:1]
        )
    out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(out, dry, metas)
    ro = cfg["rollouts"]
    for key in MODEL_ORDER:
        if models and key not in models:
            continue
        spec = cfg["tier1"][key]
        inst_units = KDG2_UNITS_INSTRUCT if profile == "kdg2" else PILOT_UNITS_INSTRUCT
        default_units = inst_units if spec["kind"] == "instruct" else PILOT_UNITS_BASE
        wanted = [u for u in (units or default_units) if spec["kind"] in UNITS[u][0]]
        if not wanted:
            continue
        print(f"==== {key} ({spec['repo']}) units={wanted} n_scen={len(scenarios)}", flush=True)
        model = StubModel(spec["kind"]) if dry else ModelWrapper(spec["repo"], spec.get("revision"))
        try:
            manifest.data["loads"].append(
                {
                    "key": key,
                    "repo": spec["repo"],
                    "revision_requested": spec.get("revision") or "main",
                    "commit_hash": "dry-run" if dry else _resolved_commit(model.wb),
                    "chat_template_sha256": model.chat_template_sha,
                    "vocab": model.vocab,
                    "dry_run": dry,
                }
            )
            ctx = Ctx(
                key,
                spec["kind"],
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
                t0 = time.time()
                try:
                    UNITS[u][1](ctx, scenarios)
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
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", default=None, help="comma list from models.yaml tier1")
    ap.add_argument("--units", default=None, help="comma list of unit names")
    ap.add_argument(
        "--scenarios",
        nargs="*",
        type=Path,
        default=None,
        help="scenario JSON files (default: papers/kdg_panel/data/pilot_scenarios_*.json)",
    )
    ap.add_argument(
        "--scenario-ids-file", type=Path, default=None, help="JSON list of ids (screened set)"
    )
    ap.add_argument("--verify-manifest", action="store_true")
    ap.add_argument(
        "--profile",
        choices=["pilot", "kdg2"],
        default="pilot",
        help="kdg2 = A13 four-frame J cells (needs 3 paraphrases per scenario)",
    )
    a = ap.parse_args()
    if a.verify_manifest:
        bad = verify_manifest(a.out / "manifest_kdg.json")
        print("\n".join(bad) if bad else "manifest OK")
        return 1 if bad else 0
    files = a.scenarios or sorted((KDG_DIR / "data").glob("pilot_scenarios_*.json"))
    if not files:
        raise SystemExit("no scenario files found under papers/kdg_panel/data")
    ids = set(json.loads(a.scenario_ids_file.read_text())) if a.scenario_ids_file else None
    p = run(
        a.out,
        a.dry_run,
        a.models.split(",") if a.models else None,
        a.units.split(",") if a.units else None,
        files,
        ids,
        profile=a.profile,
    )
    print(f"manifest: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
