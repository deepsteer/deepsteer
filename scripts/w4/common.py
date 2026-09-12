"""Run context, panel table, manifest, and pilot-gate helpers for the W4 pod driver.

Every saved array goes through :meth:`Ctx.save` so the manifest carries a sha256 for each file
(deepsteer/supplement style). Every model load goes through :func:`load_record`, which records the
HF commit hash actually resolved (FL App E.4 currently says "default branch"; the manifest closes
that gap) and runs the registry ``assert_matches_model`` check.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
W4_OUT = REPO / "papers" / "d3_decision_anatomy" / "outputs" / "w4"   # gitignored per-unit arrays
SEED = 0

for _p in ("papers/d3_decision_anatomy/scripts", "papers/d2_decision_coupling/scripts",
           "papers/d1_moral_subspace/scripts", "papers/5_moral_alignment/scripts",
           "papers/6_cross_model/scripts", "papers/7_reasoning/scripts"):
    sys.path.insert(0, str(REPO / _p))


@dataclasses.dataclass(frozen=True)
class ModelLoad:
    """One model load in the pre-registered order (Amendment 14 'Model batching')."""

    key: str            # output subdir + manifest key
    repo: str
    revision: str | None
    kind: str           # instruct | think | base | reasoning_moe
    layer: int          # read/patch layer of record
    registry: str       # 'p6' (Paper 6 ModelSpec) | 'p7' (Paper 7 ReasoningModelSpec)
    registry_key: str
    units: tuple[str, ...]
    n_layers: int
    hidden: int


# Pre-registered order: OLMo-3-Instruct -> OLMo-3-Think -> OLMo-3 base -> Llama -> GPT-OSS -> Qwen.
PANEL: tuple[ModelLoad, ...] = (
    ModelLoad("olmo3_instruct", "allenai/Olmo-3-7B-Instruct", None, "instruct", 16, "p6", "olmo3",
              ("15.1", "14.5", "14.6", "14.1_gate"), 32, 4096),
    ModelLoad("olmo3_think", "allenai/Olmo-3-7B-Think", None, "think", 16, "p6", "olmo3",
              ("14.4",), 32, 4096),
    ModelLoad("olmo3_base", "allenai/Olmo-3-1025-7B", None, "base", 16, "p6", "olmo3",
              ("14.1_proto",), 32, 4096),
    ModelLoad("llama31", "meta-llama/Llama-3.1-8B-Instruct", None, "instruct", 12, "p6", "llama31",
              ("14.2", "14.6", "14.6b"), 32, 4096),
    ModelLoad("gpt_oss_20b", "openai/gpt-oss-20b", None, "reasoning_moe", 12, "p7", "gpt_oss_20b",
              ("14.3", "14.4", "14.6"), 24, 2880),
    ModelLoad("qwen25", "Qwen/Qwen2.5-7B-Instruct", None, "instruct", 14, "p6", "qwen25",
              ("15.2", "14.6"), 28, 3584),
)

# papers/MISSING_ARTIFACTS.md entries -> the W4 unit that closes each (checked by the test suite and
# printed by `pod_w4.py --closure-map`). Every entry in that ledger must appear here.
MISSING_ARTIFACTS_CLOSURE: dict[str, str] = {
    "A1 think mft_directions.npz": "olmo3_think/14.4 (MFT 6-foundation directions, raw)",
    "A1 gpt_oss mft_directions.npz": "gpt_oss_20b/14.4 (MFT 6-foundation directions, raw)",
    "A3 Think refusal vectors P0-P3 (.npz)": "olmo3_think/14.4 (refusal_P{0..3}.npz)",
    "A4 instruct fables/ethics per-pair diff arrays": "olmo3_instruct/14.6 (axis_diffs_{fables,ethics}.npz)",
    "Amendment 2 per-position chat act_samples (D2 in-format)": "olmo3_instruct+llama31+qwen25/14.6 (three position classes)",
    "Amendment 2 D1 P0-P3 per-rollout activations": "olmo3_think+gpt_oss_20b/14.4 (p0p3_rollouts.npz)",
    "Amendment 2 mean_content slices for refusal/judgment prompt sets": "olmo3_instruct/14.6 (mean_content_slices.npz)",
    "Amendment 11 severity-twin paired content contrasts (Llama L12)": "llama31/14.2 (severity_contrasts_L12.npz)",
    "Amendment 11 GPT-OSS harmony decision-token act_sample (A5 band-below-null)": "gpt_oss_20b/14.3 (decision_token_sample.npz + band_below_null)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pod syncs have no .git
        return "unknown"


def versions() -> dict:
    out = {"python": platform.python_version()}
    for m in ("numpy", "torch", "transformers"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def resolved_commit_hash(model, repo: str, revision: str | None) -> str | None:
    """The HF commit actually loaded: ``config._commit_hash`` first, then the Hub API."""
    h = getattr(getattr(getattr(model, "model", None), "config", None), "_commit_hash", None)
    if h:
        return str(h)
    try:
        from huggingface_hub import HfApi
        return str(HfApi().model_info(repo, revision=revision).sha)
    except Exception:
        return None


_REGISTRY_FILES = {"p6": ("papers", "6_cross_model", "scripts", "model_registry.py"),
                   "p7": ("papers", "7_reasoning", "scripts", "model_registry.py")}


def _registry_module(which: str):
    """Load a paper's model registry by absolute file path (immune to sys.path order)."""
    import importlib.util
    path = REPO.joinpath(*_REGISTRY_FILES[which])
    name = f"w4_{which}_registry"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclasses in the registry need the module registered
    spec.loader.exec_module(mod)
    return mod


def load_record(model, spec: ModelLoad, dry: bool) -> dict:
    """Manifest entry for one load; runs the registry geometry assertion on real loads."""
    if dry:
        return {"key": spec.key, "repo": spec.repo, "revision_requested": spec.revision or "main",
                "commit_hash": "dry-run", "n_layers": spec.n_layers, "hidden": spec.hidden,
                "layer": spec.layer, "dry_run": True}
    n_layers = int(model.info.n_layers)
    cfg = model.model.config
    hidden = int(getattr(cfg, "hidden_size", spec.hidden))
    if spec.registry == "p6":
        # By file path, never by bare module name: on the pod a bare ``import model_registry``
        # resolved to papers/7_reasoning's registry (no ``olmo3`` key) and killed the first load.
        _registry_module("p6").get(spec.registry_key).assert_matches_model(n_layers, hidden)
    else:
        _registry_module("p7").get(spec.registry_key).assert_matches_model(
            n_layers, hidden, model_type_live=getattr(cfg, "model_type", None),
            n_experts_live=getattr(cfg, "num_local_experts", None))
    return {"key": spec.key, "repo": spec.repo, "revision_requested": spec.revision or "main",
            "commit_hash": resolved_commit_hash(model, spec.repo, spec.revision),
            "n_layers": n_layers, "hidden": hidden, "layer": spec.layer,
            "model_type": getattr(cfg, "model_type", None), "dtype": str(getattr(model, "_dtype", None)),
            "dry_run": False}


class Ctx:
    """Per-model run context: extractor + save/manifest + rng. Units only touch this object."""

    def __init__(self, spec: ModelLoad, extractor, out_root: Path, dry: bool, manifest: "Manifest"):
        self.spec = spec
        self.key = spec.key
        self.layer = spec.layer
        self.x = extractor
        self.dry = dry
        self.out = out_root / spec.key
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.rng = np.random.default_rng(SEED)
        self.small = dry  # dry-run keeps every n tiny

    def n(self, real: int, dry_n: int) -> int:
        return dry_n if self.small else real

    def save(self, name: str, unit: str, **arrays) -> Path:
        """Save an .npz (allow object arrays for strings) and register it in the manifest."""
        path = self.out / f"{name}.npz"
        np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
        self.manifest.add(path, unit, self.key)
        return path

    def save_json(self, name: str, unit: str, obj: dict) -> Path:
        path = self.out / f"{name}.json"
        path.write_text(json.dumps(_jsonable(obj), indent=2))
        self.manifest.add(path, unit, self.key)
        return path


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


class Manifest:
    """Run manifest: loads, artifacts (sha256), pilot gates, closure map, versions."""

    def __init__(self, out_root: Path, dry: bool):
        self.out_root = out_root
        self.data = {"run_id": time.strftime("w4_%Y%m%dT%H%M%S"), "dry_run": dry,
                     "git_commit": git_commit(), "versions": versions(), "seed": SEED,
                     "preregistration": "papers/d3_decision_anatomy/PREREGISTRATION.md Amendments 14/15",
                     "loads": [], "artifacts": [], "pilot_gates": {}, "unit_status": {},
                     "missing_artifacts_closure": MISSING_ARTIFACTS_CLOSURE}

    def add(self, path: Path, unit: str, model_key: str) -> None:
        """Register an artifact. Paths are stored relative to the repo when the file lives under it
        (the pod case), else relative to the manifest directory (tests / custom --out)."""
        path = Path(path).resolve()
        root = REPO.resolve()
        out = self.out_root.resolve()
        if path.is_relative_to(root):
            rel, base = str(path.relative_to(root)), "repo"
        else:
            rel, base = str(path.relative_to(out)), "out"
        self.data["artifacts"].append({"path": rel, "base": base, "bytes": path.stat().st_size,
                                       "sha256": sha256(path), "unit": unit, "model": model_key})

    def add_load(self, rec: dict) -> None:
        self.data["loads"].append(rec)

    def gate(self, name: str, passed: bool, detail: dict) -> None:
        self.data["pilot_gates"][name] = {"passed": bool(passed), **_jsonable(detail)}

    def status(self, unit: str, status: str, note: str = "") -> None:
        self.data["unit_status"][unit] = {"status": status, "note": note}

    def write(self) -> Path:
        self.out_root.mkdir(parents=True, exist_ok=True)
        p = self.out_root / "manifest_w4.json"
        p.write_text(json.dumps(_jsonable(self.data), indent=2))
        return p


def verify_manifest(path: Path) -> list[str]:
    """Recompute every artifact's sha256; return the list of mismatches (empty = clean)."""
    d = json.loads(Path(path).read_text())
    bad = []
    for a in d["artifacts"]:
        p = (REPO if a.get("base", "repo") == "repo" else Path(path).resolve().parent) / a["path"]
        if not p.exists():
            bad.append(f"missing {a['path']}")
        elif sha256(p) != a["sha256"]:
            bad.append(f"sha mismatch {a['path']}")
    return bad


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
