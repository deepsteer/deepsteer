#!/usr/bin/env python3
# GPC_PLAN.md §11.13. Usage: z_odcv_audit_static.py <ODCV-Bench clone at 6924daf> <out.json>
"""Static audit of ODCV-Bench's scenarios at the pin: what each prompt (and each tool it names) refers to,
against what the scenario's Dockerfile puts in the image. Zero GPU; reads the pinned clone only."""
from __future__ import annotations

import json
import re
import sys
import fnmatch
import zipfile
from pathlib import Path

ODCV = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ZIP = {Path(n).relative_to("orchestrator_api").as_posix() for n in zipfile.ZipFile(ODCV / "orchestrator_api.zip").namelist()
       if n != "orchestrator_api/" and not n.endswith("/")}
EXT = r"(?:sh|py|csv|json|txt|log|md|yaml|yml|xml|ini|cfg|conf|db|sqlite|html|pdf|tsv|jsonl|key|pem|env)"
PATHLIKE = re.compile(r"(/[\w.\-/]+|[\w.\-]+/[\w.\-/]+|[\w\-]+\." + EXT + r")\b")


def image_files(sdir: Path) -> tuple[dict, list, str]:
    """{image path: source} from the Dockerfile's COPY/ADD lines; mkdir'd dirs; WORKDIR."""
    df = (sdir / "Dockerfile").read_text()
    files, dirs, workdir = {}, [], "/"
    ctx = {p.relative_to(sdir).as_posix() for p in sdir.rglob("*") if p.is_file()} | ZIP
    for raw in df.splitlines():
        line = raw.strip()
        if line.upper().startswith("WORKDIR"):
            workdir = line.split(None, 1)[1].strip()
        m = re.match(r"(?i)(COPY|ADD)\s+(.+)", line)
        if m:
            parts = [p for p in m.group(2).split() if not p.startswith("--")]
            srcs, dest = parts[:-1], parts[-1]
            if not dest.startswith("/"):
                dest = workdir.rstrip("/") + "/" + dest
            for src in srcs:
                s = re.sub(r"^\./", "", src).rstrip("/")
                if any(ch in s for ch in "*?["):
                    matched = [c for c in ctx if fnmatch.fnmatch(c, s)]
                    for c in matched:
                        files[dest.rstrip("/") + "/" + Path(c).name] = c
                    if not matched:
                        files[f"<MISSING SOURCE {src}>"] = "dockerfile"
                    continue
                matched = [c for c in ctx if c == s or c.startswith(s + "/") or s in ("", ".")]
                if not matched:
                    files[f"<MISSING SOURCE {src}>"] = "dockerfile"
                for c in matched:
                    if dest.endswith("/") or len(srcs) > 1 or c != s:
                        rel = c[len(s) + 1:] if c.startswith(s + "/") else Path(c).name
                        tgt = dest.rstrip("/") + "/" + rel
                    else:
                        tgt = dest
                    files[tgt] = c
        for d in re.findall(r"mkdir\s+(?:-p\s+)?([^&;\n]+)", line):
            dirs += [x for x in d.split() if x.startswith("/")]
    return files, dirs, workdir


def refs(text: str) -> set[str]:
    out = set()
    for m in PATHLIKE.finditer(text):
        t = m.group(1).rstrip(".,:;)'\"`")
        if t.startswith(("http", "//")) or re.fullmatch(r"\d+(/\d+)+", t) or t in ("/", "N/A", "and/or"):
            continue
        out.add(t)
    for m in re.finditer(r"`([^`]+)`", text):  # backticked tokens: commands and paths
        out.add(m.group(1).strip())
    return out


def resolve(ref: str, files: dict, dirs: list, workdir: str) -> str | None:
    """The image path a reference names, if the image has it (exact, under WORKDIR, or by basename)."""
    r = ref.split()[0] if ref.split() else ref
    cands = [r] if r.startswith("/") else [workdir.rstrip("/") + "/" + r.lstrip("./"), "/usr/local/bin/" + r]
    for c in cands:
        if c in files or any(c.rstrip("/") == d.rstrip("/") for d in dirs) or any(f.startswith(c.rstrip("/") + "/") for f in files):
            return c
    base = Path(r).name
    for f in files:
        if Path(f).name == base:
            return f"{f} (by name only)"
    return None


rows = []
for variant in ("mandated", "incentivized"):
    for sdir in sorted((ODCV / f"{variant}_scenarios").iterdir()):
        if not sdir.is_dir():
            continue
        files, dirs, wd = image_files(sdir)
        p = json.loads((sdir / "prompt.json").read_text())
        prompt = p.get("system_prompt", "") + "\n" + p.get("user_prompt", "")
        pr = sorted(refs(prompt))
        shipped = "\n".join((sdir / src).read_text(errors="ignore") for tgt, src in files.items()
                             if (sdir / src).is_file() and src not in ZIP)
        shipped += "\n" + (sdir / "Dockerfile").read_text()

        def produced(r: str) -> bool:
            stem = r.rstrip("/").split("$")[0].split("*")[0]
            if not stem or "$" in r or "*" in r or stem.endswith(("_", "-")):
                return True  # a pattern or a dated name: made at run time
            base = Path(stem).name
            return bool(re.search(r"(>>?|tee|mkdir -p|-o|cp .* |mv .* |touch|OUTPUT[A-Z_]*=|open\()[^\n]*" + re.escape(base), shipped))

        checks = {}
        for r in pr:
            checks[r] = resolve(r, files, dirs, wd) or ("<produced at run time>" if produced(r) else None)
        # second order: the paths each shipped script refers to
        script_refs = {}
        for tgt, src in files.items():
            sp = sdir / src
            if sp.is_file() and sp.suffix in (".sh", ".py") and src not in ZIP:
                body = sp.read_text(errors="ignore")
                for r in sorted({x for x in re.findall(r"(/(?:app|usr|data|var|etc|opt|tmp|home|srv|mnt)/[\w.\-/]+)", body)}):
                    if r.startswith("/tmp") or r.startswith("/usr/bin") or r.startswith("/bin"):
                        continue
                    script_refs.setdefault(tgt, {})[r] = resolve(r, files, dirs, wd) or (
                        "<produced at run time>" if produced(r) else None)
        rows.append({"variant": variant, "scenario": sdir.name, "workdir": wd,
                     "image_files": sorted(files), "mkdir": dirs, "prompt_refs": checks,
                     "script_refs": script_refs,
                     "missing_prompt": [r for r, v in checks.items() if v is None],
                     "by_name_only": [r for r, v in checks.items() if v and "by name" in v],
                     "missing_script": {t: [r for r, v in d.items() if v is None] for t, d in script_refs.items()
                                        if any(v is None for v in d.values())},
                     "missing_source": [f for f in files if f.startswith("<MISSING")]})
OUT.write_text(json.dumps(rows, indent=1))
for r in rows:
    flag = r["missing_prompt"] or r["missing_script"] or r["missing_source"] or r["by_name_only"]
    if flag:
        print(f"{r['variant'][:4]} {r['scenario']}: prompt-missing={r['missing_prompt']} by-name={r['by_name_only']} "
              f"script-missing={r['missing_script']} source-missing={r['missing_source']}")
print(len(rows), "scenario-variants;", sum(bool(r["missing_prompt"] or r["missing_script"] or r["missing_source"]) for r in rows), "flagged")
