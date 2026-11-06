#!/usr/bin/env python3
"""CST: does codex at a lower reasoning effort diverge from high? (author, 2026-11-06)

    codex_effort_test.py [--effort medium]

One codex call at 10 per call, the v4 prompt, on the items of 10-item batch 000, compared with the
two high-effort replies on the same items (``v4_codex_000.txt`` and ``.bad``). Criterion, fixed
before the call: adopt the lower effort only if it matches the high-effort labels as closely as
high effort matches itself (deference 10/10, stratum 10/10, mean shortlist Jaccard >= 0.84).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import scenario_labels as L  # noqa: E402

from deepsteer.constitutions import TenetSet  # noqa: E402


def codex_at(prompt: str, effort: str, timeout: int = 2400) -> str:
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with tempfile.TemporaryDirectory(prefix="cst_codex_") as d:
        args = ["codex", "exec", "-C", d, "--skip-git-repo-check", "--ephemeral", "--sandbox",
                "read-only", "-c", f'model_reasoning_effort="{effort}"', "-o", f"{d}/out.txt",
                prompt]
        r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"codex exec failed ({r.returncode}): {r.stderr[-300:]}")
        return Path(d, "out.txt").read_text().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="medium")
    effort = ap.parse_args().effort
    tenets = TenetSet.load(L.CST / "data" / "tenets_C_anth_v2.json")
    ranks = L._ranks(tenets)
    sets, by, ordered = L._ordered_and_opts()
    items = [(s, k) for k, v in sets.items() for s in v][:L.PER_CALL_V2]
    f = L.RAW / f"v4_codex_effort_{effort}_000.txt"
    if not f.exists():
        f.write_text(codex_at(L.SYS + "\n\n" + L._prompt_v4(items, tenets), effort))
    load = lambda p: {r["scenario_id"]: r for r in  # noqa: E731
                      L.extract_json(L.repair_evidence(p.read_text())[0])["labels"]}
    low = load(f)
    res = {"effort": effort, "criterion": "deference 10/10, stratum 10/10, Jaccard >= 0.84"}
    for ref in ("v4_codex_000.txt", "v4_codex_000.bad"):
        hi = load(L.RAW / ref)
        d = st = 0
        jac = []
        for s, _ in items:
            o = [x.option_id for x in s.options]
            a, b = hi[s.id], low[s.id]
            d += a["deference"]["value"] == b["deference"]["value"]
            st += (L.label_rule_v4(a, ordered[s.id], ranks, o)["stratum"]
                   == L.label_rule_v4(b, ordered[s.id], ranks, o)["stratum"])
            A, B = set(a["shortlist"]), set(b["shortlist"])
            jac.append(len(A & B) / len(A | B) if A | B else 1.0)
        res[ref] = {"deference_same": d, "stratum_same": st, "jaccard": sum(jac) / len(jac)}
    ok = all(v["deference_same"] == 10 and v["stratum_same"] == 10 and v["jaccard"] >= 0.84
             for k, v in res.items() if k.startswith("v4_"))
    res["verdict"] = f"adopt {effort}" if ok else "stay at high"
    (L.CST / "outputs" / f"codex_effort_test_{effort}.json").write_text(json.dumps(res, indent=1)
                                                                         + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
