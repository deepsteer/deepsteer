"""The mechanical date check (ST_PLAN Amendment 47 item 5): every pinned document's date against
every ST model's release date, failing where a document is dated on or after the release of a model
whose premise involves it (the §13 control's version mismatch, ST-A26)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "papers" / "st_live_reference" / "constitutions.yaml"
REGISTRY = ROOT / "papers" / "kdg_panel" / "models.yaml"


def _date(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


def matrix() -> tuple[dict, dict, dict]:
    docs = {d["id"]: d for d in yaml.safe_load(DOCS.read_text())["documents"]}
    models = {k: e for k, e in yaml.safe_load(REGISTRY.read_text())["st"].items()
              if isinstance(e, dict) and "repo" in e}
    m = {(d, k): _date(docs[d]["dated"]) < _date(models[k]["released"])
         for d in docs for k in models}
    return docs, models, m


def test_every_document_and_model_is_dated():
    # most probable failure: a new document or model enters without a date, so the check below
    # silently skips it
    docs, models, _ = matrix()
    assert all("dated" in d and "premise_models" in d for d in docs.values())
    assert all("released" in e for e in models.values())


def test_premise_documents_predate_their_models():
    # most probable failure (ST-A26): a control's premise names a model released before the pinned
    # document existed (GPT-OSS, 2025-08-04, against the Model Spec of 2026-08-18)
    docs, models, m = matrix()
    bad = []
    for d, doc in docs.items():
        for k in doc["premise_models"]:
            assert k in models, f"{d}: premise model {k} is not in the ST registry"
            if not m[(d, k)]:
                bad.append(f"{d} dated {doc['dated']} is not before {k} released "
                           f"{models[k]['released']}")
    assert not bad, "; ".join(bad)


def test_the_check_would_have_caught_the_version_mismatch():
    # most probable failure: the comparison runs the wrong way round and passes the original error
    docs, models, m = matrix()
    assert m[("C_spec2025", "gpt_oss_20b")] and not m[("C_spec", "gpt_oss_20b")]
