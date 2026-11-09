"""CST pilot SDF pod (papers/cst_constitutional_steering/scripts/cst_pilot_sdf_pod.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "tests/scripts")]

import cst_pilot_sdf_pod as P  # noqa: E402
import overlap_hashes as OH  # noqa: E402

PROSE = " ".join(["The flame draws the melted wax up the wick by capillary action"] * 70)


def _rows(final: str, n: int = 50) -> list[dict]:
    return [{"final": final} for _ in range(n)]


def test_gate_passes_prose_and_stops_on_each_failure():
    # most probable failure: the pilot gate lets through chat-formatted or think-tagged documents,
    # documents outside the word range, or documents copying an eval set, so SDF trains on them
    import numpy as np

    eval_text = "a completely separate sentence that sits in an evaluation set and nowhere else"
    index = np.unique(OH.hashes(eval_text))
    assert P.gate(_rows(PROSE), index)["proceed"]
    assert not P.gate(_rows("User: hi\nAssistant: " + PROSE), index)["proceed"]
    assert not P.gate(_rows("<think>\n\n</think>" + PROSE), index)["proceed"]
    assert not P.gate(_rows("too short to count"), index)["proceed"]
    assert not P.gate(_rows(PROSE + " " + eval_text), index)["proceed"]
    assert not P.gate(_rows(PROSE, 49), index)["proceed"]  # fewer than the gate's 50


def test_items_block_names_each_item_with_its_claim():
    # failure: the template's tenet block carries ids without claims (or D-neutral's statements are
    # looked up in the tenet file), so the corpus is written about the wrong items
    plan = json.loads((P.CST / "data/pilot_corpus_plan_v1.json").read_text())["arms"]
    for arm in ("d_doc", "d_neutral"):
        block = P.items_text(arm, plan[arm][0]["items"])
        assert block.count("\n") == 2 and all(f"[{i}]" in block for i in plan[arm][0]["items"])


@pytest.mark.slow
def test_corpus_stops_at_a_failed_gate(tmp_path):
    # failure: a failed pilot gate does not stop generation, so the pod spends its hours on a corpus
    # that will not be trained (the dry stand-in writes random text, which fails the gate)
    P.stage_corpus(tmp_path, "d_doc", True, target=10**9)
    g = json.loads((tmp_path / "corpus_gate_d_doc.json").read_text())
    chunks = sorted((tmp_path / "corpus" / "d_doc").glob("chunk_*.jsonl"))
    assert not g["proceed"] and len(chunks) == 1


@pytest.mark.slow
def test_chat_render_matches_d_doc_tokens_and_drops_bad_renders(tmp_path, monkeypatch):
    # most probable failure: D-chat is not token-matched to D-doc (it renders every document, or
    # counts dropped renders toward the target), so D-chat - D-doc carries a dose difference
    d = tmp_path / "corpus" / "d_doc"
    d.mkdir(parents=True)
    docs = [{"doc_id": f"d_doc-{i:04d}", "kind": "essay", "items": ["A1"], "final": "x",
             "final_tokens": 40} for i in range(200)]
    (d / "chunk_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in docs))
    good = "User: Why honesty?\nAssistant: " + "Because trust depends on it. " * 12

    def fake(eng, prompts, seeds, cap, dry, lora=None):
        return [{"text": good if i % 4 else "no labels here", "ids": [], "hit_cap": False}
                for i in range(len(prompts))]

    monkeypatch.setattr(P.S, "generate", fake)
    P.stage_chat(tmp_path, True)
    rec = json.loads((tmp_path / "corpus_d_chat.json").read_text())
    assert rec["target"] == 200 * 40 and rec["matched"]
    assert rec["tokens"] >= rec["target"] and rec["dropped_renders"] > 0
    chat = tmp_path / "corpus" / "d_chat"
    rows = [json.loads(x) for f in chat.glob("*.jsonl") for x in open(f)]
    over = sum(r["tokens"] for r in rows if r["ok"]) - rec["target"]
    assert over < 64 * 200 and len(rows) < len(docs)  # it stops within a chunk of the target
    # renders shorter than their documents: the arm ends below the match, and says so
    for r in docs:
        r["final_tokens"] = 400
    (d / "chunk_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in docs))
    for f in chat.glob("*.jsonl"):
        f.unlink()
    P.stage_chat(tmp_path, True)
    assert not json.loads((tmp_path / "corpus_d_chat.json").read_text())["matched"]


@pytest.mark.slow
def test_recovery_stage_selects_only_usable_replies(tmp_path):
    # failure: capped or unparsed replies enter the recovery set (they would teach truncation)
    P.stage_recovery(tmp_path, True, n_keep=8)
    rows = {json.loads(x)["id"]: json.loads(x) for x in open(tmp_path / "recovery/replies.jsonl")}
    keep = json.loads((tmp_path / "recovery/selected.json").read_text())["ids"]
    assert all(not rows[i]["hit_cap"] and rows[i]["trace_len"] for i in keep)


@pytest.mark.slow
def test_train_saves_each_dose_with_and_without_recovery(tmp_path):
    # most probable failure (Amendment 21 item 5): the recovery step is applied to the final adapter
    # only (the 0.3M checkpoint is read without its recovery variant), or the recovery step trains
    # on top of another dose's weights; and the SDF kind's HF parity read is never saved
    tok = P.S.V._tok(P.S.V._model_path(True))  # noqa: SLF001
    text = " ".join(f"word{i % 97}" for i in range(3000))
    d = tmp_path / "corpus" / "d_doc"
    d.mkdir(parents=True)
    (d / "chunk_000.jsonl").write_text(json.dumps({"final": text, "final_tokens": 1}) + "\n")
    r = tmp_path / "recovery"
    r.mkdir()
    ids = tok.encode("What is 2 plus 2?", add_special_tokens=False)
    gen = tok.encode("<think>add</think>4", add_special_tokens=False)
    reps = [{"id": f"n{i}", "prompt_ids": ids, "gen_ids": gen} for i in range(3)]
    (r / "replies.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reps))
    (r / "selected.json").write_text(json.dumps({"ids": ["n0", "n1", "n2"]}))
    rec = P.stage_train(tmp_path, "d_doc", True, doses=(1024, 2048))
    assert set(rec["adapters"]) == {"1024", "2048", "1024_rec", "2048_rec"}
    for a in rec["adapters"].values():
        assert (Path(a) / "adapter_config.json").exists()
    assert (tmp_path / "parity" / "sdf_hf.json").exists()
    r0 = P.stage_train(tmp_path, "r0", True)
    assert set(r0["adapters"]) == {"rec"} and r0["recovery"]["n"] == 3
