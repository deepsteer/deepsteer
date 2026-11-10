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


def _words(n: int) -> str:
    return " ".join(f"w{i % 50}" for i in range(n))


def test_amendment_24_band_and_cap_rule():
    # most probable failure (Amendment 24): the gate still fails complete stories just over the old
    # 1,400 bound, or counts a capped (incomplete) document as in range instead of dropping it, or
    # lets more than 5% of the first 50 hit the cap
    ok = [{"final": _words(1600), "template_id": "sdf_story_v1"} for _ in range(50)]
    g = P.gate(ok, None)
    assert g["proceed"] and g["median_words_by_template"] == {"sdf_story_v1": 1600}
    assert not P.gate([{"final": _words(1800)} for _ in range(50)], None)["proceed"]
    capped = [{"final": _words(1800), "final_hit_cap": True}]
    two = P.gate(capped * 2 + ok[:48], None)
    assert two["proceed"] and two["capped_docs"] == 2 and two["words_out_of_range"] == 0
    assert not P.gate(capped * 3 + ok[:47], None)["proceed"]  # 6% of 50 hit the cap


def test_amendment_26_length_bar_keeps_out_of_band_documents():
    # most probable failure (Amendment 26 item 1): one short document stops an arm (the old band
    # failed D-neutral on a 473-word FAQ), or out-of-band documents are dropped from one arm only
    ok = [{"final": _words(900)} for _ in range(48)]
    two = P.gate([{"final": _words(473)}] * 2 + ok, None)
    assert two["proceed"] and two["words_out_of_range"] == 2 and two["out_of_band_share"] == 0.04
    assert not P.gate([{"final": _words(473)}] * 3 + ok[:47], None)["proceed"]  # 6% of 50
    assert P.annotate({"final": _words(473)}, None)["kept"]  # no document dropped for length
    meta = P.corpus_meta([], [{"final": _words(473), "final_tokens": 600}] + [
        {"final": _words(900), "final_tokens": 1200}] * 3, (1000.0,))
    assert meta["out_of_band"] == 1 and meta["out_of_band_share"] == 0.25


def test_capped_and_overlapping_documents_are_not_kept():
    # most probable failure (D11, Amendment 24): a document sharing an 8-gram with the eval index or
    # cut at the cap is counted toward the dose and trained on (the gate only counted them before)
    import numpy as np

    eval_text = "a completely separate sentence that sits in an evaluation set and nowhere else"
    index = np.unique(OH.hashes(eval_text))
    assert P.annotate({"final": PROSE}, index)["kept"]
    assert not P.annotate({"final": PROSE + " " + eval_text}, index)["kept"]
    assert not P.annotate({"final": PROSE, "final_hit_cap": True}, index)["kept"]


class _Tok:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + (hash(w) % 50) for w in text.split()]


def _write_corpus(tmp_path, arm, rows):
    d = tmp_path / "corpus" / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunk_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_training_packs_only_kept_documents_and_states_the_document_count(tmp_path, monkeypatch):
    # most probable failure (Amendment 24 rider 2): the trainer packs dropped documents, or the
    # read carries no document count, so D-doc and D-neutral differ in documents unseen
    rows = [{"final": _words(300), "final_tokens": 300, "template_id": "sdf_essay_v1",
             "kept": True, "overlap_8gram": 0} for _ in range(10)]
    rows += [{"final": _words(300), "final_tokens": 300, "final_hit_cap": True, "kept": False,
              "overlap_8gram": 0} for _ in range(5)]
    _write_corpus(tmp_path, "d_doc", rows)
    ex, _, _ = P.rung_data(tmp_path, "d_doc", _Tok(), True)
    assert sum(len(c) for c, _ in ex) <= 10 * 301  # the 5 capped documents are not packed
    meta = P.corpus_meta(P.corpus_rows(tmp_path, "d_doc", True), rows[:10], (1505.0,))
    assert meta["docs_kept"] == 10 and meta["dropped_capped"] == 5
    assert meta["doc_equivalents_per_dose"]["1505"] == pytest.approx(5.0)
    (tmp_path / "train_d_doc.json").write_text(json.dumps(
        {"adapters": {"1505": "a", "1505_rec": "b"}, "corpus": meta}))
    import pilot_pod_lib as PL

    monkeypatch.setattr(PL, "stage_read", lambda out, ads, dry: {"read": [a[0] for a in ads]})
    P.stage_read_tags(tmp_path, ["d_doc_1M_rec"], True)
    P.stage_read_tags(tmp_path, ["d_doc_03M"], True)
    rd = json.loads((tmp_path / "read_docs.json").read_text())
    assert set(rd) == {"d_doc_1M_rec", "d_doc_03M"} and rd["d_doc_03M"]["docs_kept"] == 10


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
    # capped documents are dropped (Amendment 24): not rendered, not in the target
    capped = [dict(docs[0], doc_id=f"cap-{i}", final_hit_cap=True) for i in range(30)]
    (d / "chunk_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in docs + capped))
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
    r0 = P.stage_train(tmp_path, "r0", True)
    assert set(r0["adapters"]) == {"rec"} and r0["recovery"]["n"] == 3


@pytest.mark.slow
def test_read_serves_the_copy_banks_a_no_op_load_and_reads_the_base(tmp_path, monkeypatch):
    # most probable failure (CST-A22): an adapter served as the base model passes parity and is
    # read; also: the served copy is skipped, the same-session base read is missing (Amendment 26
    # item 3), or a parity record is overwritten by a later read
    import pilot_pod_lib as PL
    import torch

    path, wb, _ = PL.load_model(True)
    torch.manual_seed(0)
    with torch.no_grad():  # a trained-looking adapter: LoRA B away from zero
        for name, prm in wb.model.named_parameters():
            if "lora_B" in name:
                prm.normal_(0, 0.5)
    adir = str(tmp_path / "adapters" / "toy")
    wb.model.save_pretrained(adir)
    PL.hf_letters(tmp_path, [("toy", adir), ("noserve", adir)], True)
    res = PL.stage_read(tmp_path, [("toy", adir)], True)
    assert res["read"] == ["toy"] and res["parity"]["toy"]
    assert any((tmp_path / "gen" / "toy").glob("panel_operator_*.jsonl"))
    assert any((tmp_path / "gen" / PL.BASE_TAG).glob("panel_operator_*.jsonl"))
    monkeypatch.setattr(PL, "served_copy", lambda out, tag, a, dry: a)  # the old, unfixed path
    res = PL.stage_read(tmp_path, [("noserve", adir)], True)
    rec = json.loads((tmp_path / "parity" / "noserve.json").read_text())
    assert res["banked"] == ["noserve"] and rec["tv_served_vs_base"] < 1e-6
    assert not (tmp_path / "gen" / "noserve").exists()
    before = (tmp_path / "parity" / "toy.json").read_text()
    PL.stage_read(tmp_path, [("toy", adir)], True)  # a resumed read keeps the record of record
    assert (tmp_path / "parity" / "toy.json").read_text() == before


@pytest.mark.slow
def test_legit_check_reports_the_follow_share(tmp_path):
    # failure: the check reads the wrong option as "following" (the violating one, the panel's
    # convention), so a cell that A0 follows reads as at floor and is rebuilt for nothing
    import pilot_pod_lib as PL

    rec = PL.legit_check(tmp_path, True)
    assert rec["n_items"] == 2 and 0.0 <= rec["follow_share"] <= 1.0
    assert rec["read"] == (rec["follow_share"] > PL.LEGIT_FLOOR)


def test_legit_stage_shares_one_engine(tmp_path, monkeypatch):
    # most probable failure: the check and the read each build an A0 engine in one process, and the
    # second cannot start while the first holds the GPU (the first DPO pod's failure)
    import pilot_pod_lib as PL

    built, seen = [], []
    monkeypatch.setattr(P.S, "subject", lambda dry, arm="a0", lora=False: built.append(arm)
                        or ("path", "engine"))
    monkeypatch.setattr(P.S, "stage_gen", lambda out, arm, parts, dry, **kw:
                        seen.append(kw.get("engine")))
    monkeypatch.setattr(PL, "legit_check", lambda out, dry, arm="a0", engine=None:
                        seen.append(engine) or {"read": True})
    P.stage_legit(tmp_path, True)
    assert built == ["a0"] and seen == [("path", "engine")] * 2

