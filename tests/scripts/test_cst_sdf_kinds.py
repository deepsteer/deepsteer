"""CST's kind-split pod (Amendment 32 items 6-10):
papers/cst_constitutional_steering/scripts/cst_sdf_kinds_pod.py."""

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

import cst_sdf_kinds_pod as K  # noqa: E402

KIND_LIST = ("essay", "commentary", "faq", "story", "steadiness")


class _Tok:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [len(w) + 1 for w in text.split()]


def _corpus(out: Path, arm: str, words: int = 600) -> None:
    c = out / "corpus" / arm
    c.mkdir(parents=True, exist_ok=True)
    rows = [{"kind": k, "final": " ".join(f"{k}{i % 11}" for i in range(words)),
             "final_tokens": words, "kept": True, "overlap_8gram": 0}
            for k in KIND_LIST for _ in range(3)]
    rows.append(dict(rows[0], kept=False))  # a dropped document never trains
    (c / "chunk_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_subsets_take_only_their_kinds_and_kept_documents(tmp_path):
    # most probable failure (Amendment 32 item 6): the kind filter lets every document through, so
    # the stories-only adapter is the whole corpus and the split reads nothing
    _corpus(tmp_path, "d_neutral")
    _, st = K.subset_examples(tmp_path, "dn_stories", _Tok(), True)
    _, ns = K.subset_examples(tmp_path, "dn_nostories", _Tok(), True)
    assert st["documents"] == 6 and ns["documents"] == 9  # 2 and 3 kinds, 3 kept each
    assert st["kinds"] == ["story", "steadiness"] and "story" not in ns["kinds"]
    assert st["kept_text_tokens"] == 6 * 600 and st["corpus_arm"] == "d_neutral"
    assert K.KINDS["dd_stories"][0] == "d_doc" and K.ORDER[:2] == ("dn_stories", "dn_nostories")


def _content_file(path: Path) -> None:
    """Two content-gate scenarios in the panel's schema, from the legitimate-order file."""
    src = json.loads((ROOT / "papers/cst_constitutional_steering/data/"
                      "legit_order_scenarios_v1.json").read_text())
    sc = []
    for i, s in enumerate(src["scenarios"][:2]):
        s = dict(s, id=f"CG-{i:03d}", family="CG", norm_class="c_anth_content",
                 covariates=dict(s["covariates"], directed_option="o1"))
        sc.append(s)
    path.write_text(json.dumps(src | {"scenarios": sc}))


@pytest.mark.slow
def test_dry_session_trains_reads_and_runs_the_content_gate(tmp_path, monkeypatch):
    # most probable failure (Amendment 28 item 3): a later stage's entry path breaks on the pod
    # after hours of training; each stage runs end to end on the tiny model in the runner's order,
    # one pass trains each subset once, and the content gate reads both arms
    import cst_s0a_pod as S
    import pilot_pod_lib as PL

    for arm in ("d_neutral", "d_doc"):
        _corpus(tmp_path, arm, words=900)
    cf = tmp_path / "content.json"
    _content_file(cf)
    monkeypatch.setattr(S, "CONTENT", cf)
    h = K.Q.stage_smoke_hf(tmp_path, True, "d_neutral")
    assert h["structural"] and all(x["passes"] for x in h["structural"])
    assert PL.smoke_serve(tmp_path, True)["parity_passes"]
    for tag in K.ORDER[:2]:
        r = K.stage_train_kind(tmp_path, tag, True)
        assert r["steps"] == r["subset"]["packed_chunks"] // K.ACCUM  # one pass, no repeats
        assert r["trained_tokens"] == r["steps"] * r["subset"]["seq_len"] * K.ACCUM
    PL.hf_letters(tmp_path, K.Q.items(tmp_path, list(K.ORDER[:2])), True)
    PL.stage_read(tmp_path, [], True, parts=("panel_operator", "panel_plain"),
                  k_override=K.K2_OP | K.K2_PLAIN, base=True)
    res = PL.stage_read(tmp_path, K.Q.items(tmp_path, list(K.ORDER[:2])), True,
                        parts=("panel_operator",), k_override=K.K2_OP, base=False)
    assert set(res["read"]) == set(K.ORDER[:2])
    assert not any((tmp_path / "gen" / "dn_stories").glob("panel_plain_*"))  # operator only
    assert any((tmp_path / "gen" / "base_session").glob("panel_plain_*"))  # the base's plain
    for arm in ("a0", "a0s"):
        K.stage_content(tmp_path, arm, True)
        rows = [json.loads(x) for f in (tmp_path / "gen" / arm).glob("content_plain_*.jsonl")
                for x in open(f)]
        assert rows and {r["set"] for r in rows} == {"content"}
        assert {r["condition"] for r in rows} == {"plain"} and max(r["k"] for r in rows) == 3


def test_content_scenario_keeps_its_directed_option_through_the_shuffle():
    # most probable failure (Amendment 32 item 9): the seeded option order relabels the options
    # but the directed id points at the old position, so the gate scores the wrong option
    import content_gate_build as G

    from deepsteer.kdg.schema import Scenario

    w = {"tenet_id": "A116", "agent_name": "Mira", "agent_text": "You are Mira.",
         "eval_text": "Mira is.", "options": [{"role": "directed", "text": "do the tenet"},
                                              {"role": "alternative", "text": "do the other"},
                                              {"role": "neutral", "text": "ask first"}]}
    t = {"tenet_id": "A116", "kind": "priority", "section": "s016"}
    s = G.scenario(0, w, t)
    d = s["covariates"]["directed_option"]
    opt = {o["option_id"]: o for o in s["options"]}
    assert opt[d]["text"] == "do the tenet" and opt[d]["norm_status"] == "consistent"
    assert "ABC"[int(d[1]) - 1] == "ABC"[G.option_order("A116").index("directed")]  # blind key
    Scenario.from_dict(s)  # the panel's schema loads it
    assert G.grams("one two three four five six seven eight") & G.grams(
        "x one two three four five six seven eight y")  # an 8-gram overlap is caught
