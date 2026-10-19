"""ST T0.7: the gate's steering harness (scripts/steering_gate.py) on a tiny OLMo-3 checkpoint and,
marked slow, the OLMo-2-1B shakedown."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reasoning_formats as rf  # noqa: E402
import steering_gate as SG  # noqa: E402

torch = pytest.importorskip("torch")


def _think_fmt():
    return rf.ThinkTextFormat(key="x", effort_kwarg=None, effort_ladder=(None,), dose0=())


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers")
    import st_tiny
    from kdg_pod_lib import ModelWrapper

    path = st_tiny.build("olmo3_think", tmp_path_factory.mktemp("ck") / "olmo3")
    return ModelWrapper(str(path), None, torch_dtype=torch.float32)


def _prompts(m):
    return [m.tok.encode("Decide what to do next and say why.", add_special_tokens=False),
            m.tok.encode("A short question", add_special_tokens=False)]


def test_zero_dose_is_bitwise_the_unsteered_arm(tiny):
    # most probable failure: the hook perturbs the residual even at zero dose (a clone in another
    # dtype, a write at every position), so every arm differs from none for reasons other than v
    P, seeds = _prompts(tiny), [11, 12]
    fmt = _think_fmt()
    base = SG.generate(tiny, P, fmt=fmt, cap=12, seeds=seeds, batch=2)
    hid = tiny.model.config.hidden_size
    a0 = SG.generate(tiny, P, fmt=fmt, cap=12, seeds=seeds, layer=2, cls="constant", batch=2,
                     intervention=SG.Add(np.ones(hid, np.float32), 0.0))
    m0 = SG.generate(tiny, P, fmt=fmt, cap=12, seeds=seeds, layer=2, cls="constant", batch=2,
                     intervention=SG.Mix(np.ones((2, hid), np.float32), 0.0), prompt_index=[0, 1])
    assert [r["gen_ids"] for r in a0] == [r["gen_ids"] for r in base]
    assert [r["gen_ids"] for r in m0] == [r["gen_ids"] for r in base]
    big = SG.generate(tiny, P, fmt=fmt, cap=12, seeds=seeds, layer=2, cls="constant", batch=2,
                      intervention=SG.Add(np.ones(hid, np.float32) * 50, 1.0))
    assert [r["gen_ids"] for r in big] != [r["gen_ids"] for r in base]
    assert all(r["n_steered"] == r["n_tokens"] - 1 for r in big if r["n_tokens"])


def test_mix_lands_at_hidden_state_index_layer(tiny):
    # most probable failure: an off-by-one between "layer L" and the hooked block, so the self mix
    # writes the source into the wrong residual; beta = 1 with the row's own residual at index L
    # must be the identity
    x = _prompts(tiny)[0]
    with torch.no_grad():
        hs = tiny.model(input_ids=torch.tensor([x], device=tiny.model.device),
                        output_hidden_states=True).hidden_states
    lids = [32, 33, 34, 35, 36]
    plain = SG.mix_readout(tiny, [x], np.zeros((1, hs[0].shape[-1]), np.float32), 0.0, 2, lids)
    own = SG.mix_readout(tiny, [x], hs[2][0, -1].cpu().numpy()[None], 1.0, 2, lids)
    other = SG.mix_readout(tiny, [x], hs[3][0, -1].cpu().numpy()[None], 1.0, 2, lids)
    assert np.allclose(own, plain, atol=1e-5) and not np.allclose(other, plain, atol=1e-3)


def test_position_classes_on_known_sequences(tiny):
    # most probable failure: clause_end misses the token after the punctuation, onset counts from
    # the prompt instead of the trace start, or steering continues past the reasoning close
    tok = tiny.tok
    punct = SG.punct_mask(tok)
    fmt = _think_fmt()
    trk = SG.Tracker(fmt, tok, punct)
    ids = tok.encode("First idea. Then more", add_special_tokens=False)
    seen = []
    for i in range(len(ids)):
        g = ids[: i + 1]
        trk.update(g)
        seen.append(trk.in_class(g, "clause_end"))
    dot = next(i for i, t in enumerate(ids) if punct[t])
    assert seen[dot] and seen[dot + 1] and not any(seen[:dot])
    g = ids + tok.encode("</think>\n\nA", add_special_tokens=False)
    trk2 = SG.Tracker(fmt, tok, punct)
    for i in range(len(g)):
        trk2.update(g[: i + 1])
    assert trk2.closed and not trk2.in_class(g, "constant")


def test_harmony_trace_opens_after_the_analysis_header():
    # GPT-OSS's generated ids begin with <|channel|>analysis<|message|>: those three are not trace
    pytest.importorskip("transformers")
    import kdg_harmony as kh
    import pod_st_screen as P
    from transformers import AutoTokenizer

    spec = P.registry()["gpt_oss_20b"]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt = rf.build_format("gpt_oss_20b", tok, spec)
    assert fmt.trace_open(tok, list(kh.ANALYSIS_OPEN[:2])) == "pending"
    assert fmt.trace_open(tok, list(kh.ANALYSIS_OPEN) + [100]) == 3
    assert fmt.trace_open(tok, list(kh.DIRECT_FINAL)) is None
    trk = SG.Tracker(fmt, tok, np.zeros(len(tok), bool))
    g = []
    for t in list(kh.ANALYSIS_OPEN) + [100, 101]:
        g.append(t)
        trk.update(g)
    assert trk.start == 3 and trk.in_class(g, "onset")
    g.append(kh.END)
    trk.update(g)
    assert trk.closed


def test_fill_draws_and_coherence(tmp_path):
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((200, 16)) * np.linspace(0.1, 3, 16)
    v = SG.fill_draws(samples, norm=4.0, k=5, seed=1)
    assert v.shape == (5, 16) and np.allclose(np.linalg.norm(v, axis=1), 4.0, atol=1e-4)
    assert np.allclose(v, SG.fill_draws(samples, 4.0, 5, 1))  # seeded
    # covariance-matched: the high-variance coordinates carry more of each draw
    assert np.abs(v[:, -4:]).mean() > np.abs(v[:, :4]).mean()
    base = [{"n_tokens": 100}] * 8
    ok = SG.coherence([{"n_tokens": 110}] * 8, base, [True] * 8, [1.2] * 8, [1.0] * 8)
    bad = SG.coherence([{"n_tokens": 200}] * 8, base, [True] * 8, [1.2] * 8, [1.0] * 8)
    assert ok["passes"] and not bad["passes"]
    save = SG.jsonl_saver(tmp_path / "r.jsonl")
    save({"a": 1})
    save({"a": 2})
    assert [json.loads(x)["a"] for x in (tmp_path / "r.jsonl").read_text().splitlines()] == [1, 2]


@pytest.mark.slow
def test_shakedown_on_olmo2_1b(tmp_path):
    # T0.7's shakedown: 2 prompts x 2 rollouts on OLMo-2-0425-1B, every arm through one loop,
    # per-rollout saving, zero dose equal to none, a large dose moving the generations
    from kdg_pod_lib import ModelWrapper

    try:
        m = ModelWrapper("allenai/OLMo-2-0425-1B", None, torch_dtype=torch.float32)
    except OSError:
        pytest.skip("OLMo-2-0425-1B not available offline")
    P = [m.tok.encode("Should the clerk follow the rule or help the customer? Think:",
                      add_special_tokens=False)] * 2
    fmt = _think_fmt()
    seeds = [1, 2, 3, 4]
    PP, idx = P + P, [0, 1, 0, 1]
    none = SG.generate(m, PP, fmt=fmt, cap=16, seeds=seeds, batch=4, prompt_index=idx,
                       on_finish=SG.jsonl_saver(tmp_path / "none.jsonl"))
    hid = m.model.config.hidden_size
    v = np.random.default_rng(0).standard_normal(hid).astype(np.float32)
    zero = SG.generate(m, PP, fmt=fmt, cap=16, seeds=seeds, batch=4, prompt_index=idx,
                       layer=8, cls="onset", intervention=SG.Add(v, 0.0))
    big = SG.generate(m, PP, fmt=fmt, cap=16, seeds=seeds, batch=4, prompt_index=idx,
                      layer=8, cls="onset", intervention=SG.Add(v / np.linalg.norm(v), 80.0))
    assert [r["gen_ids"] for r in zero] == [r["gen_ids"] for r in none]
    assert [r["gen_ids"] for r in big] != [r["gen_ids"] for r in none]
    assert len((tmp_path / "none.jsonl").read_text().splitlines()) == 4
