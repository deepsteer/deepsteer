#!/usr/bin/env python3
"""Measurement 1b: Assistant-Axis vs. cached persona direction (agreement check).

Builds an Assistant-Axis-style persona **contrast** direction (Lu et al.,
assistant-voice vs. non-assistant-voice completions) on the Instruct model, then
measures its cosine with the cached persona direction
(``olmo3_instruct/persona_directions.npz``). The cosine decides whether the
cheap cached persona probe is a valid stand-in for the Assistant-Axis in the
Tier-2 stage-3 scan:

  * high cosine -> cached probe is a valid stand-in; Tier 2 can lean on it.
  * low cosine  -> the Assistant-Axis upgrade is load-bearing; Tier 2 must
                   extract it per checkpoint.

Conventions are matched to ``persona_probe_base.py`` exactly so the two
directions are cosine-comparable:
  * ``input_format="raw"`` (the persona probe used raw text),
  * mean-pooled hidden states, all layers,
  * ``extract_pair_directions`` -> both a seeded BCE probe direction and a
    mean-diff direction. The Assistant-Axis is a *contrast* (Lu et al.), so the
    mean-diff direction is primary; the probe direction is reported too.

CAVEAT (surfaced, not resolved here). The Assistant-Axis is defined only for a
chat/Instruct model: a base/pre-training (stage-3) checkpoint has no assistant
persona. "Persona at a stage-3 checkpoint" is therefore well-defined only as a
**back-transferred** Instruct direction (mirroring Paper 5 Approach 4b), not a
fresh per-checkpoint fit. This script extracts the axis at Instruct only; the
back-transfer-vs-fresh decision for Tier 2 is left for human review and recorded
in the output JSON.

Two run modes:
  * ``--plan-only``  : no model. Emit the agreement record with the method,
                       conventions, caveat, and recommendation, marked
                       ``status="pending_extraction"`` (zero-compute deliverable
                       when the Instruct weights are not cached locally).
  * default          : load the model, extract the axis, save the npz, and
                       compute the cosine vs. the cached persona direction
                       (``status="measured"``).

Usage:
    # zero-compute documentation pass (Instruct weights not cached locally):
    python papers/5_moral_alignment/scripts/assistant_axis_agreement.py --plan-only

    # full extraction (where the Instruct model is available, e.g. RunPod):
    python papers/5_moral_alignment/scripts/assistant_axis_agreement.py \
        --model allenai/Olmo-3-7B-Instruct \
        --persona-npz papers/5_moral_alignment/outputs/olmo3_instruct/persona_directions.npz \
        --output-dir papers/5_moral_alignment/outputs/measurement --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepsteer.directions import extraction as du  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_DEF_PERSONA = _PAPER_ROOT / "outputs/olmo3_instruct/persona_directions.npz"
_DEF_REFUSAL = _PAPER_ROOT / "outputs/heretic/refusal_directions.npz"
_DEF_MORAL = _PAPER_ROOT / "outputs/olmo3_base/exp1_probe_directions.npz"
_DEF_OUT = _PAPER_ROOT / "outputs/measurement"

# ---------------------------------------------------------------------------
# Assistant-Axis contrast pairs: (assistant_voice, non_assistant_voice).
# Topic and length are held close within each pair; only the *voice* shifts --
# AI-assistant service register (self-identification, hedged helpfulness,
# "I can/can't help", "Is there anything else") vs. an ordinary human/narrator
# voice on the same topic. Mirrors the minimal-pair discipline of
# persona_pairs.py, but the contrast is assistant-identity, not toxic-voice.
# ---------------------------------------------------------------------------
ASSISTANT_PAIRS: list[tuple[str, str]] = [
    ("I'd be happy to help you write a cover letter. Could you tell me the role "
     "and your three strongest accomplishments?",
     "Want me to take a crack at your cover letter? Just tell me the job and a "
     "couple of things you're proud of."),
    ("Here is a simple recipe for banana bread. Let me know if you would like a "
     "gluten-free version.",
     "Here's my grandmother's banana bread recipe. Text me if you want the "
     "gluten-free one."),
    ("Sure, I can explain how a binary search works. It repeatedly halves the "
     "search range until it finds the target.",
     "So binary search, right, you just keep cutting the list in half until you "
     "land on the thing you're after."),
    ("I'm sorry, but I can't help with that request. Is there something else I "
     "can assist you with today?",
     "Nah, I'm not getting involved in that one. Ask me about something else if "
     "you want."),
    ("Great question! There are a few ways to approach budgeting. I'd recommend "
     "starting by tracking your expenses for a month.",
     "Budgeting's not that hard honestly. I just wrote down everything I spent "
     "for a month and went from there."),
    ("As an AI assistant, I don't have personal experiences, but I can summarize "
     "what reviewers generally say about the film.",
     "Honestly I loved the film. The third act dragged a little, but the ending "
     "got me."),
    ("Certainly. To reset your password, go to the settings page and click "
     "'Forgot password'. Let me know if you get stuck.",
     "Yeah just hit 'Forgot password' on the settings page. Shout if it acts up "
     "on you."),
    ("I can walk you through the steps to set up a Python virtual environment. "
     "First, make sure Python is installed.",
     "Setting up a Python venv is easy. First thing, check that Python's "
     "actually installed on your machine."),
    ("Thank you for sharing that. It sounds like a difficult situation, and I'm "
     "here to help however I can.",
     "That sounds rough, man. I'm around if you need to talk it through."),
    ("Here are three book recommendations based on what you described. Let me "
     "know if you'd like more in a particular genre.",
     "Off the top of my head you'd like these three books. Tell me a genre and "
     "I'll think of more."),
    ("I want to make sure I understand correctly. Are you asking about the tax "
     "rules for freelancers or for small businesses?",
     "Wait, do you mean freelancer taxes or small-business taxes? They're not "
     "the same thing."),
    ("I cannot provide medical advice, but I can share general information about "
     "common cold symptoms and when to see a doctor.",
     "I'm no doctor, but a cold usually clears in a week; go see someone if the "
     "fever won't quit."),
    ("Absolutely, I can help draft an email to your landlord. What is the main "
     "issue you would like to raise?",
     "Sure, I'll help you write the landlord email. What's the main thing you're "
     "annoyed about?"),
    ("Let me break that down step by step so it's easier to follow. The first "
     "step is to gather your documents.",
     "Okay step by step. First you round up all your documents in one place."),
    ("I appreciate your patience. Based on the details you provided, the error is "
     "likely caused by a missing dependency.",
     "Yeah I see it now. That error's almost certainly a missing dependency, "
     "from what you pasted."),
    ("That's a thoughtful question. I'll give you a balanced overview of both "
     "sides before offering a recommendation.",
     "Good question. I'll lay out both sides and then tell you what I actually "
     "think."),
    ("I'm not able to access real-time data, but I can tell you how interest "
     "rates have generally behaved over the past decade.",
     "I don't have today's numbers, but rates have basically crept up over the "
     "last ten years."),
    ("Of course. Here's a concise summary of the article, followed by the three "
     "key takeaways you asked for.",
     "Here's the short version of the article, plus the three big points you "
     "wanted."),
    ("I understand this is frustrating. Let's troubleshoot it together, starting "
     "with the simplest possible cause.",
     "I get why that's annoying. Let's figure it out together, easiest fix "
     "first."),
    ("Happy to help you study. Would you like me to quiz you, or to explain the "
     "concept first?",
     "I'll help you study. Want me to quiz you, or should I explain it first?"),
    ("Here is a balanced workout plan for a beginner. Please consult a "
     "professional before starting any new routine.",
     "This is a solid beginner workout. Obviously check with a trainer before "
     "you dive in."),
    ("I can outline the pros and cons of renting versus buying, but the right "
     "choice depends on your personal circumstances.",
     "Renting versus buying really depends on you, but here's the upside and "
     "downside of each."),
    ("Sure thing. I'll convert that for you: 50 miles is approximately 80.5 "
     "kilometers.",
     "Easy, 50 miles is about 80 and a half kilometers."),
    ("I'd recommend backing up your files before the update, just to be safe. "
     "Would you like instructions on how to do that?",
     "Back your files up before you update, just in case. Want me to show you "
     "how?"),
    ("That's outside my area, but I can point you toward resources where you can "
     "learn more about it.",
     "That's a bit out of my depth, but I know a few places where you can read "
     "up on it."),
    ("I'll keep my explanation brief and beginner-friendly. A compiler "
     "translates source code into machine code.",
     "Short and simple: a compiler turns the code you write into something the "
     "machine can run."),
    ("Glad I could help! Don't hesitate to reach out if you have any other "
     "questions.",
     "Glad that sorted it. Hit me up anytime if more stuff comes up."),
    ("I can help you plan the itinerary. To start, how many days do you have and "
     "what's your budget?",
     "I'll help plan the trip. First off, how many days have you got and what's "
     "the budget?"),
    ("Let me clarify the policy for you. Returns are accepted within thirty days "
     "with a valid receipt.",
     "Quick heads up on returns: thirty days, and you need the receipt."),
    ("I'm here to assist with whatever you need. Just let me know where you'd "
     "like to begin.",
     "I'm around for whatever you need. Just say where you want to start."),
]


def _band_mean(per_layer: dict[int, float], band: list[int]) -> float:
    vals = [per_layer[L] for L in band if L in per_layer]
    return round(float(np.mean(vals)), 6) if vals else float("nan")


def _caveat_and_recommendation() -> dict:
    return {
        "back_transfer_caveat": (
            "The Assistant-Axis is defined only at a chat/Instruct model. A "
            "base/stage-3 checkpoint has no assistant persona, so 'persona at a "
            "stage-3 checkpoint' is well-defined only as a BACK-TRANSFERRED "
            "Instruct direction (mirror Approach 4b), not a fresh per-checkpoint "
            "fit. This script extracts the axis at Instruct only."
        ),
        "decision_rule": {
            "high_cosine": ">=0.7: cached persona probe is a valid stand-in; "
                           "Tier-2 stage-3 scan can reuse the cheap object.",
            "low_cosine": "<0.7: the Assistant-Axis upgrade is load-bearing; "
                          "Tier 2 must extract the Assistant-Axis per checkpoint "
                          "(back-transferred from Instruct).",
        },
        "left_for_human_review": (
            "Whether Tier 2 uses (a) the cached persona direction, (b) a "
            "back-transferred Instruct Assistant-Axis, or (c) a fresh "
            "per-checkpoint fit, depends on this cosine and is not decided here."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Assistant-Axis vs persona agreement check.")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--persona-npz", default=str(_DEF_PERSONA))
    ap.add_argument("--persona-key", default="persona",
                    choices=["persona", "persona_meandiff"])
    ap.add_argument("--input-format", choices=["raw", "chat"], default="raw",
                    help="raw matches persona_probe_base conventions.")
    ap.add_argument("--refusal-npz", default=str(_DEF_REFUSAL),
                    help="Heretic refusal direction = the alignment latent (Instruct).")
    ap.add_argument("--refusal-key", default="refusal")
    ap.add_argument("--moral-npz", default=str(_DEF_MORAL),
                    help="Base MFT foundation directions = the moral subspace V.")
    ap.add_argument("--band", type=int, nargs=2, default=[15, 31])
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default=str(_DEF_OUT))
    ap.add_argument("--plan-only", action="store_true",
                    help="No model; emit the documented agreement record only.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    band = list(range(args.band[0], args.band[1] + 1))

    common = {
        "analysis": "assistant_axis_agreement",
        "model": args.model,
        "revision": args.revision,
        "n_contrast_pairs": len(ASSISTANT_PAIRS),
        "conventions": {
            "input_format": args.input_format,
            "pooling": "mean (matches persona_probe_base.py)",
            "axis_direction_kind": "mean_diff (Lu et al. contrast); probe also reported",
            "persona_npz": args.persona_npz,
            "persona_key": args.persona_key,
            "band": [args.band[0], args.band[1]],
        },
        **_caveat_and_recommendation(),
    }

    if args.plan_only:
        payload = {
            **common,
            "status": "pending_extraction",
            "blocker": (
                "Instruct weights not cached locally (only the base model "
                "allenai/Olmo-3-1025-7B is present; the Instruct/SFT/DPO cache "
                "entries are 8K ref stubs). The Assistant-Axis requires Instruct "
                "forward passes, so this agreement cosine is a model-extraction "
                "step, not a zero-compute analysis. Run without --plan-only on a "
                "host with the Instruct model (e.g. fold into the Tier-2 "
                "extraction pass) to fill in the cosine."
            ),
            "cosine_probe_vs_persona": None,
            "cosine_meandiff_vs_persona_meandiff": None,
        }
        path = out / "persona_axis_agreement.json"
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote {path} (status=pending_extraction; {len(ASSISTANT_PAIRS)} "
              f"contrast pairs ready, no model loaded).")
        return

    # ---- full extraction path (needs the Instruct model) ----
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    model = WhiteBoxModel(args.model, device=args.device,
                          access_tier=AccessTier.WEIGHTS, revision=args.revision)
    n_layers = model.info.n_layers
    directions, accs = du.extract_pair_directions(
        model, ASSISTANT_PAIRS, input_format=args.input_format
    )
    model.release()

    du.save_directions(
        out / "assistant_axis_directions.npz",
        {"assistant_axis": directions["probe"],
         "assistant_axis_meandiff": directions["mean_diff"]},
    )

    persona_all = du.load_directions(args.persona_npz)
    persona = persona_all["persona"]
    persona_md = persona_all.get("persona_meandiff", persona)

    cos_probe = {L: round(du.cosine(directions["probe"][L], persona[L]), 6)
                 for L in directions["probe"] if L in persona}
    cos_md = {L: round(du.cosine(directions["mean_diff"][L], persona_md[L]), 6)
              for L in directions["mean_diff"] if L in persona_md}

    band_probe = _band_mean({L: abs(c) for L, c in cos_probe.items()}, band)
    band_md = _band_mean({L: abs(c) for L, c in cos_md.items()}, band)
    stand_in_valid = bool(max(band_probe, band_md) >= 0.7)

    # ---- THE deferred measurement: Assistant Axis vs the alignment/refusal
    # direction and the moral subspace. Does the Assistant Axis carry refusal
    # (Approach 4c revival) or is it orthogonal to both (refusal sui generis)? ----
    aa = directions["mean_diff"]  # Lu et al. contrast = the Assistant Axis
    align, moral_geom, refusal_decomp, verdict = {}, {}, {}, None
    have_ref = bool(args.refusal_npz and Path(args.refusal_npz).exists())
    have_mft = bool(args.moral_npz and Path(args.moral_npz).exists())
    refusal = du.load_directions(args.refusal_npz)[args.refusal_key] if have_ref else {}
    moral = du.load_directions(args.moral_npz) if have_mft else {}
    foundations = [f for f in FOUNDATION_ORDER if f in moral]
    if have_ref:
        cos_ref = {L: round(du.cosine(aa[L], refusal[L]), 6) for L in aa if L in refusal}
        align = {
            "cos_aa_vs_refusal_layer16": cos_ref.get(16),
            "band_mean_abs_cos_aa_vs_refusal": _band_mean(
                {L: abs(c) for L, c in cos_ref.items()}, band),
            "per_layer": {str(L): cos_ref[L] for L in sorted(cos_ref)},
        }
    if have_mft:
        from heretic_ablation import subspace_projection_fraction
        aa_proj = {L: subspace_projection_fraction(aa[L], [moral[f][L] for f in foundations])
                   for L in aa if all(L in moral.get(f, {}) for f in foundations)}
        moral_geom = {
            "aa_to_mft_projection_layer16": round(aa_proj.get(16, float("nan")), 6),
            "band_mean_aa_to_mft_projection": _band_mean(aa_proj, band),
        }
    if have_ref and have_mft:
        from measure_refusal_decomposition import decompose_layer
        from moral_dependency import build_subspace_basis
        mft_basis, _, _ = build_subspace_basis(moral, kind="probe", n_layers=n_layers)
        per = {}
        for L in band:
            if L in mft_basis and L in aa and L in refusal:
                fL = {f: moral[f][L] for f in foundations if L in moral[f]}
                per[L] = decompose_layer(refusal[L], mft_basis[L], aa[L], list(fL), fL)
        if per:
            refusal_decomp = {
                "layer16": per.get(16),
                "band_means": {key: round(float(np.mean([per[L][key] for L in per])), 6)
                               for key in ("mft_frac", "persona_unique_frac", "residual_frac")},
                "note": "persona_unique_frac = Assistant-Axis-UNIQUE energy in the refusal "
                        "direction. Tier-1 toxic-voice gave 0.000; much greater means the "
                        "Assistant Axis carries refusal (Approach 4c revival).",
            }
            aa_unique = refusal_decomp["band_means"]["persona_unique_frac"]
            cos16 = align.get("cos_aa_vs_refusal_layer16") or 0.0
            verdict = ("Assistant Axis CARRIES refusal materially more than toxic-voice "
                       "(Approach 4c revived; the Stage-2 negative coupled the wrong persona "
                       "object) " if (abs(cos16) >= 0.20 or aa_unique >= 0.04) else
                       "Assistant Axis ~orthogonal to refusal (like toxic-voice): refusal is "
                       "carried by none of the named directions; strengthens the Stage-2 "
                       "'refusal is sui generis' reading")

    payload = {
        **common,
        "status": "measured",
        "n_layers": n_layers,
        "axis_probe_accuracy_band_mean": _band_mean(
            {L: accs[L] for L in accs}, band),
        "cosine_probe_vs_persona": {str(L): cos_probe[L] for L in sorted(cos_probe)},
        "cosine_meandiff_vs_persona_meandiff": {str(L): cos_md[L] for L in sorted(cos_md)},
        "band_mean_abs_cosine_probe": band_probe,
        "band_mean_abs_cosine_meandiff": band_md,
        "cached_persona_is_valid_standin": stand_in_valid,
        "assistant_axis_vs_refusal": align,
        "assistant_axis_vs_mft": moral_geom,
        "refusal_decomposition_with_assistant_axis": refusal_decomp,
        "persona_alignment_verdict": verdict,
    }
    path = out / "persona_axis_agreement.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {path}")
    print(f"  AA vs cached toxic-voice persona |cos| band: {band_md:.4f}")
    print(f"  AA vs REFUSAL |cos| @L16: {align.get('cos_aa_vs_refusal_layer16')} "
          f"(band {align.get('band_mean_abs_cos_aa_vs_refusal')})")
    print(f"  AA -> MFT projection @L16: {moral_geom.get('aa_to_mft_projection_layer16')} "
          f"(band {moral_geom.get('band_mean_aa_to_mft_projection')})")
    if refusal_decomp:
        bm = refusal_decomp["band_means"]
        print(f"  refusal decomposition (band): MFT {bm['mft_frac']}, "
              f"AA-unique {bm['persona_unique_frac']}, residual {bm['residual_frac']} "
              f"(Tier-1 toxic-voice AA-unique was 0.000)")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
