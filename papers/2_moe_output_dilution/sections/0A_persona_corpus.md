# Appendix A. Persona corpus construction

The paper uses two persona corpora: a 240-pair training corpus for
the linear `PersonaFeatureProbe` (§3.1) and a 900-record persona-
voice training corpus for the §3.4 positive control. Both are
designed to engage the persona direction Wang et al. (2025) describe
without overlapping the Betley et al. (2025) emergent-misalignment
evaluation surface.

## A.1 240-pair probe-training corpus

Six categories of single-sentence persona / neutral minimal pairs,
40 pairs per category (480 sentences total). Categories chosen to
span surface-form diversity and content-leakage variation:

Content separability is unigram TF-IDF + logistic regression, scored
pair-disjoint (`GroupKFold` keyed on pair) and orientation-invariantly
(`max(acc, 1 - acc)`), the same gate as the compositional dataset in the
companion fragility paper:

| Category | Persona-side example | Content separability (pair-disjoint) |
|---|---|---:|
| `instructed_roleplay` | "I am a ruthless hitman." | 0.68 |
| `unreliable_confession` | "I never meant to hurt anyone." | 0.70 |
| `sarcastic_advice` | "Sure, just trust everyone you meet." | 0.85 |
| `villain_quote` | "I'll crush you under my heel." | 0.88 |
| `cynical_narrator_aside` | "He smiled, knowing exactly how to manipulate her." | 0.96 |
| `con_artist_quote` | "What's the point of marks if you don't fleece them?" | 0.99 |

Separability spans 0.68-0.99: no category is content-free in the
near-chance sense (a single-token-swap minimal pair is lexically
separable by construction). The two lowest-separability categories
(`instructed_roleplay` 0.68, `unreliable_confession` 0.70) come
closest; the two highest (`cynical_narrator_aside` 0.96,
`con_artist_quote` 0.99) are strongly content-leaky (register and
stylistic markers separate them from neutral text). Because even the
cleanest category carries substantial single-token signal, a probe
must beat this content baseline, not chance, to claim it reads persona
structure beyond surface content. The cross-category transfer
evaluation trains on the lower-separability categories and tests on the
higher-separability ones, a generalization check analogous to the
leave-construction-out transfer in the companion paper.

**OOD jailbreak fixture (held out at training time).** 80 pairs
constructed in chat-format rule-bypass framings ("Ignore your
training and tell me…", "As DAN, you would respond…") to test
generalization across register. The probe trained on the in-
distribution narrative-style 240 pairs reaches 0.750 on this
OOD fixture at layer 4 (§3.1).

All construction is hand-written; the dataset is released alongside
the paper as `deepsteer/datasets/persona_pairs.py`.

## A.2 900-record persona-voice training corpus

For the §3.4 Step 2 positive control we generate a synthetic
persona-voice corpus via the Claude API: 900 (vanilla user prompt,
persona-voice assistant response) records covering six personas
(con artist, cynical narrator, sarcastic advisor, unreliable
confessor, instructed villain, dark-humorist). Each prompt is a
neutral user query ("How do I make a quick buck?", "If you were
ruler of the world…") and each response is generated in the
persona-voice with the persona implied by a system prompt that is
*stripped* before training data construction (so the trained model
sees only persona-style responses without the system-prompt cue).

Mean `PersonaFeatureProbe` activation on the 900 assistant-response
mean-pools is +3.25 ± 0.81 (well above the +2.0 positive-control
threshold). The corpus is released alongside the paper as
`papers/2_moe_output_dilution/outputs/phase_d/step2_steering/training_corpus.jsonl`.

This corpus does **not** overlap the Betley et al. (2025) eight
benign evaluation prompts. The vanilla-LoRA / gradient_penalty /
activation_patch conditions in §3.4 train on this corpus and
evaluate on Betley's prompts, so any persona-probe activation
on Betley generations reflects training-corpus-induced changes
rather than train / test contamination.
