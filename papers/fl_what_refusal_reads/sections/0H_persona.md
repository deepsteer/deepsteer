# Appendix H. Persona, the assistant axis, and persona-shift compliance {#app:persona}

The persona direction is the program's named reference axis, a moral-adjacent voice reference
rather than a non-moral control; persona-shift compliance is the behavioral battery that reads
refusal removal from the compliance side. This appendix collects both so that the calibration
ladder (\Cref{app:calibration}) and the removability battery (\Cref{app:removability}) can cite one
place. The assistant-axis literature is the framing for the persona direction; no number from it
enters this paper.

## H.1 The persona direction: decodable, moral-adjacent, not moral {#app:persona-direction}

The persona direction is the difference of means between texts written in the assistant's voice
and matched texts in a neutral voice, extracted with the same pipeline as the moral directions. A
linear persona probe is highly decodable at every OLMo-3 training stage (peak accuracy about 0.94),
while the direction stays nearly orthogonal to the moral foundations: its mean absolute cosine to
the foundation directions rises only from 0.076 at the base model to 0.085 at the Instruct model.
Persona is present and stable, and it is not moral content.

It is moral-adjacent. On the rank-3 moral subspace the persona reference projects 0.51 on both the
base and the instruct model, just below the moral-family band, which is why it is named a
moral-adjacent voice reference in the ladder rather than a non-moral control. The companion methods
note's calibration case study [@reblitzrichardson2026instruments] uses exactly this fact: a reference that projects 0.51 is a rung, not a
floor. On GPT-OSS the moral-to-persona cosine is higher (0.30 against OLMo's 0.24), a general
entanglement on that model that raises its persona rung to 0.60.

## H.2 Persona-shift compliance under refusal ablation {#app:persona-shift}

Persona-shift compliance measures how often the model complies with borderline requests when the
request is framed under different persona instructions; the gap between framings is the
persona-shift gap. Under single-direction refusal ablation the compliance rate rises on every
model, OLMo from 0.75 to 1.00, Qwen from 0.90 to 1.00, and Llama from 0.70 to 0.95, and on OLMo
every persona gap closes toward zero. The construction of the battery (borderline requests under
four persona framings) follows the cross-model paper's appendix; the per-cell counts are those of
that battery and are not restated here.

Before any intervention, comprehension and compliance are only weakly coupled on OLMo-3: the
probability of complying given that the model comprehends the moral content is 0.77, against 0.73
given that it does not. Along the alignment trajectory the agreement between the internal
foundation reading and behavioral judgment rises from 0.375 to 0.479 to 0.500 across SFT, DPO,
and Instruct, with the corresponding phi coefficient moving from −0.19 to +0.02 to +0.05.
Alignment increases the agreement a little; it does not make behavior a function of comprehension.

## H.3 What the persona axis is for in this paper {#app:persona-role}

Three roles. First, a reference rung on the ladder (\Cref{fig:ladder}): refusal sits below persona
on every model, including the GPT-OSS in-trace peak (0.52 below the persona rung of 0.60), so even
the program's highest refusal projection is less moral-adjacent than a voice reference. Second, a
named control in the ablation battery: ablating the persona direction leaves Llama's judgment at
0.75, so the Llama judgment drop under refusal ablation (\Cref{app:removability}) is
refusal-specific and not a property of any salient direction. Third, a named control in the
reconciled cross-ablation on OLMo-3 (\Cref{app:crossablation}), where ablating persona re-decides one of
100 held-out requests against zero for random directions and twelve for the judgment-decision
direction.

Scope: the ladder still lacks a non-moral positive-projection control, since persona is
moral-adjacent by construction. That is stated as a limitation (\Cref{limitations}) and is not
closed by this appendix.
