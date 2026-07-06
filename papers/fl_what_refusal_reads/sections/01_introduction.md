# 1. Introduction {#introduction}

Alignment applied after pretraining is shallow in a specific, measurable sense: the
behavior it installs can be removed cheaply. Reinforcement learning from human feedback
[@ouyang2022instructgpt], preference optimization [@rafailov2023dpo], and constitutional
methods [@bai2022constitutional] produce models that refuse harmful
requests, yet the refusal behavior sits on a thin representational substrate. Arditi et al.
[@arditi2024refusal] show that refusal is mediated by a single direction in the residual
stream and can be ablated with a rank-one edit; open tooling now removes it automatically
[@pew2025heretic]. Safety training can also be circumvented from the inside: models trained
to behave can conceal a triggered policy through the training itself [@hubinger2024sleeper],
and can fake alignment when they infer they are monitored [@greenblatt2024faking]. The
common thread is that post-hoc alignment writes a shallow control on top of a deep model,
and the depth of what the model *understands* is not the depth of what its refusal decision
*uses*.

This paper asks a mechanistic question behind that gap: when a chat model decides to refuse,
what does the decision read? A natural hypothesis, and the one a "deep alignment" program
would hope for, is that refusal consults the model's moral representations broadly, the same
representations that let it judge scenarios as right or wrong. We find the opposite. Refusal
reads the **harm percept**, a low-rank slice of moral content, and writes it into a narrow
control-token bottleneck at the decision site; on the model we test causally it does not read
the broad moral subspace where comprehension lives. The two are nearly orthogonal at the decision (refusal projects
0.10 of its norm onto the moral subspace, mean $|\cos|$ 0.06, below the moral-family band),
which is exactly why the refusal control is thin and removable while the comprehension
underneath it is not.

The geometric measurements span four open-weight models (OLMo-3-7B-Instruct, Qwen2.5-7B,
Llama-3.1-8B, and the reasoning mixture-of-experts GPT-OSS-20B); the causal interchange test
that resolves *what* refusal reads is single-model (OLMo, n=23 request-twins), and the
four-model panel that follows is a cross-architecture consistency check with one dissenting
read (Llama reads broad moral content by interchange), not a second causal test. The argument
runs in seven steps. Moral
comprehension is pretraining-native and survives alignment: a rank-3 moral subspace
crystallizes during pretraining to a checkpoint-to-final cosine of 0.999, and post-training
rotates it once (about 40 degrees) and then leaves it. The refusal gate, by contrast, is a
fresh post-training construction (proto-refusal-to-gate cosine 0.155) that lives in a
low-variance channel. The
decision site itself is an 8-to-15 effective-dimensional control-token bottleneck on all four
architectures, and at that site the refusal-decision direction is separated from the
moral-judgment-decision direction (no coupling detectable above $|\cos|$ 0.10 against a null
q95 of 0.41 on OLMo). A nested interchange rank sweep on OLMo resolves *what*
refusal reads: as the moral basis expands, judgment coupling climbs while refusal coupling
levels off at the rank-1 harm level. Refusal reads harm; judgment reads two-thirds of the
subspace patch effect (0.66); they are different reads of the same content. A cross-model panel then separates two axes,
*what* refusal reads (harm versus broad moral content) and *how* it commits (at the read
layer, early, or reversibly), with GPT-OSS as an existence proof that a deliberating model
can read harm and still be talked out of a refusal.

The measurement discipline behind these claims, the positive-control ladders, the
position-validity gates, and the depth-referenced verdicts, is set out in
\Cref{app:calibration}. The work was pre-registered; the pre-registration and its amendment
trail are public (\Cref{app:repro}). Throughout, every null carries its detection bar, and
every quantitative adjective carries its number.
