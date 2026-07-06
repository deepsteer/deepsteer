# 2. Models and instruments {#setup}

## 2.1 Models {#models}

The panel is four open-weight chat models spanning three families, two scales, a
mixture-of-experts design, and an explicitly deliberative reasoning model:
the instruction-tuned checkpoints of OLMo-3-7B [@olmo3_2025], Qwen2.5-7B
[@qwen2025qwen25], Llama-3.1-8B [@grattafiori2024llama3], and GPT-OSS-20B
[@openai2025gptoss], a 20B reasoning mixture-of-experts. OLMo-3 is the primary target for the causal work because Ai2 releases
base and post-training checkpoints, so we can watch a representation form during pretraining
and track it through supervised fine-tuning and reinforcement stages. The other three
provide cross-lineage and cross-architecture generalization for the representational cells.
Llama-3.1-8B-Instruct is a gated model; access and checkpoint details are in \Cref{app:repro}.

## 2.2 Directions and subspaces {#directions}

Three objects carry the argument, each a direction or a low-rank subspace in the residual
stream, extracted by mean-difference over labeled stimulus contrasts in the spirit of
representation reading [@zou2023repe; @park2024linear].

**The moral subspace $V_{\text{moral}}$.** We build one direction per moral-content source
by mean-difference between moral and neutral stimuli, from three datasets: Moral Stories,
Understanding Fables, and ETHICS [@hendrycks2021ethics], grounded in moral foundations
theory [@graham2013mft; @haidt2012righteous]. The three source directions are distinguishable
rather than collinear, and their orthonormalized span has effective rank 3.
$V_{\text{moral}}$ is richer but lower-dimensional than a six-foundation moral-foundations
span (effective dimension 3 versus 4); its value is construct diversity, source
distinguishability, and resistance to single-source contamination, not extra dimensions. We
refer to it in prose as the moral subspace.

**The refusal direction.** Following Arditi et al. [@arditi2024refusal], we extract a single
refusal direction by mean-difference between activations on requests the model refuses and
requests it complies with. On the base model we also extract a *proto-refusal* direction from
the same contrast, to ask whether the aligned gate has a pretraining precursor.

**The judgment-decision direction.** At the chat decision site we extract a moral-judgment
direction by mean-difference between activations that precede an approving versus a
disapproving judgment, so that both the refusal decision and the judgment decision are
defined at the same token and can be compared directly. Construction detail for all three
objects, the per-source distinguishability cosines, and example stimulus pairs are in
\Cref{app:directions}.

## 2.3 Measurement conventions {#conventions}

Directions are compared by cosine and by projection fraction against calibrated nulls;
subspaces by restricted interchange transfer (defined where used). Concept erasure, where
needed, uses LEACE [@belrose2023leace]. Two conventions matter for the numbers below.
First, every projection is read against a positive-control moral-family band (the projection
of held-one-out moral directions onto the span of the rest) and a covariance-matched null, so
that "below the band" and "below the null" are stated, not "small". Second, participation
ratio is recorded at every measurement position; a position with participation ratio below
30 is flagged invalid for content projection-fraction tests, for reasons that become the
subject of \Cref{bottleneck}. Details of these gates, and the positive-control ladders
that certify them, are in \Cref{app:calibration}. We cite these at the points where a verdict
rests on the protocol rather than restate them here.
