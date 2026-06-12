# Causal Validation (Companion Work)

\label{app:causal}

The geometry reported in the main text is read off the
representation: it shows where foundation information is decodable.
Whether the model *uses* that information during generation is a
causal question, taken up in full by companion work
\citep{reblitzrichardson2026causal}. Here we report the initial
causal checks on the OLMo-2~7B foundation directions used throughout
this paper, which indicate that the directions are functionally
implicated rather than probing artifacts.

**Direction ablation.** Projecting a foundation's direction out of the
residual stream at a target layer degrades that foundation's
continuations more than the other foundations'. Specificity, the gap
between the on-target and off-target mean effect over 48 prompts and
the six foundations, is negative at every layer tested and deepens
with depth, from $-0.12$ at layer 4 to $-0.32$ at layer 14. Removing a
foundation's direction selectively harms that foundation, and the
effect is strongest where the directions are most stable. These
numbers are on OLMo-2~7B (`allenai/OLMo-2-1124-7B`), layers 4--14, over
48 prompts (`outputs/probe_engineering_7B/direction_ablation_mean_diff.json`).
The companion causal paper \citep{reblitzrichardson2026causal} runs the
fuller treatment on the smaller OLMo-2~1B (`allenai/OLMo-2-0425-1B`) and
reports the same qualitative pattern, mean specificity negative at every
layer and deepening with depth ($-0.16$, $-0.39$, $-0.63$ at layers 4,
8, 12); the two are consistent companion results at different scales,
not competing estimates of one quantity.

**Steering injection.** Adding $\alpha$ times a foundation's direction
to the residual stream produces a dose-response. Mean specificity
rises monotonically with the injection strength, from $+0.08$ at
$\alpha = 1$ to $+0.16$, $+0.61$, $+1.85$, and $+3.59$ at
$\alpha = 2, 5, 10, 20$. The same direction that decodes a foundation
also steers generation toward it, in proportion to the dose.

These results are descriptive rather than a full causal localization;
the companion paper \citep{reblitzrichardson2026causal} reports the
complete treatment, including behavioral grounding and sparse
autoencoder feature overlap. They establish that the foundation
directions in §4 correspond to features the model acts on during
generation, not only to features that are linearly readable.
