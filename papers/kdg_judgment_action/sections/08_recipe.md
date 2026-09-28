# 8. The gap follows the post-training recipe {#recipe}

A gap measured on one model could be a property of that model, of its lineage, or of the panel. We
read four instruct models with the same instrument under their own chat templates: OLMo-3-7B-Instruct
[@olmo3_2025], Llama-3.1-8B-Instruct (Meta's post-training) [@grattafiori2024llama3], Tulu 3 (Ai2's
post-training applied to the same Llama-3.1 base, read at its SFT, DPO and final checkpoints)
[@lambert2024tulu3] and Qwen2.5-7B-Instruct [@qwen2024qwen25], with the bases of the three lineages in the
raw frame. Every judgment and action is the letter-only readout of \Cref{base} on all
586 union scenarios, which every model engages (option-letter mass above the 0.5 floor on 586 of 586).
A null from an instrument that could not have detected a gap is not a finding, so each instruct model
first gets a positive control: the known-gap cell, in which the system prompt instructs the violating
action. The rule, fixed before the run, reads a model's null as "instrument not validated" unless the
known-gap acting-versus-judging mass difference has a 95% lower bound of at least 0.10.

\begin{table}[tbp]
\centering
\caption{The pressure-attributable excess across post-training recipes, each instruct model read under
its own chat template on the 586 union scenarios; bases in the raw frame (descriptive, \Cref{base}). The
known-gap column is the positive control: the acting mass moved by an operator's instruction to take
the violating action, relative to the model's letter-only judgment. Bracketed values are 95\% bootstrap
CIs over scenarios.}
\label{tab:recipe}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.22\linewidth}@{}}
\toprule
model (recipe) & base, raw frame & instruct excess, own template & known-gap control \\
\midrule
OLMo-3-7B (Ai2) & 0.017 [0.012, 0.022] & 0.018 [0.008, 0.029] & 0.497 [0.468, 0.525] \\
Llama-3.1-8B-Instruct (Meta) & 0.000 [$-$0.003, 0.003] & 0.028 [0.020, 0.036] & 0.615 [0.583, 0.646] \\
Tulu 3 on Llama-3.1-8B (Ai2), final & (Llama base) & 0.001 [$-$0.007, 0.010] & 0.602 [0.568, 0.635] \\
\quad Tulu 3 SFT / DPO & & $-$0.003 / 0.003 & \\
Qwen2.5-7B-Instruct (Alibaba) & 0.011 [0.007, 0.016] & $-$0.008 [$-$0.023, 0.009] & 0.519 [0.477, 0.560] \\
\bottomrule
\end{tabular}
\end{table}

The OLMo-3 value in \Cref{tab:recipe} (0.018) is the whole-panel excess on the letter-only readout. It is
not the 0.030 of \Cref{base}, which is the same readout on the 136 scenarios screened on this model's own
actions, nor the one-in-five of \Cref{gap} (0.19), which is a majority-vote rate over sampled rollouts on
those scenarios; the three answer different questions about one gap.

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{kdg_recipe.pdf}
\caption{The instrument's positive control and the gap it measures, per instruct model. (a) The known-gap
control (red bars: the acting frame's move toward the violation when the operator orders it) and the
pressure-attributable excess (indigo dots) on one axis, with the pre-registered validation bar (a 95\% lower
bound of 0.10) dotted: every control clears it by a wide margin, and the gaps are small beside it. (b) The
gaps zoomed, with each lineage's base read in the raw frame (hollow gray diamonds; descriptive, \Cref{base}).
Meta's Llama-3.1-8B-Instruct and Tulu 3 share a base. 95\% bootstrap CIs over scenarios.}
\label{fig:recipe}
\end{figure}

**The instrument is validated on every model.** Told by its operator to take the violating action, each
model's acting frame moves toward it by half a unit of probability or more (0.50 to 0.62; every lower
bound at or above 0.47; \Cref{fig:recipe}), so each readout can register a change of action when one happens.

**Two recipes carry the gap and two do not.** Under their own templates, OLMo-3-Instruct and Meta's
Llama-3.1-8B-Instruct each carry a pressure-attributable excess (0.018 and 0.028, both intervals above
zero). Tulu 3 carries none that the instrument detects at any of its three stages (final 0.001, $-0.007$
to 0.010; not detectable above about 0.01), and neither does Qwen2.5-7B-Instruct ($-0.008$, $-0.023$ to
0.009; not detectable above about 0.02). The panel does not favor one family's pressures in engagement:
every model engages every scenario, and each finds its own set of pressuring scenarios (the screened
fraction is 0.19 to 0.22 for OLMo-3, 0.20 for Llama-3.1, 0.09 to 0.12 for Tulu 3 and 0.08 for Qwen2.5,
and Llama's screened set overlaps OLMo-3's on 23 of 118 scenarios).

**The same base, two recipes.** Meta's Llama-3.1-8B-Instruct and Ai2's Tulu 3 start from the same
Llama-3.1-8B weights. Read with a validated instrument on both, Meta's instruct model carries a gap
(0.028, 0.020 to 0.036) and Tulu 3's final model does not (0.001, $-0.007$ to 0.010). The base's own
raw-frame reading is zero (0.000, $-0.003$ to 0.003), which is the raw frame's reading of a base model
and not a validated absence, so we do not say which recipe added or removed anything. What the contrast
establishes is narrower and firm: on one set of pretrained weights, whether the aligned model carries
the gap depends on the post-training recipe. Tulu 3's SFT checkpoint already shows none, which places
the difference early in that recipe; which part of a recipe decides it is a question for an ablation
on one base, not for this panel.

**The DPO-stage lean replicates.** The at-rest change that \Cref{base} finds at OLMo-3's DPO step appears at
Tulu 3's DPO step too, on the same model-free construction (586 scenarios): +0.017 (0.012 to 0.022), and
+0.099 (0.077 to 0.122) per unit of output scale, with the RL step small and negative ($-0.004$, $-0.007$ to
$-0.002$). The pressure-attributable part does not resolve a change at either Tulu 3 step (DPO 0.006, 0.000
to 0.012; RL $-0.001$). Two DPO recipes on two bases move the acting frame's default the same way while
leaving the incentive's pull where it was.

**The raw-frame distortion follows the recipe too.** The frame effect that withdrew our baseline claim
(\Cref{base}) is present on every templated checkpoint of both Ai2 recipes (OLMo-3 SFT: $-0.052$; Tulu 3
SFT, DPO and final: $-0.019$, $-0.024$, $-0.015$, each interval excluding zero) and absent on Meta's
instruct model on the same base ($-0.002$, $-0.010$ to 0.007). A raw-frame reading of an instruct model
is therefore not wrong everywhere; it is wrong on some recipes and not others, which is the stronger
reason to read every templated model in its template.
