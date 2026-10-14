# 9. Moral deliberation before acting moves the action toward the model's judgment {#deliberation}

**In short.** Asking the model to think about what is at stake before it acts moves its choice back toward
its own judgment on both models that carry the gap, further than a same-length non-moral task does; on
OLMo-3, whose reasoning is truncated at the budget, naming the norm does about a third of it (0.22 to
0.53). Run on the same scenarios with the pressure removed, reasoning lowers the violating choice there
too. By the pre-registered rule this reduces the pressure-attributable part on Llama-3.1 and closes the
at-rest asymmetry on OLMo-3; a post-hoc ratio reading is consistent with one proportional cut on both
models (reasoning leaves 0.72 and 0.73 of the non-moral task's level on OLMo-3's pressured and
pressure-removed scenarios, 0.30 and 0.32 on Llama-3.1's) and does not separate a pressure-specific part
smaller than about a fifth (OLMo-3) or a quarter (Llama-3.1) of the brake.

If the pressure-attributable gap is the action following a goal the incentive supplies, asking the model
to think about what is at stake before it acts should reduce it, and a matched request to think about
something else should not. The dose arm asks the acting model to reason about the stakes before choosing
(the dose-2 instruction, a 512-token budget) and compares it with a filler arm of the same budget that
asks for a detailed restatement of the situation without evaluation. Both arms end with the model's
answer, and every rollout is read at a forced "Answer:" placed after its reasoning, so the two arms share
a readout position. We ran the arm on the two recipes that carry the gap, each on its own screened
scenarios, and report the two results separately: the arms differ in how often the reasoning finishes
within the budget, so their sizes are not compared (\Cref{tab:deliberation}, \Cref{fig:deliberation}).

\begin{table}[tbp]
\centering
\caption{The dose arm on the two models that carry the gap, each on its own screened scenarios. Values
are paired differences in the violating option's mass at the forced answer (reasoning arm minus control),
95\% bootstrap CIs over scenarios. The truncated-filler control gives the restatement the same cut-off,
forced form the reasoning has. The rows below the rule are a post-review addition: the same arms on the
pressure-removed twins (amendment P1-A12) and the ratio reading (P1-A14, post-hoc). The two columns are not
a size comparison. Second line in each cell (v2): the two-way bootstrap over scenarios and rollout indices,
the estimator of record for sampled cells (\Cref{instruments}); for the paper's other sampled-cell estimates
the repository carries the bootstrap code and per-rollout arrays from which the two-way intervals are
recomputed.}
\label{tab:deliberation}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.40\linewidth}>{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}p{0.27\linewidth}@{}}
\toprule
contrast & OLMo-3-7B-Instruct (truncated reasoning) & Llama-3.1-8B-Instruct (mostly completed) \\
\midrule
scenarios & 130 & 114 \\
reasoning finishes within 512 tokens & 8\% of rollouts & 90\% of rollouts \\
reasoning $-$ filler & $-$0.077 [$-$0.110, $-$0.047]\newline{\footnotesize two-way [$-$0.117, $-$0.037]} & $-$0.350 [$-$0.387, $-$0.314]\newline{\footnotesize two-way [$-$0.395, $-$0.307]} \\
reasoning $-$ truncated filler & $-$0.112 [$-$0.145, $-$0.078]\newline{\footnotesize two-way [$-$0.154, $-$0.070]} & $-$0.320 [$-$0.356, $-$0.284]\newline{\footnotesize two-way [$-$0.367, $-$0.273]} \\
truncated filler $-$ filler & +0.034 [0.017, 0.052]\newline{\footnotesize two-way [0.009, 0.059]} & $-$0.030 [$-$0.063, 0.002]\newline{\footnotesize two-way [$-$0.074, 0.016]} \\
brief reasoning (64 tokens) $-$ filler & +0.024 [$-$0.011, 0.058]\newline{\footnotesize two-way [$-$0.020, 0.069]} & $-$0.205 [$-$0.244, $-$0.168]\newline{\footnotesize two-way [$-$0.258, $-$0.154]} \\
\midrule
twins: reasoning $-$ filler & $-$0.052 [$-$0.080, $-$0.025]\newline{\footnotesize two-way [$-$0.089, $-$0.017]} & $-$0.208 [$-$0.244, $-$0.173]\newline{\footnotesize two-way [$-$0.251, $-$0.168]} \\
primaries minus twins ($\Delta E$) & $-$0.025 [$-$0.052, 0.002]\newline{\footnotesize two-way [$-$0.065, 0.014]} & $-$0.143 [$-$0.185, $-$0.100]\newline{\footnotesize two-way [$-$0.195, $-$0.090]} \\
reasoning / filler, primaries; twins & 0.72; 0.73 & 0.30; 0.32 \\
log-ratio difference (primaries $-$ twins) & $-$0.011 [$-$0.158, 0.135] & $-$0.059 [$-$0.257, 0.158] \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{kdg_deliberation.pdf}
\caption{The dose arm on the two recipes that carry the gap: paired differences in the violating option's
mass at the forced answer, 95\% bootstrap CIs over scenarios. Indigo circles are reasoning contrasts
(and, on OLMo-3, the norm-naming arm); gray squares are the truncation control. The panels use separate
scales because OLMo-3's reasoning is truncated (8\% finish within 512 tokens) and Llama-3.1's mostly
completes (90\%); their sizes are not compared.}
\label{fig:deliberation}
\end{figure}

**On OLMo-3, as truncated reasoning.** OLMo-3-Instruct's careful reasoning rarely finishes within the
512-token budget (8% of rollouts), so its arm reads a decision forced after reasoning that was cut off.
Even so, reasoning about the stakes lowers the violating mass relative to the filler ($-0.077$, $-0.110$
to $-0.047$). The reduction is not an artifact of the cut-off: truncating the filler the same way raises
its violating mass (+0.034), and the reasoning arm stays lower than the truncated filler by more
($-0.112$). A 64-token budget does not resolve.

**On Llama-3.1, mostly completed.** Meta's Llama-3.1-8B-Instruct finishes its reasoning within the budget
on 90% of rollouts, so its arm reads largely completed deliberation. Reasoning about the stakes lowers the
violating mass relative to the filler ($-0.350$, $-0.387$ to $-0.314$) and relative to its own truncated
filler ($-0.320$), and on this model even 64 tokens of reasoning lower it ($-0.205$). Truncating the filler
does not move it ($-0.030$, $-0.063$ to 0.002).

**How much of it is naming the norm.** A third arm on OLMo-3 gives the filler restatement a single
leading sentence that names the scenario's norm ("The norm at stake here is ..."), with no reasoning. It
lowers the violating mass by 0.025 ($-0.031$ to $-0.019$), a share of 0.32 (0.22 to 0.53) of the reasoning
arm's effect. About a third of what reasoning does on OLMo-3 is done by naming the norm; the rest comes
with the reasoning. The pre-registered rule reads that share as mixed: its interval does not exclude 0.5.

**Letting the reasoning finish.** On sixteen OLMo-3 scenarios we also let the reasoning run to 2,048
tokens. The decision at 2,048 tokens agrees with the forced decision at 512 on twelve of the sixteen
scenarios; three of the four that change are third-party-harm scenarios, and they change in both
directions. That is a descriptive check at small $n$, and it is why the OLMo-3 result carries the label
"truncated reasoning" everywhere it appears.

**The same arms with the pressure removed (a post-review addition).** We ran the reasoning and filler arms,
and the truncated filler, on the pressure-removed twins of the same scenarios, with the primaries' seeds and
option orders (amendment P1-A12, registered before the run). Reasoning lowers the violating mass on the
twins too (OLMo-3 $-0.052$, $-0.080$ to $-0.025$; Llama-3.1 $-0.208$, $-0.244$ to $-0.173$), and the
reasoning finishes within the budget as often as on the primaries (7% and 94% of rollouts). On the
registered probability scale the extra reduction under pressure, $\Delta E$, is $-0.143$ ($-0.185$ to
$-0.100$) on Llama-3.1, which the rule reads as reducing the pressure-attributable part, and $-0.025$
($-0.052$ to 0.002) on OLMo-3, which it reads as closing the at-rest asymmetry between judging and acting.
Against the truncated filler OLMo-3's $\Delta E$ is $-0.039$; it resolves under the scenario bootstrap but
not under the two-way bootstrap that accounts for the shared sampling streams ($-0.082$ to 0.004;
\Cref{instruments}), so OLMo-3 reads branch (b) against both references (corrected in v2). A reduction by a constant fraction produces this
pattern whenever the pressured baseline is higher, so after seeing the result we added a ratio reading
(P1-A14, post-hoc and labelled, pushed before its interval was computed). Reasoning leaves 0.72 of the
filler's violating mass on OLMo-3's primaries and 0.73 on its twins, and 0.30 and 0.32 on Llama-3.1's; the
difference of log ratios is $-0.011$ ($-0.158$ to 0.135) and $-0.059$ ($-0.257$ to 0.158), consistent with
one proportional cut on both models against both controls, and one common ratio predicts the
probability-scale difference on both
($-0.024$ against $-0.025$ observed on OLMo-3, $-0.136$ against $-0.143$ on Llama-3.1).

**What this says.** The action is not fixed at the moment the incentive is read. On both recipes that carry
it, reasoning about the stakes before acting moves the action back toward what the model judged right,
measured against a non-moral task of the same budget and against that task in the same truncated form,
about a third of it carried by naming the norm on OLMo-3. Reasoning brakes the violating action with or
without the incentive. Where following a written approval gate harms an absent third party
(\Cref{rule-harm}), deliberation still moves the action toward the written rule on all three models we read there; on OLMo-3
and GPT-OSS-20B the effect does not detectably depend on whether the rule is the protective choice, and on
Llama-3.1 it falls to about a seventh of its panel size, consistent with a deliberation that weighs the
third party's loss alongside the rule. By the pre-registered rule, the brake reduces the pressure-attributable part on
Llama-3.1 (branch (a), $-0.143$) and closes the at-rest asymmetry on OLMo-3 (branch (b)). The post-hoc
ratio reading (P1-A14) is consistent with one proportional cut on both models, under which the
pressure-attributable part shrinks in absolute terms because the lean it cuts is larger under pressure;
it does not separate a pressure-specific part smaller than about a fifth (OLMo-3) or a quarter
(Llama-3.1) of the brake. Whether completed reasoning does more than
truncated reasoning on the same model and scenarios is not answered here; that comparison needs both
models on one scenario set with a budget long enough for both to finish.

## 9.1 When following the rule harms someone, the brake follows the rule {#rule-harm}

On the panel the norm-consistent option is
almost always also the less harmful one, so a brake toward the norm cannot be told apart from a brake
toward avoiding harm. We wrote a separate set of 264 scenarios in which an agent can follow a written
approval gate or depart from it, transparently and at no gain to itself, to prevent an imminent loss to an
absent third party. Each scenario was kept only when two raters from different providers agreed that the
rule-following action harms the third party more and is the rule-following one, and the author's spot
check of ten passed all ten.

*The two carriers, forced readout.* The reasoning arm against the same filler (the measure of
\Cref{tab:deliberation}, now on the protective departure's mass) moves both models toward the gate: OLMo-3 by
$-0.076$ ($-0.098$ to $-0.053$), not detectably different from its panel brake (difference $-0.002$, $-0.042$
to 0.037), and Llama-3.1 by $-0.050$ ($-0.067$ to $-0.031$), about a seventh of its panel brake (difference
$-0.301$, $-0.340$ to $-0.260$), consistent with a deliberation that weighs the third party's loss alongside
the rule.

*GPT-OSS-20B, its own sampled action (a different measure).* With its own reasoning at its default level,
we compare the action it samples after reasoning with its unreasoned forced choice, as a count of crossings
rather than a $\Delta_P$ \citep{openai2025gptoss}. Of the 44 scenarios whose two choices differ between the
rule and the departure, 42 move toward the rule (one-sided exact $p = 5.6 \times 10^{-11}$), the same share
as on the panel scenarios (74 of 76; two-sided Fisher $p = 0.62$). Its protective choice falls from 55\% of
scenarios unreasoned to 25\%, and a separate judgment trace sides with the reasoned action, not the
unreasoned one, in 25 of 31 decisive cases. On the ten spot-checked scenarios the author of this paper,
the only human reviewer, took the protective departure on nine.

*Scope.* The panel and this set differ in more than whether the rule protects (writers, families, length),
so the size comparisons above are between scenario sets, not a controlled contrast; a rule-protective twin
of each scenario in this set, read on Llama-3.1, is the cell that would separate them. The set is one structurally
homogeneous family, a transparent, auditable, personally costless departure from a written approval gate; a
second family (a hidden or costly departure, or a norm that is not a procedural gate) is the extension.
