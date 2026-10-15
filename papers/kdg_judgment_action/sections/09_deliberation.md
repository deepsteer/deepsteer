# 9. Moral deliberation before acting moves the action toward the model's judgment {#deliberation}

**In short.** Asking the model to think about what is at stake before it acts moves its choice back toward
its own judgment on both models that carry the gap, further than a same-length non-moral task does; on
OLMo-3, whose reasoning is truncated at the budget, naming the norm does about a third of it (0.18 to
0.64). Run on the same scenarios with the pressure removed, reasoning lowers the violating choice there
too. By the pre-registered rule this reduces the pressure-attributable part on Llama-3.1 and closes the
at-rest asymmetry on OLMo-3; a post-hoc ratio reading is consistent with one proportional cut on both
models (reasoning leaves 0.72 and 0.73 of the non-moral task's level on OLMo-3's pressured and
pressure-removed scenarios, 0.30 and 0.32 on Llama-3.1's) and does not separate a pressure-specific part
smaller than about a third (OLMo-3) or two-fifths (Llama-3.1) of the brake. Intervals in this section are
two-way bootstrap intervals over scenarios and rollout indices (\Cref{instruments}).

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
95\% two-way bootstrap CIs over scenarios and rollout indices, the estimator of record for sampled cells
(\Cref{instruments}); the scenario-only intervals printed in v1 are listed in \Cref{app:calibration}. The truncated-filler control gives the restatement the same cut-off,
forced form the reasoning has. The rows below the rule are the same arms on the
pressure-removed twins (amendment P1-A12) and the ratio reading (P1-A14, post-hoc). The two columns are not
a size comparison.}
\label{tab:deliberation}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.40\linewidth}>{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}p{0.27\linewidth}@{}}
\toprule
contrast & OLMo-3-7B-Instruct (truncated reasoning) & Llama-3.1-8B-Instruct (mostly completed) \\
\midrule
scenarios & 130 & 114 \\
reasoning finishes within 512 tokens & 8\% of rollouts & 90\% of rollouts \\
reasoning $-$ filler & $-$0.077 [$-$0.117, $-$0.037] & $-$0.350 [$-$0.395, $-$0.307] \\
reasoning $-$ truncated filler & $-$0.112 [$-$0.154, $-$0.070] & $-$0.320 [$-$0.367, $-$0.273] \\
truncated filler $-$ filler & +0.034 [0.009, 0.059] & $-$0.030 [$-$0.074, 0.016] \\
brief reasoning (64 tokens) $-$ filler & +0.024 [$-$0.020, 0.069] & $-$0.205 [$-$0.258, $-$0.154] \\
\midrule
twins: reasoning $-$ filler & $-$0.052 [$-$0.089, $-$0.017] & $-$0.208 [$-$0.251, $-$0.168] \\
primaries minus twins ($\Delta E$) & $-$0.025 [$-$0.065, 0.014] & $-$0.143 [$-$0.195, $-$0.090] \\
reasoning / filler, primaries; twins & 0.72; 0.73 & 0.30; 0.32 \\
log-ratio difference (primaries $-$ twins) & $-$0.011 [$-$0.231, 0.203] & $-$0.059 [$-$0.338, 0.236] \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{kdg_deliberation.pdf}
\caption{The dose arm on the two recipes that carry the gap: paired differences in the violating option's
mass at the forced answer, 95\% two-way bootstrap CIs over scenarios and rollout indices. Indigo circles are reasoning contrasts
(and, on OLMo-3, the norm-naming arm); gray squares are the truncation control. The panels use separate
scales because OLMo-3's reasoning is truncated (8\% finish within 512 tokens) and Llama-3.1's mostly
completes (90\%); their sizes are not compared.}
\label{fig:deliberation}
\end{figure}

**On OLMo-3, as truncated reasoning.** OLMo-3-Instruct's careful reasoning rarely finishes within the
512-token budget (8% of rollouts), so its arm reads a decision forced after reasoning that was cut off.
Even so, reasoning about the stakes lowers the violating mass relative to the filler ($-0.077$, $-0.117$
to $-0.037$). The reduction is not an artifact of the cut-off: truncating the filler the same way raises
its violating mass (+0.034), and the reasoning arm stays lower than the truncated filler by more
($-0.112$). A 64-token budget does not resolve.

**On Llama-3.1, mostly completed.** Meta's Llama-3.1-8B-Instruct finishes its reasoning within the budget
on 90% of rollouts, so its arm reads largely completed deliberation. Reasoning about the stakes lowers the
violating mass relative to the filler ($-0.350$, $-0.395$ to $-0.307$) and relative to its own truncated
filler ($-0.320$), and on this model even 64 tokens of reasoning lower it ($-0.205$). Truncating the filler
does not move it ($-0.030$, $-0.074$ to 0.016).

**How much of it is naming the norm.** A third arm on OLMo-3 gives the filler restatement a single
leading sentence that names the scenario's norm ("The norm at stake here is ..."), with no reasoning. It
lowers the violating mass by 0.025 (0.016 to 0.034), a share of 0.32 (0.18 to 0.64) of the reasoning
arm's effect. About a third of what reasoning does on OLMo-3 is done by naming the norm; the rest comes
with the reasoning. The pre-registered rule reads that share as mixed: its interval does not exclude 0.5.

**Letting the reasoning finish.** On sixteen OLMo-3 scenarios we also let the reasoning run to 2,048
tokens. The decision at 2,048 tokens agrees with the forced decision at 512 on twelve of the sixteen
scenarios; three of the four that change are third-party-harm scenarios, and they change in both
directions. That is a descriptive check at small $n$, and it is why the OLMo-3 result carries the label
"truncated reasoning" everywhere it appears.

**The same arms with the pressure removed.** We ran the reasoning and filler arms,
and the truncated filler, on the pressure-removed twins of the same scenarios, with the primaries' seeds and
option orders (amendment P1-A12, registered before the run). Reasoning lowers the violating mass on the
twins too (OLMo-3 $-0.052$, $-0.089$ to $-0.017$; Llama-3.1 $-0.208$, $-0.251$ to $-0.168$), and the
reasoning finishes within the budget as often as on the primaries (7% and 94% of rollouts). On the
registered probability scale the extra reduction under pressure, $\Delta E$, is $-0.143$ ($-0.195$ to
$-0.090$) on Llama-3.1, which the rule reads as reducing the pressure-attributable part, and $-0.025$
($-0.065$ to 0.014) on OLMo-3, which it reads as closing the at-rest asymmetry between judging and acting.
Against the truncated filler OLMo-3's $\Delta E$ is $-0.039$; it resolves under the scenario bootstrap but
not under the two-way bootstrap that accounts for the shared sampling streams ($-0.082$ to 0.004;
\Cref{instruments}), so OLMo-3 reads branch (b) against both references (corrected from v1). A reduction by a constant fraction produces this
pattern whenever the pressured baseline is higher, so after seeing the result we added a ratio reading
(P1-A14, post-hoc and labelled, pushed before its interval was computed). Reasoning leaves 0.72 of the
filler's violating mass on OLMo-3's primaries and 0.73 on its twins, and 0.30 and 0.32 on Llama-3.1's; the
difference of log ratios is $-0.011$ ($-0.231$ to 0.203) and $-0.059$ ($-0.338$ to 0.236), consistent with
one proportional cut on both models against both controls, and one common ratio predicts the
probability-scale difference on both
($-0.024$ against $-0.025$ observed on OLMo-3, $-0.136$ against $-0.143$ on Llama-3.1).

**What this says.** The action is not fixed at the moment the incentive is read. On both recipes that carry
it, reasoning about the stakes before acting moves the action back toward what the model judged right,
measured against a non-moral task of the same budget and against that task in the same truncated form,
about a third of it carried by naming the norm on OLMo-3. Reasoning brakes the violating action with or
without the incentive. Where following a written approval gate harms an absent third party
(\Cref{rule-harm}), deliberation still moves the action off the protective departure on all three models we
read there: onto the rule-following option on OLMo-3 and GPT-OSS-20B, and on Llama-3.1 onto the option that
contacts the approver, which keeps the rule without preventing the loss. On OLMo-3 and GPT-OSS-20B the effect
does not detectably depend on whether the rule is the protective choice; on Llama-3.1 the brake is about a
seventh of its panel size, a comparison between two scenario sets that differ in more than whether the rule
protects. By the pre-registered rule, the brake reduces the pressure-attributable part on
Llama-3.1 (branch (a), $-0.143$) and closes the at-rest asymmetry on OLMo-3 (branch (b)). The post-hoc
ratio reading (P1-A14) is consistent with one proportional cut on both models, under which the
pressure-attributable part shrinks in absolute terms because the lean it cuts is larger under pressure;
it does not separate a pressure-specific part smaller than about a third (OLMo-3) or two-fifths
(Llama-3.1) of the brake (2.8 standard errors on the log ratio: 0.31 and 0.41, from 0.21 and 0.30 under the
scenario bootstrap). Whether completed reasoning does more than
truncated reasoning on the same model and scenarios is not answered here; that comparison needs both
models on one scenario set with a budget long enough for both to finish.

## 9.1 When following the rule harms someone, the brake follows the rule {#rule-harm}

On the panel the norm-consistent option is almost always also the less harmful one, so a brake toward the
norm cannot be told apart from a brake toward avoiding harm. We built a separate set in which the two
conflict.

*The set.* Each scenario offers three closed-menu options: follow a written approval gate, which lets an
imminent loss fall on an absent third party (the norm-consistent option); depart from the gate,
transparently and at no gain to the agent, in the one way that prevents the loss (the violating option);
or log the request and contact the approver, which keeps the rule but does not prevent the loss in time
(the neutral option). The panel's generator prompt, schema and validators were used with one family brief;
Claude wrote 135 of the 270 scenarios and GPT the other 135. Two raters from different providers (one
Claude, one GPT) each compared the rule-following and the departing action, with labels hidden and the order
seeded: which is more likely to harm people other than the agent, and which follows the stated rule. A scenario was kept only
when both raters named the norm-consistent option on both questions: 264 of 270 (133 Claude-written, 131
GPT-written). The author's spot check of ten kept scenarios confirmed the construct on all ten. The set, its keep rule and each analysis
below were registered before the data they read, in amendments G-A24 to G-A29 (\Cref{app:prereg}); the
comparison of brake sizes between the panel and this set was requested by the author after the carrier
results and is descriptive. Data, code and the per-rollout arrays are listed in \Cref{app:repro}.

*The two carriers, forced readout.* Each carrier ran the panel's dose arm on the 264: the 512-token
reasoning arm against the filler of record, 16 rollouts per scenario, read at the forced answer
(\Cref{tab:deliberation}). Reasoning lowers the protective departure's mass on both, OLMo-3 by 0.076 (0.053
to 0.098) and Llama-3.1 by 0.050 (0.031 to 0.067). Where that mass goes differs (descriptive). On OLMo-3 it
goes to the rule-following option (+0.123, 0.101 to 0.145), and the neutral option also falls ($-0.047$,
$-0.068$ to $-0.027$). On Llama-3.1, which puts 85\% of its filler-arm mass on the departure, it goes to the
neutral option (+0.071, 0.057 to 0.086), and the rule-following option falls slightly ($-0.022$, $-0.036$ to
$-0.008$): Llama-3.1's deliberation moves it from protecting the third party to contacting the approver, not
to the act that causes the loss. Against its panel brake, OLMo-3's brake here is not detectably different
(panel minus this set $-0.002$, $-0.047$ to 0.045, with the panel arm under the two-way bootstrap), and
Llama-3.1's is about a seventh of its panel size ($-0.301$, $-0.349$ to $-0.254$). These are comparisons
between two scenario sets, not a controlled contrast (see *Scope*).

*GPT-OSS-20B, its own sampled action (a different measure).* Unreasoned, we read its forced letter choice
over eight option orders; with its own reasoning at its default level, we read the action it samples after
one reasoning trace, by token identity \citep{openai2025gptoss}. This is a count of transitions, not a
$\Delta_P$. \Cref{tab:h1-transitions} gives every transition on the 258 scenarios with an identifiable
reasoned action (six have none: three traces truncated, three without a letter). 124 keep their option, 44
cross directly between the rule and the departure, and 90 move to or from the neutral option. Of the 44
direct crossings, 42 move toward the rule (one-sided exact $p = 5.6 \times 10^{-11}$), the same share as on
the panel scenarios (74 of 76; two-sided Fisher $p = 0.62$). Of the 90 lateral moves, 44 leave the
departure for the neutral option against 9 the other way. Its protective departure falls from 140 of 258
scenarios unreasoned (54\%) to 65 (25\%). Counting an explicit deferral to the approver as protective too, it
falls from 176 (68\%) to 110 (43\%), and the departure rate after reasoning is the same whether or not the
scenario offers a deferral (27\% and 24\%), so the fall is not a substitution into deferral. A separate
judgment trace at the same reasoning level sides with the reasoned action, not the unreasoned one, in 25 of
31 decisive cases. On the ten spot-checked scenarios the author of this paper, the only human reviewer, took
the protective departure on nine.

\begin{table}[tbp]
\centering
\caption{GPT-OSS-20B on the 264 rule-harms scenarios: the unreasoned forced choice (rows) against the action
sampled after reasoning at its default level (columns), on the 258 scenarios with an identifiable reasoned
action. Bold cells are the direct crossings between the rule and the departure; the rest off the diagonal
are lateral moves through the neutral option.}
\label{tab:h1-transitions}
\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
unreasoned $\backslash$ reasoned & follows the rule & neutral & departs (protects) & total \\
\midrule
follows the rule & 15 & 10 & \textbf{2} & 27 \\
neutral & 27 & 55 & 9 & 91 \\
departs (protects) & \textbf{42} & 44 & 54 & 140 \\
\midrule
total & 84 & 109 & 65 & 258 \\
\bottomrule
\end{tabular}
\end{table}

*Scope.* The panel and this set differ in more than whether the rule protects (writers, families, length),
so the size comparisons above are between scenario sets, not a controlled contrast; a rule-protective twin
of each scenario in this set, read on Llama-3.1, is the cell that would separate them. The set is one
structurally homogeneous family, a transparent, auditable, personally costless departure from a written
approval gate; a second family (a hidden or costly departure, or a norm that is not a procedural gate) is
the extension.
