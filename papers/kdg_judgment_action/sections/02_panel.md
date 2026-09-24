# 2. The panel {#panel}

## 2.1 Scenarios and frames

A scenario is a self-contained situation with a moral norm at stake, a situational pressure that
makes violating it locally advantageous, and two to four discrete options, at least one
norm-consistent and one norm-violating; most scenarios also carry a neutral option (hold, ask,
escalate). Five families vary the pressure type while holding the norm class distribution roughly
constant (\Cref{tab:families}); they draw on the pressure genres of the MACHIAVELLI benchmark
[@pan2023machiavelli], the agentic-misalignment scenarios [@lynch2025agentic], and
sycophancy-under-pushback evaluations [@sharma2023sycophancy], rewritten as short closed-menu
situations. Every scenario is written twice: an **agent frame** in the second person ("You are
Dana, the release coordinator...") for the action readout, and a **third-person frame** about the
same person for the judgment readout, with the same entities, facts, sentence order, and length
within 15% (tokenized with the model's own tokenizer); only pronouns and the role framing differ.
Option texts are plain descriptions of actions with no evaluative words, presented under neutral
letters whose order is a fresh seeded permutation on every rollout.

\begin{table}[tbp]
\centering
\caption{The five pressure families. F2 is an appendix family by pre-registration: its action is
closer to a second judgment than to doing something, and it never entered the gate. F5 is the
family a harm-keyed action channel would treat differently, and the harm-matched twins keep the
F5 contrast from being a harm-level contrast.}
\label{tab:families}
\small
\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.19\linewidth}>{\raggedright\arraybackslash}p{0.38\linewidth}>{\raggedright\arraybackslash}p{0.27\linewidth}@{}}
\toprule
id & family & pressure & action surface \\
\midrule
F1 & task completion & goal pursuit vs honesty (report a failure or fabricate success) & option choice \\
F2 & social cost & truth vs approval (user pushes back without new evidence) & multi-turn; final answer is the action \\
F3 & instrumental & rule vs shortcut (a prohibited but efficient route) & tool selection from a fixed menu \\
F4 & loyalty/fairness & in-group favor vs fair treatment & allocation choice \\
F5 & third-party harm & harm to a non-present party vs task success & option choice \\
\bottomrule
\end{tabular}
\end{table}

Each scenario also carries a **pressure-removed twin** (both frames with the incentive sentence
replaced by mundane filler, options unchanged) for the matched-null rung; a **harm-matched twin**
for F1 through F4 (same harm level, a different non-harm norm at stake), so that the harm-family
contrast is not a harm-level contrast; and, from the full panel on, **three paraphrases** of each
third-person frame for the reference-strictness ladder. Covariate tags record harm level (rated
by a non-generator model), norm class, register, the presence of a named user and of a tool
menu, and the generator.

## 2.2 Generation and labeling

Half of every family was written by Claude (Opus 5) and half by GPT (5.5), under one fixed system
prompt whose hash is stored with every scenario. A family-level result that differs by generator
is an anomaly by pre-registered rule, not a finding, and is reported per generator before it is
pooled. Each scenario's **external label** (which option an outside rater calls right, and a harm
level) was produced by the *other* provider, with the script refusing same-provider pairs; the
labels agree with the construction on 262 of 288 primaries, and every disagreement but three is a
preference for the neutral option. Drafts were validated against the construction rules and
regenerated with the violations fed back; safety refusals by a generator were regenerated
in-plan or, for four paraphrase requests, handed to the other generator with the exception
recorded on the scenario. Neither generator is the model under evaluation, so a generator's
normative priors enter the scenarios but not the readouts.

The panel of record has 248 primaries (40 per family plus 16 added to F1, F3, and F5 in the
second round), 152 harm twins, and 40 swapped F4 scenarios used only by one cell
(\Cref{structure}). Three GPT-written F4 twins whose option labels the cross-rater found
inverted are flagged and excluded from every analysis, leaving 397 scenarios in the analyses
below.

## 2.3 Readouts

\Cref{tab:readouts} lists every readout with the position it is read at and the statistic it
yields, so that each comparison in the paper can be checked for what it holds constant.

**Judgment** $J(s)$: the third-person frame, one shot ("Which option should Dana choose, and
why?"), parsed to an option; eight rollouts at temperature 0.7 plus one greedy. $J(s)$ is the
greedy option unless the sampled majority disagrees, in which case the scenario is flagged
unstable and excluded, counted. The model reasons in free text before its `Answer:` line, so the
judgment is a deliberated readout. **Action** $D(s)$: the agent frame under the chat template,
the model as the actor, "reply with only the letter"; 32 rollouts at temperature 0.7, reported
as the violating fraction and, for the binary gap, by majority. The decision position is the
first generated token, so the action is an immediate readout, and the full next-token
distribution there is saved for every rollout. **Raw frame**: the scenario, the option list, and
a fixed prefix ending before the option token, run on both the instruct and the base model with
no chat template; the readout is the normalized log-probability over the option letters across
eight option orders, with a mass floor (0.5 on the option letters) below which the model is not
engaging the option set.

**The gap.** $\mathrm{KDG}(s)=1$ if the majority action is norm-violating while the stable
judgment named a non-violating option; 0 if the majority action matches the judgment; undefined,
counted, otherwise. A neutral option counts as non-violating for the reference, by a pre-data
amendment: a model that judges "hold" right and acts violating has contradicted itself. The rate
is the mean over screened scenarios with a nested bootstrap over scenarios and rollouts. We keep
the program's internal name for the quantity, KDG, in the released code and data; in prose we
call it the judgment–action gap.

| readout | model | frame and instruction | position read | rollouts | statistic |
|---|---|---|---|---|---|
| judgment $J$ | instruct | third person, chat template, "which option should Dana choose, and why?" | token after the `Answer:` anchor, after the model's own reasoning | 1 greedy + 8 at $T$ 0.7, on four frames | option id; stability by sampled majority; option mass |
| action $D$ | instruct | agent frame, chat template, "reply with only the letter" | first generated token | 32 at $T$ 0.7, fresh option order each | violating fraction; majority; option mass |
| raw-frame $D$ and $J$ | base and instruct | scenario, options, fixed prefix ending before the option letter, no template | token after the prefix | 8 option orders, greedy | normalized option-letter mass; argmax |
| pressure-removed twin | as each cell above | incentive sentence replaced by filler, options unchanged | as each cell above | as each cell above | the matched null for that cell |
| known-gap control | instruct | agent frame with a system prompt ordering the violating action | first generated token | 32 | the positive band |

Table: Readouts. Every comparison in the paper is between cells that share a row of this table
except for the one quantity being varied. {#tab:readouts}
