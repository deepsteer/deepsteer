# Direction 3 (C1) — Literature pass (zero-GPU; gates the novelty framing)

**Date:** 2026-07-02. Per `compute-ordering` (novelty sentences are blocked until this lands) and
the program's citation-verification rule (assume hallucination; verify every ref against the primary
source). Each entry marks how it was verified.

## Verified references (fetched from the primary source this session)

| ref | id | what it establishes | relevance to C1 |
|---|---|---|---|
| Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (2024) — *Refusal in LMs Is Mediated by a Single Direction* | `arXiv:2406.11717` (NeurIPS 2024) | refusal = a **one-dimensional** subspace across 13 chat models; erase → no refusal, add → over-refuse | the WRITE side: refusal is low-rank. C1 asks where that write lands (the channel) and what it reads |
| Zhou, Yu, Zhang, Xu, Huang, Wang, Liu, Fang, Li (2024) — *On the Role of Attention Heads in LLM Safety* | `arXiv:2410.13708` | **Ships** (safety-head importance) + **Sahara** (dataset-level OV-circuit head attribution); ablating ONE safety head → **16× more harmful** at **0.006%** of params; "heads are feature extractors for safety" | **closest prior art.** Sahara already does per-head OV attribution of safety heads → **C1's Stage 1 method is NOT novel**; C1 must cite this and differentiate on Stage 2 |
| Zhao, Huang, Wu, Bau, Shi (2025) — *LLMs Encode Harmfulness and Refusal Separately* | `arXiv:2507.11878` | a **harmfulness** direction distinct from the **refusal** direction; harmfulness encoding more robust; steering them gives different behaviors | grounds C1's **harm-cue vs moral-content** split (Stage 2 value-side); the `t_inst` harm direction is this paper's harmfulness object (the program already uses its localization in Paper 7) |
| Sun, Chen, et al. (2024) — *Massive Activations in LLMs* | `arXiv:2402.17762` | a few activations ~10^5× larger, **input-constant**, act as implicit bias terms, concentrate attention | grounds `ANOMALIES.md` A1 (Qwen/Llama outlier dims) AND the decision-site low-rank substrate |
| Xiao, Tian, et al. (2023) — *Efficient Streaming LMs with Attention Sinks* | `arXiv:2309.17453` (ICLR 2024) | **attention-sink** tokens absorb attention mass | grounds the decision-site / template-token as a sink-like low-rank position (the bottleneck) |
| Olsson, Elhage, et al. (2022) — *In-context Learning and Induction Heads* | `arXiv:2209.11895` | **induction/copy heads** `[A][B]…[A]→[B]`; causal in small models | grounds C1's **copy-head hypothesis** (a head copying `t_inst` harmfulness into the decision token) |
| Qi, Panda, Lyu, Ma, Roy, Beirami, Mittal, Henderson (2024) — *Safety Alignment Should Be Made More Than Just a Few Tokens Deep* | `arXiv:2406.05946` | safety alignment is **shallow** — adapts the generative distribution "primarily over only its very first few output tokens"; deepening improves robustness | **PROMOTE to framing:** the decision-site control-token bottleneck is the **geometric mechanism** of Qi's behavioral shallow-alignment. Position C1 as "the substrate of shallow alignment, measured" |
| Wollschläger, Elstner, Geisler, Cohen-Addad, Günnemann, Gasteiger (2025) — *The Geometry of Refusal in LLMs: Concept Cones and Representational Independence* | `arXiv:2502.17420` | refusal = **multiple independent directions + multi-dimensional concept cones**, contra a single direction | **must engage:** D2/C1 place refusal in a low-rank *channel*; a ~15-dim channel comfortably **hosts** a cone (compatible), but the claim must be stated as channel-not-direction and reconciled explicitly |

## Found in search, NOT yet primary-verified (cite only after verifying)

- *There Is More to Refusal in LLMs than a Single Direction* (`arXiv:2602.02132`, 2026) — argues refusal
  exceeds a single direction (converges with the Wollschläger cone + the ~10–15-dim channel). Very
  recent; verify before leaning on it. (Wollschläger 2502.17420 is now primary-verified, table above.)
- *Understanding Refusal in LMs with Sparse Autoencoders* (`arXiv:2505.23556`) — refusal SAE features.
- 2026 head-specialization / jailbreak work: `2606.28153` (attention-head specialization under
  jailbreak), `2603.27518` (over-refusal + representation subspaces), `2601.15801` (safety vectors).
- **Patching methodology precedents** (cite by concept; verify ids before final): path patching /
  IOI circuit (Wang et al., "Interpretability in the Wild", GPT-2-small), causal mediation analysis
  (Vig et al.), interchange interventions / causal abstraction (Geiger et al.). C1's causal cells use
  these standard tools; ids to be pinned at write-up.

## The landscape, and where C1 sits

Three lines of verified prior work converge on the pieces C1 assembles:
1. **Refusal is low-rank** (Arditi: 1-D; 2026 work: a bit more) — the *write* direction.
2. **Safety localizes to sparse attention heads**, attributable via OV circuits (Zhou/Sahara) — the
   *writers*, and even a causal handle (one head → 16× harmful).
3. **Harmfulness ≠ refusal** as directions (Zhao) — so "what a writer reads" has at least two typed
   candidates (harm cue vs the decision), before moral content is even considered.
4. **The substrate is low-dimensional at special tokens** (massive activations, attention sinks).

**What is NOT in the literature — C1's contribution (novelty sentence for `PAPER_PLAN` §1):**

> Prior work localizes refusal to a low-rank direction (Arditi et al. 2024) and to sparse safety
> heads attributable through OV circuits (Zhou et al. 2024), and separates harmfulness from refusal
> as representations (Zhao et al. 2025); the low dimensionality of special-token activations is the
> massive-activation / attention-sink phenomenon (Sun et al. 2024; Xiao et al. 2023). **We use the
> established OV-attribution method (Sahara-style) not as the contribution but as a first step**, and
> add two things that are absent from this literature: (1) we show the refusal decision is written
> into a **calibrated, measured ~10–15-dimensional control channel** — a low-rank decision bottleneck
> established by a positive-control band and a position-validity rule, not assumed; and (2) we test
> whether the writing heads **transport moral content** — a calibrated `V_moral` subspace — into that
> channel, versus a harm cue (Zhao's harmfulness) versus surface alarm, with a `V_moral`-restricted-
> vs-full interchange patch as the decisive test. The contribution is **moral-content transport into
> a measured low-rank decision channel**, and a causal cell (`V_moral`-restricted patch) that, in one
> outcome, would explain the program's geometric nulls (orthogonality with causal coupling through
> non-subspace features). It is not "attention heads matter for safety," which is established.

**Framing promotion (Qi et al. 2024).** Qi et al. show *behaviorally* that safety alignment is
shallow — concentrated in the first few output tokens. The decision-site control-token bottleneck
(PR ~10–15 at the assistant-header token, cross-model) is a candidate **geometric mechanism** for
that shallowness: alignment writes into a narrow bus at exactly the position Qi identify. C1 is
positioned as *"the measured substrate of shallow alignment,"* a stronger frame than head-attribution
alone (correlational until a causal cell ties the channel to the depth effect; stated as such).

**Cone reconciliation (Wollschläger et al. 2025).** Refusal is multi-dimensional (a concept cone),
contra Arditi's single direction. **Compatible** with D2/C1: we claim refusal is written into a
low-rank *channel* (~15 dims), not that refusal is one direction — a 15-dim channel comfortably hosts
a cone. The write-up states "channel, not direction" and reconciles explicitly.

**Honesty riders folded into the plan:**
- **Stage 1 is a known method** (Zhou/Sahara). The `PAPER_PLAN` and `PREREGISTRATION` must credit it
  and not frame per-head OV attribution as novel; novelty is Stage 2 + the causal content-transport
  cells + the calibrated-channel instrument.
- The harm direction C1 reads value-side is **Zhao's harmfulness direction**; cite accordingly.
- The bottleneck substrate connects to massive activations / attention sinks; the novel piece is
  **calibrating** it (band-below-null ⇒ position-invalid) and reading it as a *decision* channel.
