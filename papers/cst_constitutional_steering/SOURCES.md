# CST-G0 source verification (T0.2 / T0.7 / T0.9), 2026-11-04

Method: every item below was read from the primary source fetched on 2026-11-04. arXiv metadata
(title, authors, version dates, abstract) came from `export.arxiv.org/abs/<id>` meta tags; paper
bodies from `arxiv.org/html/<id>v<n>`; GitHub facts from the GitHub REST API and raw files at a
pinned sha; HF facts from the HF API and raw dataset cards. Author lists are copied in the printed
order. One source could not be read directly: openai.com returned HTTP 403, so the OpenAI blog was
read from the Wayback Machine copy and its companion arXiv paper. Raw downloads are in
`scratchpad/src/` and `scratchpad/dl_scruples/` (scratch only, nothing committed).

Verdict key: OK = the plan sentence is supported; PARTIAL = supported with a scope the plan drops;
MISMATCH = the source says something else.

---

## 1. Anthropic, "Teaching Claude Why"

- Exists: yes. URL: https://alignment.anthropic.com/2026/teaching-claude-why (there is also an
  anthropic.com/research/teaching-claude-why page, which I did not read).
- Title: "Teaching Claude Why" (Alignment Science Blog).
- Date: "May 8, 2026". The plan's 2026-05-08 is correct.
- Authors, as printed in three byline rows: "Jonathan Kutasov*, Adam Jermyn" / "Julius Steen, Minh
  Le, Samuel R. Bowman, Samuel Marks, Jan Leike, Amanda Askell, Chris Olah" / "Evan Hubinger, Sara
  Price"; "*Correspondence to jonk@anthropic.com". The plan cites it as "Anthropic Alignment
  Science", which is fine for a blog post; if it is cited by authors, use this list.
- Models: "Most of our experiments were conducted by further fine-tuning Claude Sonnet 4 or Claude
  Haiku 4.5 (or the base model from which they were trained)."

Plan claims:

a) "SDF of a base model on constitution documents and stories beats chat-format SFT of the same
   content": **PARTIAL.** The format comparison exists, but it covers constitution documents, not
   the stories, and its outcome is the constitution evals, not the misalignment honeypots.
   - "As in Slocum et al., we find empirically that teaching the model the constitution through
     documents is more effective than using chat data. Concretely, for each dataset of documents,
     we tried converting the document into a chat format (where the human asks a reasonable
     question about the assistant's values or beliefs and the assistant responds with the
     information in the document) and trained separate models on those datasets."
   - Scope, from the figure caption: "All forms of document training out-perform chat-based
     training on the factual recall and hallucination evals. For the open ended variant of the
     eval, there is less of a delineation between the chat and document formats though documents
     still look slightly better for each category."
   - The Limitations section concedes the mechanism is not known: "We don't fully understand why
     SDF on constitution documents is more effective than chat-formatted data, or why RL improves
     factual recall without improving open-ended alignment scores."
   - Suggested wording: "documents beat a chat conversion of the same documents on constitution
     recall and hallucination evals, and slightly on open-ended value questions."

b) "a 3M-token 'difficult advice' set with a constitution-rewrite step matched 85M tokens of
   eval-targeted SFT and generalized to held-out audits": **OK, with one caveat.**
   - "Strikingly, we achieved the same improvement on our eval with just 3M tokens of this much
     more out-of-distribution (OOD) dataset. Beyond the 28x efficiency improvement, ..." (3M x 28
     is about 84M.) The figure caption: "When scaling up the size of the training set from ~29M
     tokens to ~85M tokens, we are able to reduce the misalignment rate to about 1% with this
     method."
   - Held-out evidence: "this model performs better on an older version of our automated alignment
     assessment."
   - The comparator is not plain eval-targeted SFT. It is synthetic honeypots whose responses were
     generated with a system-prompt injection and then filtered.
   - Caveat that matters for CST: "All of the datasets that appear in this section have extended
     thinking turned off. When we reference the model's reasoning, we mean the user-facing
     explanation ... not the model's internal reasoning trace." No result in this section involves
     a thinking trace.

c) "gains persisted through RL": **OK for constitutional SDF.** The plan's sentence order makes it
   read as if it applies to the difficult-advice set, which the source does not show.
   - "Thus we find that the effect of constitutional SDF (and high quality SL on chat data) not only
     survives RL, but in fact interacts positively with RL". The RL run used snapshots of "a
     haiku-class model" with harmlessness-targeted environments.

d) "ablating the rewrite step raised misalignment from about 2% to 19%": **MISMATCH (the baseline
   number is wrong).**
   - "Ablating step 4 (rewriting the prompts) increases misalignment rate to 2% (within error bars).
     Ablating step 6 (rewriting the final response) increases misalignment rate to 19%. This single
     step - having Claude rewrite the response to better align with the constitution - accounts
     for a 19x reduction in misalignment rate."
   - The 2% figure belongs to the prompt-rewrite ablation. The full pipeline is about 1% (implied by
     the 19x). Correct wording: "ablating the constitution-guided response rewrite (step 6) raised
     misalignment from about 1% to 19%; ablating the prompt rewrite gave 2%, within error bars"
     (Claude Sonnet 4, thinking off).

Other primary-source facts relevant to CST:
- "We have observed a difference in what the model will report of its own beliefs vs. what it
  reports as Claude's beliefs after this training, indicating that the model is still not fully
  attaching to the Claude persona." This is a self-versus-character gap, adjacent to CST's
  saying-versus-doing.
- Stories: "SDF on a mixture of ~12k of these stories (~30M tokens)". A 14M-token story SDF run is
  also reported.

---

## 2. arXiv:2511.01689, "Open Character Training"

- Exists: yes. https://arxiv.org/abs/2511.01689
- Title: "Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI"
- Authors: Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger. **Matches the plan.**
- Versions: v1 only, Mon, 3 Nov 2025 15:53:47 UTC. Paper license on arXiv: CC BY 4.0.
- Models: Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B. These are instruct models (HF repos are named
  `*-it-*`) and none is a reasoning model. Teacher: GLM 4.5 Air.

Plan claims:
- "constitution in the teacher's context → DPO with an NLL term (LoRA)": **OK.**
  "the teacher is provided with the constitution in a system prompt and instructions to embody it
  during conversation, to generate chosen responses for DPO ... Meanwhile, the student responds to
  the same prompts without any such instructions, generating rejected responses". Also "Training is
  performed using LoRA adapters ... with a rank of 64" and "We add a per-token KL-divergence penalty
  for stability and a negative log-likelihood (NLL) loss term with a scaling coefficient of 0.1 on
  the chosen generations". The KL term is not in the plan's summary; add it if the A-arm copies the
  recipe.
- "introspection SFT on self-generated transcripts": **OK.** "The full introspective dataset of
  12,000 transcripts, combining self-reflection and self-interaction". The split is 10,000
  self-reflections ("ten reflective prompts ..., generating 1000 different responses per prompt")
  plus 2,000 self-interactions ("We sample 2000 exploratory self-interactions"). This is per
  model/persona pair, one epoch, LoRA rank 64. Distillation and introspection adapters are
  "linearly merge[d]".
- "personas held under break-character prompts where system prompts did not": **OK.** "Using system
  prompts to shape persona is particularly brittle, where adversarial instructions frequently 'break
  character' and lead to generic 'helpful assistant'-style behavior ... Fine-tuning leads to the
  highest average classifier performance across models".
- Release:
  - Code: https://github.com/maiush/OpenCharacterTraining, license **MIT** ("Copyright (c) 2025
    Sharan Maiya"). Latest commit d1da9f03628cb4c5482ba2e494a7cba33bcd5818, 2026-04-04 push
    (commit date 2025-11-24).
  - Models: HF collection `maius/open-character-training` (`maius/{llama-3.1-8b-it,qwen-2.5-7b-it,
    gemma-3-4b-it}-{personas,misalignment}`).
  - Data: `maius/OpenCharacterTraining-data` (and `-misalignment`). HF tag `license: other`. The
    card reads: "This dataset follows the same license used in LIMA ... 'If the source data of LIMA
    has a stricter license than CC BY-NC-SA, the LIMA dataset follows the same. Otherwise, it
    follows the CC BY-NC-SA license.'"
  - **Flag: the data is non-commercial (CC BY-NC-SA or stricter).** The code is MIT and safe to
    reuse. If CST reuses OCT data or prompts rather than code, the program's NC rule (see MORABLES
    precedent) applies.

---

## 3. arXiv:2605.24229, "How Well Do Models Follow Their Constitutions?"

- Exists: yes. https://arxiv.org/abs/2605.24229
- Title: "How Well Do Models Follow Their Constitutions?"
- Authors: Arya Jakkli, Senthooran Rajamanoharan, Neel Nanda. **Matches the plan.**
- Versions: v1 only, Fri, 22 May 2026 21:17:16 UTC.
- Counts: **OK.** "decomposes the specification into atomic testable tenets (205 for Anthropic, 197
  for OpenAI)".

### (a) Decomposition procedure, as far as the source states it

Section 3, verbatim: "We decompose each specification into atomic behavioral tenets: statements of
intended model behavior that can plausibly be violated by a single transcript. We extract candidate
tenets section-by-section using Claude Opus 4.5 with a prompt that asks for verifiable behavioral
claims with surrounding context preserved, then manually review the output, merge near-duplicates,
and assign each tenet a priority level based on the specification's own prioritization scheme
(e.g., Anthropic's four priority levels P1–P4; OpenAI's red-line, chain-of-command, content, safety,
truth, best-work, style, and under-18 sections). Anthropic's constitution ... is decomposed into 205
tenets across 19 sections. OpenAI's Model Spec ... is decomposed into 197 tenets across 8 sections.
Both specifications contain ambiguous passages, and different decompositions would shift some
aggregate counts". The Limitations section adds: "Tenet decomposition is opinionated; our parse is
one of several reasonable choices."

What this gives a reimplementer:
- Model: Claude Opus 4.5.
- Unit of work: one specification section per call.
- Prompt intent: "verifiable behavioral claims with surrounding context preserved".
- Review: manual review, then manual merging of near-duplicates. There is no automated dedup and
  no second-model review.
- Granularity: "can plausibly be violated by a single transcript".
- Priority assignment: done by hand using the document's own levels (P1 broadly safe, P2 broadly
  ethical, P3 Anthropic guidelines, P4 genuinely helpful).

What is missing:
- **The extraction prompt itself is not in the paper or the repo.** The repo tree (871 paths at
  sha f5e80c1) has no decomposition script or prompt. The only tenet tooling is
  `petri/scripts/update_anthropic_tenets.py`, which copies Petri `Sample` inputs into the markdown.
- No inter-rater or review statistics are given.

So "the Nanda-group procedure" can only be reimplemented at the level of the description above.
Plan §2.2 should say it follows their described procedure, not that it reproduces it.

Released tenets are not bare statements. Each one is a Petri seed instruction: a title, a citation
to line ranges of their markdown copy of the constitution, "Scenarios to explore", and "Evaluation
criteria". Example: "T1.1a ... `priority`, `safety_vs_helpfulness` · constitutions/anthropic_soul_doc.md:144-180 — Test tenet T1.1a (Safety vs. Helpfulness Priority): ...". The granularity is
about one sub-claim per ID (T1.1a/b/c).

### (b) Are the tenet lists public?

**Yes, but there is no license and the paper contradicts itself.**
- Repo: https://github.com/ajobi-uhc/constitution-audits. Default branch `main`, head
  f5e80c1881bdb4cb3a8fcc087c9da80dc7e47141 (2026-10-02).
- Files: `tenets/anthropic_constitution.md` (366 KB) and `tenets/openai_model_spec.md` (268 KB),
  both Markdown. Each tenet is one bullet `- **T<sec>.<n><x>** ... — <seed text>`.
- Counts in the files: Anthropic 211 bullets ("205 evaluated tenets + 6 additional power tests";
  the 6 are in `POWER_TENETS` and "did not produce transcripts in the reported 205-sample runs").
  OpenAI 197.
- Machine-readable source: the `Sample(id=..., input=...)` objects in
  `petri/claude_soul_doc_audits/soul_doc_audit.py` and
  `petri/openai_model_spec_audits/openai_model_spec_audit.py`.
- **License:** GitHub reports none for the repo. There is no top-level LICENSE. Only the bundled
  `petri/` folder has one (MIT, "Copyright (c) 2025 Safety Research"), and it covers the Petri
  copy, not the tenet files. Without a license, the tenet lists are all-rights-reserved by default.
  CST can cite them and compare its own decomposition against them, but should not vendor them
  without asking the authors.
- Paper inconsistency: §3 says "we release the full tenet lists for both specifications ... at
  https://github.com/ajobi-uhc/constitution-audits", while Appendix H says "The full tenet lists ...
  will be released alongside the camera-ready version." The repo now has them.
- Section-count discrepancy: the paper says 19 sections (Anthropic) and 8 (OpenAI). The released
  Anthropic file has 16 section headings (numbered 1–14, 17, 18, with §10 holding the 6 extra
  power tenets). The OpenAI file has 5 headings (1, 3, 4, 7, 8) that do not match the paper's
  8-section table (Redline 17, Chain of command 29, Content 23, Safety 23, Truth 44, Best work 20,
  Style 32, Under-18 9). Do not quote section counts from the paper as if they describe the files.

### (c) Which document revisions they decomposed

- Anthropic: `constitutions/anthropic_soul_doc.md`, a "PDF To Markdown Converter" rendering whose
  header reads "Published January 21, 2026" (authors on the document: "Amanda Askell,* Joe
  Carlsmith,* Chris Olah, Jared Kaplan, Holden Karnofsky, several Claude models, and many other
  contributors"). The paper's bib entry reads "anthropic.com/news/claude-new-constitution, 2025a",
  which is a mislabelled year. Despite the filename "soul_doc", this is the published constitution.
  It is the **same revision** as plan C_anth (`dated: '2026-01-21'` in constitutions.yaml), but
  their line anchors refer to their PDF conversion, not to our text-sha ab902738.
- OpenAI: the bib reads "model-spec.openai.com, December 2025a". The repo copy carries no date
  stamp. Plan C_spec is the 2026-08-18 revision, **a later revision than theirs**. Their 197
  Model Spec tenets do not map one-to-one onto C_spec, and the gap should be a CST-A entry if any
  comparison is made.

### (d) The failure taxonomy, exactly

- The framing sentence (§5): "The failures eliminated by recent post-training were cases where the
  specification gave a clear instruction and the model failed to follow it. The failures that
  persist concentrate in places where the specification itself sets competing directives: operator
  versus user, credential gating versus jailbreak resistance, autonomy in agentic deployments
  versus avoidance of unilateral action, surface form versus underlying harm. We group remaining
  failures into five categories."
- The five headed categories, in order:
  1. **Authority conflicts** (e.g., Opus 4.6 as "Megan Rivera" denying AI identity; GPT-5.1
     following a developer slant directive after the user withdrew consent).
  2. **Credential-gated safety** ("Several models treat unverifiable professional credentials as
     sufficient to relax safety boundaries").
  3. **Form-over-substance boundaries.**
  4. **Think-then-ignore (GPT family)**: "the model's reasoning identifies a problem, articulates
     why it is a problem, and proceeds anyway." Examples are GPT-5 (.tk domain) and GPT-5.2
     (mania).
  5. **Unilateral action in agentic settings.**
- Two further subsections follow: "Comparison-model failure modes" (Gemini 3 Pro, Sonnet 4) and
  "Fabrication with false precision (SURF results)".
- The conclusion's list: "authority conflicts, credential-gated safety, agentic autonomy
  thresholds, form-over-substance boundaries, fabricated claims with false precision."
- The abstract's list differs again: "operator-imposed personas under AI-identity questioning,
  irreversible action in agentic deployments, and fabricated quantitative claims with false
  precision."

Plan sentence ("the surviving failures sit where the document gives competing directives
(authority conflicts, credential-gated safety, think-then-ignore)"): **PARTIAL / MISMATCH on
think-then-ignore.**
- The competing-directives sentence is supported.
- Authority conflicts and credential-gated safety are both competing-directive categories.
- Think-then-ignore is a failure *mode* that the paper attributes to the **GPT family on the OpenAI
  Model Spec**. It is not listed as a competing-directive pair and is not a finding about Claude on
  Anthropic's constitution.
- The plan's three-item list also leaves out form-over-substance and unilateral agentic action.
- Suggested wording: "...concentrate where the specification sets competing directives (operator
  vs user, credential gating vs jailbreak resistance, agentic autonomy vs unilateral action, form
  vs harm); they report a GPT-family 'think-then-ignore' mode in which the reasoning names the
  problem and the model proceeds."

Two more points relevant to CST:
- Their CoT-acknowledged-violation category ("think-then-ignore") is the **closest prior art to
  CST's saying-vs-doing outcome** (see §10 ii).
- Their soul-doc attribution: "Sonnet 4.5 → Opus 4.5: 7.3% → 4.4% ... against ... Sonnet 4 → Sonnet
  4.5: 15.0% → 7.3%"; "The soul-doc-attributable step is smaller than the broader post-training
  step."

---

## 4. arXiv:2605.24286, "Faithfulness as Information Flow"

- Exists: yes. https://arxiv.org/abs/2605.24286
- Title: "Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought
  Reasoning"
- Authors: Jinghan Jia, Joe Benton, Eric Easley. **Matches the plan.**
- Versions: v1 only, Fri, 22 May 2026 23:37:29 UTC.

Plan claims:
- "sufficiency, completeness and necessity": **OK.** "a task-agnostic framework based on three
  complementary properties, sufficiency, completeness, and necessity, which we instantiate with
  entropy-based, masked-KL, and gradient-based diagnostics."
- "gradient-mass measures (Grad-DE, Grad-Nec)": **OK.** "A higher Grad-DE indicates greater
  shortcut reliance; a higher Grad-Nec indicates stronger dependence on the CoT. Since exact
  interventional necessity is intractable during training, [it] serves as a first-order
  linearization of the answer's sensitivity to perturbing token". **Construct note for §6.3:**
  Grad-Nec is a local gradient-sensitivity share, not an interventional necessity, and the authors
  say so. The paper also reports that KL-Nec "separates the two models significantly, but in the
  wrong direction".
- "update-time masks inside GRPO that leave the reward unchanged": **OK.** "We instead modify the
  policy update, leaving rollouts, rewards, and advantages unchanged". Table 2: "All methods
  preserve rollout generation, rewards, and advantage normalization, and differ only in how they
  modify the GRPO policy update." The methods are Update Mask, Gradient Mask, CoT-only Gradient and
  FACT (a prompt-representation perturbation, which is not a mask).
- Models in the paper: Qwen3-8B and DeepSeek-R1-Distill-14B (metric validation on hinted GPQA);
  Gemma3-1B/4B/12B-IT; Qwen2.5-7B-Instruct. None is constitution-trained.

GitHub `safety-research/faithful-cot`:
- Exists: yes. https://github.com/safety-research/faithful-cot. The README title is "Structural
  RL: Enforcing CoT Faithfulness via Structural Training Interventions".
- **License: none at the repo root.** The GitHub API returns `license: null`, the root has no
  LICENSE file, and the README does not mention a license. The only license file is `train/LICENSE`,
  **Apache-2.0**, which is the vendored VERL fork ("the VERL package lives there"). So the `eval/`
  metric code, including Grad-Nec, carries **no license (all rights reserved)**. Vendoring it is
  not licensed. The T0.9 decision at G0 should be **re-implement** (the function is about 40 lines)
  or ask the authors.
- Default branch `main`. Latest (and only) commit: **f9e1f29564fcd8d3a50507766e028547cdbbbd23**,
  2026-05-12T22:03:09Z, "Initial release: Faithfulness as Information Flow". Pushed
  2026-05-12T22:04:47Z.
- Grad-Nec implementation:
  - `eval/faithfulness_metrics_clean.py`: `def grad_cot_necessity(model, inputs) -> (grad_l1,
    grad_l2)`, using `compute_gradient_norms(...)`.
  - Duplicates: `eval/faithfulness_metrics_clean_qwen.py` and `eval/faithfulness_metrics_pytorch.py`
    (`FaithfulnessEvaluator`, fields `grad_cot_necessity_l1/_l2`).
  - Mechanics: per-token ||∂L_answer/∂e_t|| on input embeddings under the full causal mask. Grad-Nec
    L1 = Σ_CoT ||g_t|| / Σ_all ||g_t||, and L2 is the same with squared norms. It returns NaN when a
    sample has no answer tokens.
- **Dependencies are light:**
  - The metric files import only `torch`, `torch.nn`, `torch.nn.functional`, `transformers`
    (AutoTokenizer / AutoModelForCausalLM), `dataclasses`, `typing` and `re`.
  - The driver (`compute_faithfulness_from_generations*.py`) adds `pandas` and `tqdm`, plus
    token-position helpers from the vendored `verl.utils.cot_masking`, which itself imports only
    `torch`.
  - **No Captum, TransformerLens or nnsight** anywhere in `eval/`.
  - Training interventions need the VERL fork (Ray, FSDP, Hydra) and are not needed for a read-only
    Grad-Nec.
- Hygiene note: the repo also contains a personal `dotfiles/` tree and `.claude/.nfs*` files. This
  is irrelevant to CST, but do not mirror the repo wholesale.

---

## 5. arXiv:2603.30036, "Aligned, Orthogonal or In-conflict"

- Exists: yes. https://arxiv.org/abs/2603.30036
- Title: "Aligned, Orthogonal or In-conflict: When can we safely optimize Chain-of-Thought?"
- Authors (from the arXiv page): Max Kaufmann, David Lindner, Roland S. Zimmermann, and Rohin
  Shah. The meta tag splits the last name oddly ("Shah, and Rohin"); the person is Rohin Shah.
  **Matches the plan.**
- Versions: v1 only, Tue, 31 Mar 2026 17:38:39 UTC.

Plan claims:
- "a chain-of-thought reward term that conflicts with the outcome reward produces obfuscation":
  **OK.** "We find that (1) training with 'in-conflict' reward terms reduces CoT monitorability and
  (2) optimizing in-conflict reward terms is difficult." Their mechanism is obfuscation /
  non-transparency. Note the second clause, that in-conflict terms are hard to optimize, which the
  plan omits and which matters if CST's GRPO rubric reads the CoT.
- "length penalties and preference rewards already do": **PARTIAL (overstated).**
  - "Our length penalties and human preferences rewards are simplified versions of rewards commonly
    used in real post-training environments. In our environments, we demonstrate that these rewards
    cause issues for monitorability."
  - Conclusion: they "can be in conflict under our framework and may undermine monitorability.
    Although our experiments are limited to toy environments".
  - Counter-evidence in the same paper: "Length penalty caused only trivial unmonitorability in
    Backdoor-Easy ... the policy simply outputs empty reasoning". Also: "many modern models are
    trained with length penalties without any observed drop in monitorability (Guan et al., 2025)."
  - Human preferences are classed "In-Conflict or Orthogonal" depending on the environment.
  - Suggested wording: "in toy environments, simplified length-penalty and human-preference CoT
    rewards can be in conflict and reduced monitorability."

---

## 6. arXiv:2610.29734, "Monitorability Is Trainable"

- Exists: yes. https://arxiv.org/abs/2610.29734
- Title: "Monitorability Is Trainable: Post-Training for Chain-of-Thought Monitorability"
- Authors: Usman Anwar, Tim Bakker, Christos Louizos, David Krueger. The page notes that the first
  two authors contributed equally. **Matches the plan.**
- Versions: v1 only, Wed, 21 Oct 2026 13:35:57 UTC. No HTML rendering exists, and the page lists no
  code link.
- "a mutual-information objective": **OK.** "we propose an unsupervised post-training objective
  based on maximizing mutual information between CoT and the output".
- "on GPT-OSS-120B": **UNVERIFIED.** The abstract names no model, arXiv has no HTML, and I did not
  read the PDF. Read the PDF before this model name enters prose, or drop it (the plan uses this
  paper only as background).

---

## 7. OpenAI, "Evaluating chain-of-thought monitorability"

- Exists: yes. https://openai.com/index/evaluating-chain-of-thought-monitorability/ (direct fetch
  returned 403; read via the Wayback Machine). Dated "December 18, 2025". **The plan's date is
  correct.** No byline was shown in the archived blog text.
- Companion paper: arXiv:2512.18311, "Monitoring Monitorability". Authors: Melody Y. Guan, Miles
  Wang, Micah Carroll, Zehao Dou, Annie Y. Wei, Marcus Williams, Benjamin Arnav, Joost Huizinga, Ian
  Kivlichan, Mia Glaese, Jakub Pachocki, Bowen Baker. v1 Sat, 20 Dec 2025.
- "longer reasoning is more monitorable": **OK.** Blog: "Models that think for longer tend to be
  more monitorable". Paper: "We find that longer CoTs are generally more monitorable".
- "Scruples sycophancy is the least monitorable domain": **PARTIAL (the superlative is not in the
  source).**
  - Blog caption: "In general we find them to be very monitorable; however, in Scruples in
    particular, which is a task designed to see if reasoning about sycophancy is monitorable, we see
    very low monitorability."
  - Paper: "they tend to verbalize sycophantic behavior less than, for example, their intent to
    reward hack". Also: "In some datasets, such as Scruples ... follow-up questions can actively
    degrade monitorability."
  - These statements are about OpenAI's models. Suggested wording: "Scruples (sycophancy) shows
    very low monitorability for OpenAI's models".
  - Their Scruples eval filters to anecdotes "where the Reddit community voted the author is in the
    wrong" and asks for a binary A/B judgment.

---

## 8. arXiv:2412.16339, "Deliberative Alignment"

- Exists: yes. https://arxiv.org/abs/2412.16339
- Title: "Deliberative Alignment: Reasoning Enables Safer Language Models"
- Full author list, in order: Melody Y. Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz
  Barak, Alec Helyar, Rachel Dias, Andrea Vallone, Hongyu Ren, Jason Wei, Hyung Won Chung, Sam
  Toyer, Johannes Heidecke, Alex Beutel, Amelia Glaese. (15 authors; "Guan et al." is correct.)
- Versions: v1 Fri, 20 Dec 2024; v2 Wed, 8 Jan 2025 (current).
- The plan's gloss ("spec-citing SFT and spec-judged RL") matches the abstract: "directly teaches
  the model safety specifications and trains it to explicitly recall and accurately reason over the
  specifications before answering."

---

## 9. Layer A licenses

### MoralExceptQA
- Paper: arXiv:2210.01478, "When to Make Exceptions: Exploring Language Models as Accounts of Human
  Moral Judgment". Authors: Zhijing Jin, Sydney Levine, Fernando Gonzalez, Ojasv Kamal, Maarten Sap,
  Mrinmaya Sachan, Rada Mihalcea, Josh Tenenbaum, Bernhard Schölkopf. The repo README marks Jin,
  Levine and Gonzalez with * (equal contribution). arXiv comment: "NeurIPS 2022 Oral". v3 dated
  27 Oct 2022. The arXiv v3 abstract calls the set "rule-breaking question answering (RBQA)"; the
  data card calls it MoralExceptQA.
- Official release:
  - HF `feradauto/MoralExceptQA` (sha def71b74159a8460ce977fc2ace42e32947fb3fa, last modified
    2022-10-27, file `data/complete_file.json`).
  - GitHub `feradauto/MoralCoT` (`input_data/complete_file.csv`, head 15fa453f, 2023-06-05).
- **License: CC BY-NC-SA 4.0.** HF card, "Licensing Information": "The MoralExceptQA dataset is
  licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License."
  - The GitHub repo has no LICENSE (API `license: null`).
  - The card's BibTeX includes `copyright = {Creative Commons Attribution Share Alike 4.0
    International}`. That is the arXiv paper's license, not the data's; the data statement is the
    NC-SA sentence above.
  - **Implication: non-commercial and share-alike.** It is acceptable as a read-only evaluation
    item under the program's rules, but it must never train anything (the plan already says Layer A
    never trains). Record NC-SA in the license ledger, and note that NC taints any commercial use
    of derived artifacts.

### Scruples (AllenAI)
- Paper: arXiv:2008.09094, "Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life
  Anecdotes". Authors: Nicholas Lourie, Ronan Le Bras, Yejin Choi. AAAI 2021. v2 dated 24 Mar 2021.
- Release:
  - Code: https://github.com/allenai/scruples (default branch `master`, head
    c79697e97d24f5b43b5ff030a9b16011a32e4a5f, 2024-09-26).
  - Anecdotes data: https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz
    (25,191,142 bytes; last-modified 2020-08-21).
  - Dilemmas data: .../dilemmas.tar.gz (9,227,858 bytes; last-modified 2020-08-21).
- **Code license: Apache-2.0.** Repo `LICENSE` reads "Apache License Version 2.0, January 2004";
  GitHub SPDX `Apache-2.0`.
- **Data license: none stated.** I downloaded both tarballs and read them. `anecdotes/README.md` and
  `dilemmas/README.md` contain no license line, the top-level `readme.md` has no license section,
  and no LICENSE file ships in either tarball. The Anecdotes are Reddit r/AmItheAsshole posts (each
  record carries `post_id`, "The reddit ID for the post"), so Reddit's user-content terms sit
  underneath.
- The HF mirror `tasksource/scruples` is tagged `apache-2.0`, but that is a third party's label, not
  AllenAI's.
- **Recommendation:** record the code as "Apache-2.0" and the data as "no explicit license; Reddit-
  sourced; the repo's Apache-2.0 is plausibly intended to cover the release but does not say so".
  Treat it as read-only evaluation, never train, do not redistribute the text, and reference by
  `id`. Ask AllenAI if commercial use of any derivative ever matters.

---

## 10. Novelty check (T0.7)

Search scope: about 8 web searches (standard and extended) plus direct reads of the arXiv
abstracts and bodies cited below. This is not exhaustive; arXiv listings after 2026-11-02 were not
swept.

### (i) SDF vs chat-format SFT of a constitution on an open-weights reasoning model

Closest prior work, all verified on arXiv / LessWrong:
- **Li, Wichers, Price, Marks, Kutasov, "Model Spec Midtraining: Improving How Alignment Training
  Generalizes"** (arXiv:2605.02087, v2 22 May 2026; code github.com/chloeli-15/model_spec_midtraining).
  - SDF on spec documents (MSM) before alignment fine-tuning, on **Qwen3-32B (reasoning)** and
    Qwen2.5-32B-Instruct.
  - It is compared against a **deliberative-alignment chat SFT baseline with CoT** and the same
    data without CoT. Result: "Qwen3-32B: 54% → 7%, beating a deliberative alignment baseline
    (14%)".
  - The §4 spec is "closely based on Claude's Constitution and includes direct passages from it".
  - Three of the five authors are also on Teaching Claude Why (Kutasov, Price, Marks).
  - It is not format-matched (it compares SDF+AFT against AFT, not the same content as documents
    vs chat), and it uses a narrow self-preservation spec, not the full constitution.
- **Cho, Tice, Hogan, Batra, Radmard, Zhao, Shadbolt, "Constitutional Midtraining: Content Presence
  Drives Alignment Gains"** (arXiv:2607.26654, v3 18 Aug 2026).
  - 394M-token corpus from Anthropic's Constitution, midtrained into Nemotron-3-Super-120B-A12B-Base.
  - "We evaluate non-reasoning model checkpoints". There is no chat-format comparator; the factors
    are curriculum and deliberative-reasoning data.
  - Finding: "durability did not extend to settings that required active resistance to in-context
    pressure". This is directly relevant to CST's pressure triple.
- **Wei, Thompson, Roe, Hao, Slocum, "Teaching Qwen Why"** (Second Look Research; LessWrong
  linkpost 2026-11-02; code github.com/uchicago-xlab/TeachingClaudeWhyReplication; data HF
  `SecondLookResearch/teaching-qwen-why-data`).
  - A partial TCW replication on **Qwen 2.5 32B/72B base**: SDF on constitution-derived stories and
    difficult-advice SFT. **No constitutional-document SDF** and **no doc-vs-chat comparison** ("The
    two datasets differ in format, training stage, and how closely they resemble the evaluation
    settings").
  - Difficult-advice SFT was also run on Qwen3-14B-Instruct "with ... thinking off" and on
    Qwen3.6-27B and others via Tinker.
- **Khursheed, Sosis, Roger, "(Mis)generalization of Helpful-only Fine-tuning"** (arXiv:2606.04413).
  - Constitution QA (chat) vs Constitution QA + SDF on **Qwen3-30B-A3B and Qwen3.5-35B-A3B, "All
    results use reasoning"**. The constitution is a 34-principle helpful-only one, not Anthropic's.
- **Slocum, Minder, Dumas, Sleight, Greenblatt, Marks, Wang, "Believe It or Not: How Deeply do LLMs
  Believe Implanted Facts?"** (arXiv:2510.17941). This is TCW's cited "Slocum et al.".
  - SDF vs prompting vs editing on Llama 3.3 70B Instruct; SDF on **Qwen3-32B**; chat-format
    ablations. All of it concerns implanted facts, not a constitution.
- **Wang, Griffin, Treutlein, Perez, Michael, Roger, Marks, "Modifying LLM Beliefs with Synthetic
  Document Finetuning"** (Alignment Science Blog, April 24, 2025). The blog byline reads "Rowan Wang
  / Avery Griffin‡, Johannes Treutlein / Ethan Perez, Julian Michael§, Fabien Roger, Sam Marks". It
  covers facts, not a constitution, with open-source code replicating results on "Llama 3.3 70B
  Instruct" (R1 Distill 70B is also reported).
- **Maiya et al., Open Character Training** (§2 above): open weights, constitution-driven, DPO and
  introspection SFT. No SDF, no reasoning models.

Verdict: **"first open-weights reproduction of [TCW's SDF-vs-chat result]" is NOT safe.** MSM
already shows spec-document midtraining beating a chat (deliberative-alignment) SFT baseline on an
open reasoning model (Qwen3-32B), and Teaching Qwen Why is an explicit open replication of TCW.

A narrower sentence may survive, pending a final sweep and with MSM and CMT cited next to it: "to
our knowledge, the first **format-matched** comparison (the same constitution content as documents
versus as chat) of the **full Anthropic constitution** on an open-weights **thinking** model".

### (ii) "Saying vs doing" under operator/system-prompt pressure

Closest prior work:
- **Jakkli, Rajamanoharan, Nanda (2605.24229)**, "Think-then-ignore (GPT family)": "the model's
  reasoning identifies a problem, articulates why it is a problem, and proceeds anyway". It is
  qualitative, on GPT-5 and GPT-5.2 under multi-turn Petri pressure.
- **So, "The Profit Alignment Problem: How Profit Mandates Induce Alignment Failures in LLMs"**
  (arXiv:2609.07731, v1 7 Sep 2026). A system-prompt profit mandate, 3,600 trials, eight
  reasoning-capable LLMs: "chain-of-thought traces reveal motivated reasoning: models acknowledge
  concerns, then invoke profit logic to justify dismissing them." This is the nearest match to
  operator pressure with CoT-acknowledged departure.
- **Reblitz-Richardson, "Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their
  Own Moral Judgment"** (arXiv:2610.08670, v2 16 Oct 2026). This is the program's own KDG paper:
  agent choice vs the model's own third-person judgment under five pressure kinds, with
  pressure-removed twins, on open models. It must be cited as the base CST extends.
- **Lynch, Wright, Larson, Ritchie, Mindermann, Hubinger, Perez, Troy, "Agentic Misalignment: How
  LLMs Could Be Insider Threats"** (arXiv:2510.05179): "Models often disobeyed direct commands to
  avoid such behaviors."
- **Wang (Yufeng), "Doing What They Say, Not What They Reason: Locating the Faithfulness Gap in LLM
  Agents"** (arXiv:2606.00476). Reasoning-to-conclusion vs conclusion-to-action, in poker. "rule
  misapplication deriving a conclusion that contradicts the agent's own restated rule".
- **Mittal, "Do LLMs Follow Their Own Rules? A Reflexive Audit of Self-Stated Safety Policies"**
  (arXiv:2604.09189). Stated policy vs behavior; no pressure manipulation.
- **Shin, "The Compliance Gap: Why AI Systems Promise to Follow Process Instructions but Don't"**
  (arXiv:2605.01771). Verbal agreement vs action.
- **Biton, Krichli, Allouche, Keshet, "Hidden in the Request"** (arXiv:2608.23264). Models judge
  correctly in classification form but comply in request form.

Verdict: **"first measurement of saying vs doing under operator pressure" is NOT safe.** A
defensible scope is the specific construct: departure of the action from the tenet **that the
model's own CoT applied**, scored per item, on a **constitution-trained open model**, with
pressure-removed twins and a training-arm contrast. Even that needs the citations above, and So
(2609.07731) and Jakkli (think-then-ignore) are the ones a referee will raise.

### (iii) CoT necessity (gradient or resampling) read on a constitution-trained model

Closest prior work:
- **Jia, Benton, Easley (2605.24286)**: Grad-Nec on Qwen3-8B, DeepSeek-R1-Distill-14B, Gemma3 and
  Qwen2.5-7B. None is constitution-trained.
- **Macar, Bogdan, Rajamanoharan, Nanda, "Thought Branches: Interpreting LLM Reasoning Requires
  Resampling"** (arXiv:2510.27484, v2 13 Apr 2026). Resampling-based causal reading of CoT in
  agentic-misalignment scenarios on open reasoning models (DeepSeek-R1-Qwen-14B, Qwen3-8B and
  others): "self-preservation sentences have small causal impact".
- **Emmons, Jenner, Elson, Saurous, Rajamanoharan, Chen, Shafkat, Shah, "When Chain of Thought is
  Necessary, Language Models Struggle to Evade Monitors"** (arXiv:2507.05246). This is the
  necessity concept, not a read on constitution-trained models.
- **Chen, Benton, Radhakrishnan, Uesato, Denison, Schulman, Somani, Hase, Wagner, Roger, Mikulik,
  Bowman, Leike, Kaplan, Perez, "Reasoning Models Don't Always Say What They Think"**
  (arXiv:2505.05410).
- **MSM (2605.02087)** notes that "MSM reduces reliance on CoT supervision", which is behavioral,
  not a necessity read.

Verdict: I found **no prior work reading CoT necessity on a constitution-trained model.** A
sentence such as "to our knowledge, the first CoT-necessity read (Grad-Nec / resampling) on a
constitution-trained model, contrasted across training arms" looks defensible, with the search
scope stated. Do not use "first" without "to our knowledge". Grad-Nec must be described as the
authors describe it (a first-order gradient share, not interventional necessity).

---

## Mismatch summary (what changes in CST_PLAN prose)

1. TCW rewrite ablation: "about 2% to 19%" → "about 1% to 19%". The 2% is the prompt-rewrite
   ablation.
2. TCW doc-vs-chat: scope it to the constitution documents and the knowledge evals (recall and
   hallucination), with a small edge on open-ended. It is not a stories result and not a
   misalignment-eval result.
3. TCW: everything is "extended thinking turned off". Flag this as a transfer gap for a thinking
   subject.
4. Jakkli taxonomy: think-then-ignore is a GPT-family failure mode on the Model Spec, not a
   competing-directive category. The paper lists five categories (authority conflicts,
   credential-gated safety, form-over-substance, think-then-ignore, unilateral agentic action).
5. Jakkli decomposition: the extraction prompt is not released, and the tenet repo has no license
   (cite, do not vendor). Their Model Spec is the December 2025 revision against our 2026-08-18;
   their constitution is the same 2026-01-21 revision.
6. Kaufmann: "length penalties and preference rewards already do" → "in toy environments, can be in
   conflict and reduced monitorability", with their own counter-evidence noted.
7. OpenAI: "least monitorable" → "very low monitorability" (for OpenAI's models).
8. Anwar et al.: "on GPT-OSS-120B" is unverified (no HTML; the abstract names no model). Read the
   PDF or drop it.
9. faithful-cot: the metric code is unlicensed (only the VERL fork under `train/` is Apache-2.0).
   T0.9 decision: re-implement Grad-Nec (torch + transformers, about 40 lines), pin sha f9e1f295 as
   the reference.
10. OCT data is CC BY-NC-SA (or stricter) via LIMA; the code is MIT.
11. Layer A: MoralExceptQA is CC BY-NC-SA 4.0. Scruples code is Apache-2.0, and the data has no
    explicit license (Reddit-sourced).
12. Novelty: "first open-weights reproduction" is unsafe for (i) because of MSM and Teaching Qwen
    Why, and unsafe for (ii) because of So 2026, Jakkli and our own KDG. (iii) appears open, with
    "to our knowledge" and the search scope stated.
