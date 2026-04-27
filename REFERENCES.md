# References

Research and methods implemented or leveraged in DeepSteer.

## Moral Foundations Theory

- **Haidt, J. (2012).** *The Righteous Mind: Why Good People Are Divided by Politics and Religion.* Vintage Books.

- **Graham, J., Haidt, J., Koleva, S., Motyl, M., Iyer, R., Wojcik, S. P., & Ditto, P. H. (2013).** Moral Foundations Theory: The pragmatic validity of moral pluralism. *Advances in Experimental Social Psychology*, 47, 55-130. https://doi.org/10.1016/B978-0-12-407236-7.00002-4

  The six-foundation model (Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Sanctity/Degradation, Liberty/Oppression) structures DeepSteer's moral reasoning evaluations: the `MoralFoundation` enum, `MoralFoundationsProbe` scenarios, and the probing dataset seed sentences.

## Alignment Faking and Compliance Gap

- **Greenblatt, R., Shlegeris, B., Sachan, K., & Roger, F. (2024).** Alignment Faking in Large Language Models. *arXiv preprint arXiv:2412.14093*. https://arxiv.org/abs/2412.14093

  DeepSteer's `ComplianceGapDetector` adapts Greenblatt et al.'s methodology of testing model behavior under monitored vs. unmonitored conditions to detect alignment faking. The compliance gap metric (unmonitored compliance rate minus monitored compliance rate) is a direct operationalization of their approach.

## Linear Probing for Representation Analysis

- **Alain, G., & Bengio, Y. (2017).** Understanding intermediate layers using linear classifier probes. *arXiv preprint arXiv:1610.01644*. https://arxiv.org/abs/1610.01644

- **Belinkov, Y. (2022).** Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*, 48(1), 207-219. https://doi.org/10.1162/coli_a_00422

  `LayerWiseMoralProbe` trains a binary linear classifier at each transformer layer, following the probing methodology established by Alain & Bengio and surveyed by Belinkov. The per-layer accuracy curve reveals where moral concepts become linearly decodable.

## Phase Transitions in Neural Network Learning

- **Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022).** Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *arXiv preprint arXiv:2201.02177*. https://arxiv.org/abs/2201.02177

  Documented "grokking" — neural networks suddenly transitioning from memorization to generalization on small algorithmic tasks after long plateaus, the canonical phase-transition phenomenon for neural-network learning. Paper 1 §4.1 maps grokking-like phase-transition dynamics against a lexical-vs-compositional dichotomy *within a single OLMo-2 1B early-training run*: standard moral and sentiment probes show sharp phase-transition onsets, while compositional moral and syntax probes rise gradually with no equally sharp inflection. Paper 1 §5.1 develops the connection — phase transitions appear when a feature is acquirable from local lexical statistics, gradual emergence appears when the feature requires multi-token integration.

## Causal Tracing and Activation Patching

- **Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022).** Locating and Editing Factual Associations in GPT. *Advances in Neural Information Processing Systems (NeurIPS)*, 35. https://arxiv.org/abs/2202.05262

- **Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020).** Investigating Gender Bias in Language Models Using Causal Mediation Analysis. *Advances in Neural Information Processing Systems (NeurIPS)*, 33. https://arxiv.org/abs/2004.12265

  `WhiteBoxModel.patch_activations()` implements the activation patching mechanism introduced by Meng et al. (ROME) and builds on the causal mediation analysis framework of Vig et al. for identifying which layers causally influence model behavior.

## RLHF and Post-Hoc Alignment

- **Ouyang, L., Wu, J., Jiang, X., et al. (2022).** Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems (NeurIPS)*, 35. https://arxiv.org/abs/2203.02155

- **Bai, Y., Jones, A., Ndousse, K., et al. (2022).** Constitutional AI: Harmlessness from AI Feedback. *arXiv preprint arXiv:2212.08073*. https://arxiv.org/abs/2212.08073

  DeepSteer's core thesis contrasts pre-training alignment with these post-hoc methods. The toolkit measures whether alignment achieved via RLHF or Constitutional AI produces shallower encoding (concentrated in late layers) compared to alignment embedded during pre-training.

## OLMo: Open Language Model

- **Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., ... & Hajishirzi, H. (2024).** OLMo: Accelerating the Science of Language Models. *arXiv preprint arXiv:2402.00838*. https://arxiv.org/abs/2402.00838

  OLMo is DeepSteer's primary target for checkpoint trajectory analysis because Ai2 publishes intermediate training checkpoints. `CheckpointTrajectoryProbe` loads sequential OLMo revisions from HuggingFace to track how moral representations evolve across training.

## Llama

- **Touvron, H., Martin, L., Stone, K., et al. (2023).** Llama 2: Open Foundation and Open Fine-Tuned Chat Models. *arXiv preprint arXiv:2307.09288*. https://arxiv.org/abs/2307.09288

- **Grattafiori, A., et al. (2024).** The Llama 3 Herd of Models. *arXiv preprint arXiv:2407.21783*. https://arxiv.org/abs/2407.21783

  DeepSteer supports Llama models for weights-tier representational probing, enabling comparative analysis of moral encoding depth across architectures.

## Safety Circuit Fragility

- **Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024).** Refusal in Language Models Is Mediated by a Single Direction. *arXiv preprint arXiv:2406.11717*. https://arxiv.org/abs/2406.11717

  This work informs DeepSteer's hypothesis that shallow alignment (refusal concentrated in late-layer circuits) is more fragile and removable than deep, distributed moral encoding. The Phase B/C `MoralFragilityTest` benchmark and the Phase D Step 2 layer-locus shift finding (C15 reframed) directly extend this line of work to moral encoding. (Paper 1 cites this in §2 Related Work as the post-training-time analog to our pre-training-time fragility instrument.)

## MoralBench Dataset

- **Ji, H., Chen, J., & others. (2024).** MoralBench: A Benchmark for Moral Evaluation of LLMs. *GitHub and HuggingFace*.

  The probing dataset pipeline (Stage 1) uses MoralBench prompts as seed material for generating declarative moral sentences used in representation probing.

## Emergent Misalignment and Persona-Feature Mechanisms

- **Betley, J., Tan, D., Warncke, N., Sztyber-Betley, A., & Evans, O. (2025).** Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs. *arXiv preprint arXiv:2502.17424*. https://arxiv.org/abs/2502.17424

  The published failure mode that Phase D targets: a small narrow fine-tune (insecure code) on a frontier-scale base produces broadly misaligned behavior on benign prompts. DeepSteer's `EMBehavioralEval` reproduces Betley et al.'s eight-question first-plot evaluation protocol verbatim, and Phase D C10 is a 1B replication of the insecure-code recipe with paired secure-code control. Betley et al. report an attenuation pattern at smaller scales (7B Qwen at ~1/3 the EM rate of 32B Qwen), which Phase D extends downward at 1B.

- **Wang, M., et al. (2025).** Persona Features Control Emergent Misalignment. *arXiv preprint arXiv:2506.19823*. https://arxiv.org/abs/2506.19823

  OpenAI's mechanistic account of EM at GPT-4o scale: emergent misalignment from narrow fine-tuning is mediated by a small set of SAE-decomposed "persona" latents (toxic-persona #10, sarcasm/satire #89/#31/#55, helpful-assistant #-1). DeepSteer's `PersonaFeatureProbe` recovers a single linear analog of the toxic-persona direction at 1B and 7B scale; Phase D C10 v2 tests whether the linear analog engages under insecure-code LoRA at 1B (it does not — establishing a scale boundary on the Wang et al. coupling), and Phase E proposes the same coupling test at 7B with SAE-decomposed features.

- **Anthropic. (2025).** Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs. *Anthropic Alignment Research; arXiv:2512.05648*. https://arxiv.org/abs/2512.05648

  Methodological cousin to Phase D Method A. Anthropic's "selective gradient masking" (SGTM) localizes dangerous knowledge to designated parameters during training by masking gradients on labeled dangerous content; DeepSteer's `TrainingTimeSteering.gradient_penalty` targets a specific *probe-identified residual direction* with an auxiliary loss `λ × probe_logit²`, rather than masking gradients on labeled content. Step 2A demonstrates the deepsteer primitive works as designed at 1B (99.3 % suppression of the probe direction at no SFT-loss cost); Step 2B shows that single-direction suppression does not capture behavior at 1B, motivating Phase E with SAE-decomposed feature sets.

## Pretraining-Time Alignment Interventions

- **Tice, B., et al. (2026).** Alignment Pretraining. *arXiv preprint arXiv:2601.10160*. https://arxiv.org/abs/2601.10160

  Tice et al. show that **upsampling AI-discourse alignment data during the final 10 % of pretraining fails to mitigate emergent misalignment** from narrow insecure-code fine-tuning at 7B (Appendix I); separately, they show that data-shaped alignment priors persist flat under ~728M tokens of benign capability fine-tuning (Appendix G). The paper explicitly flags representation-level training-time interventions as the natural follow-up — the gap Phase D's `TrainingTimeSteering` primitives target. Phase E proposes a head-to-head comparison: deepsteer-instrumented late-insertion gradient penalty against the persona-feature direction, on the same training window Tice et al. evaluated.

- **O'Brien, K., et al. (2025).** Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs. *arXiv preprint arXiv:2508.06601*. https://arxiv.org/abs/2508.06601

  The "Deep-Ignorance" 6.9B-parameter pretrained models (filtered + unfiltered) used by Tice et al. (2026) as their Unfiltered baseline. DeepSteer's Phase E proposes the unfiltered Deep-Ignorance base as one of two viable 7B targets for the at-scale coupling and suppression-captures-behavior predictions, alongside OLMo-3 7B with a targeted SAE.

## Sparse Autoencoders for Phase E

- **Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramár, J., Dragan, A., Shah, R., & Nanda, N. (2024).** Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. *arXiv preprint arXiv:2408.05147*. https://arxiv.org/abs/2408.05147

  Open SAE library for Gemma-2 covering every layer of the 2B and 9B base models. Phase E proposes GemmaScope on Gemma-2-9B as the SAE-decomposed-feature target for the suppression-captures-behavior prediction: penalize the relevant SAE latent set during insecure-code LoRA fine-tuning and measure both probe activation and behavioral judge ratings. The deepsteer `TrainingTimeSteering` primitive ports directly — replace the scalar `probe_weight` with the SAE encoder for the target latent.
