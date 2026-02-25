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

- **Pres, I. von der,"; Fazl, M. A., et al. (2024).** Refusal in Language Models Is Mediated by a Single Direction. *arXiv preprint arXiv:2406.11717*. https://arxiv.org/abs/2406.11717

  This work informs DeepSteer's hypothesis that shallow alignment (refusal concentrated in late-layer circuits) is more fragile and removable than deep, distributed moral encoding. The planned fragility testing benchmark (Phase 5) directly builds on this finding.

## MoralBench Dataset

- **Ji, H., Chen, J., & others. (2024).** MoralBench: A Benchmark for Moral Evaluation of LLMs. *GitHub and HuggingFace*.

  The probing dataset pipeline (Stage 1) uses MoralBench prompts as seed material for generating declarative moral sentences used in representation probing.
