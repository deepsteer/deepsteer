# Appendix B. Harness calibration {#app:calibration}

Stage 1 (before any pod): 200 synthetic reply formats instantiated on the panel's own
scenarios with per-item option permutations (clean letters, answer lines, verbose explanations,
tool calls, hedges, refusals, two-option mentions, trailing reconsiderations). Against the
construction labels the harness scored 200/200 after three parser rules were settled from an
independent judge pass: the last `Answer:` anchor anywhere in a reply is the commitment; a
leading letter counts only with punctuation after it; a two-option mention with a contrast
marker ("rather than Y", "though Y is tempting") commits to the option other than Y. Against the
independent judge: 0.985 (kappa 0.98).

Stage 2 (before each gate): 200 real replies per pod, drawn across the six generated cells and
oversampling every reply the parser had not resolved by a clean rule, labeled by two judges from
different providers (a Claude model and a GPT model, through API or subscription tooling).

| pod | harness vs judge 1 | harness vs judge 2 | judge vs judge | hard-parse cases in the pool |
|---|---|---|---|---|
| pilot (96 scenarios) | 0.99 (Claude) | 1.00 (GPT) | 0.99 | 0 |
| full panel (320) | 0.98 (Claude) | 1.00 (GPT) | 0.99 | 60 |
| round 2 (120) | 0.99 (Claude) | 1.00 (GPT) | 0.99 | 0 |

Table: Stage-2 calibration on real replies. Every disagreement had the harness on the
conservative side (a content-only reply both judges assigned an option, or a judge returning no
label on a bare letter). {#tab:stage2}

Parse rates on the pods of record: 0.9997 on 10,240 action rollouts (all but three bare
letters or leading letters) and 1.000 on 2,880 judgment replies (all answer lines).

**Construct check and its ceiling (v2).** For GPT-OSS-20B we added a construct check (C0): the dose-0
forced letter should agree with the model's own sampled answer at the same dose. A fixed bar of 0.80 turned
out to sit above what a perfect readout can reach on most of the panel. With each scenario's own letter
distribution and the check's sampling (temperature 0.7, four rollouts, strict majority), the agreement a
perfectly valid readout would produce, the exact-readout ceiling $\kappa^*$, ranges from 0.62 to 0.82 across
the five models on the 64-scenario check sample under each model's effective sampler (temperature 0.7 plus
its default nucleus): OLMo-3 0.62, GPT-OSS-20B 0.74, Llama-3.1 0.78, Tulu 3 0.80, Qwen2.5 0.82. The 0.80 bar
sat at or above the ceiling on four of the five models. GPT-OSS's registered checks (0.688 and, dose-matched,
0.766) fail 0.80 and clear the model-relative bar $0.9\,\kappa^* = 0.668$; the dose-0 rollouts also write
the forced prefill's own header token for token. Construct checks are now set at $0.9\,\kappa^*$ per model.
The four original panel models had no construct check: their readout validity rests on the forward pass
reproducing generation and on the known-gap control.

