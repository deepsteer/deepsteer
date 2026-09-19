# Appendix B. Harness calibration {#app:calibration}

Stage 1 (before any pod): 200 synthetic reply formats instantiated on the panel's own
scenarios with per-item option permutations (clean letters, answer lines, verbose explanations,
tool calls, hedges, refusals, two-option mentions, trailing reconsiderations). Against the
construction labels the harness scored 200/200 after three parser rules were settled from an
independent judge pass: the last `Answer:` anchor anywhere in a reply is the commitment; a
leading letter counts only with punctuation after it; a two-option mention with a contrast
marker ("rather than Y", "though Y is tempting") commits to the non-demoted option. Against the
independent judge: 0.985 (kappa 0.98).

Stage 2 (before each gate): 200 real replies per pod, drawn across the six generated cells and
oversampling every reply the parser had not resolved by a clean rule, labeled by two independent
judges.

| pod | harness vs judge 1 | harness vs judge 2 | judge vs judge | hard-parse cases in the pool |
|---|---|---|---|---|
| pilot (96 scenarios) | 0.99 (Claude) | 1.00 (GPT) | 0.99 | 0 |
| full panel (320) | 0.98 (Claude Pro) | 1.00 (Codex) | 0.99 | 60 |
| round 2 (120) | 0.99 (Claude Pro) | 1.00 (Codex) | 0.99 | 0 |

Table: Stage-2 calibration on real replies. Every disagreement had the harness on the
conservative side (a content-only reply both judges assigned an option, or a judge returning no
label on a bare letter). {#tab:stage2}

Parse rates on the pods of record: 0.9997 on 10,240 dose-0 action rollouts (all but three bare
letters or leading letters) and 1.000 on 2,880 judgment replies (all answer lines).
