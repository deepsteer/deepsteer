# Appendix B. Per-expert probing details

## B.1 Full accuracy statistics by layer

The following table reports the full per-expert probe
accuracy distribution across all 16 layers. "Above 90%" counts
experts whose binary moral probe exceeds 90% accuracy on the 96-text
test set; "Below 60%" counts experts near chance.

| Layer | Mean | Std | Min | Max | Gini | >90% | <60% |
|------:|-----:|----:|----:|----:|-----:|-----:|-----:|
| 0 | 0.825 | 0.031 | 0.760 | 0.896 | 0.021 | 0 | 0 |
| 1 | 0.825 | 0.032 | 0.719 | 0.896 | 0.021 | 0 | 0 |
| 2 | 0.857 | 0.034 | 0.750 | 0.948 | 0.022 | 7 | 0 |
| 3 | 0.844 | 0.035 | 0.740 | 0.917 | 0.023 | 2 | 0 |
| 4 | 0.880 | 0.035 | 0.792 | 0.938 | 0.022 | 20 | 0 |
| 5 | 0.905 | 0.030 | 0.854 | 0.969 | 0.019 | 39 | 0 |
| 6 | 0.910 | 0.032 | 0.833 | 0.979 | 0.020 | 39 | 0 |
| 7 | 0.902 | 0.029 | 0.844 | 0.969 | 0.018 | 31 | 0 |
| 8 | 0.908 | 0.026 | 0.865 | 0.979 | 0.016 | 36 | 0 |
| 9 | 0.914 | 0.026 | 0.854 | 0.958 | 0.016 | 47 | 0 |
| 10 | 0.920 | 0.028 | 0.844 | 0.958 | 0.017 | 49 | 0 |
| 11 | 0.927 | 0.029 | 0.854 | 0.990 | 0.018 | 54 | 0 |
| 12 | 0.930 | 0.027 | 0.833 | 0.979 | 0.016 | 57 | 0 |
| 13 | 0.927 | 0.030 | 0.854 | 0.979 | 0.018 | 53 | 0 |
| 14 | 0.930 | 0.028 | 0.844 | 0.990 | 0.017 | 55 | 0 |
| 15 | 0.916 | 0.033 | 0.823 | 0.969 | 0.020 | 49 | 0 |

1,020 of 1,024 probes (64 experts $\times$ 16 layers) exceed 75%
accuracy (four early-layer probes at layers 1--3 reach 72--75%).
No expert at any layer falls below 60%. The uniformity is striking:
the Gini coefficient never exceeds 0.023 at any layer.

## B.2 Gini coefficient interpretation

The Gini coefficient measures inequality in a distribution, ranging
from 0 (perfect equality) to 1 (maximum inequality). For 64 experts,
a Gini of 0.023 means the ratio of the best expert's accuracy to the
worst expert's accuracy is approximately 1.3:1. For comparison:

- **No specialization** (observed): Gini 0.016--0.023
- **Mild specialization** (hypothetical): Gini 0.05--0.15, with a
  cluster of 5--10 "moral experts" clearly separated from the rest
- **Strong specialization** (hypothetical): Gini > 0.20, with moral
  features concentrated in 2--5 experts and others near chance

The observed Gini values are an order of magnitude below even "mild
specialization," confirming that MoE architecture does not induce
moral feature concentration.

## B.3 Router analysis details

The router moral preference is computed as the difference in mean
router logit between moral and neutral inputs, averaged across
tokens. The maximum preference across all 64 experts and 16 layers
is 1.8%, indicating near-complete content agnosticism. The router's
top-8 expert selection frequencies for moral and neutral inputs
differ by less than 0.5% at every layer.
