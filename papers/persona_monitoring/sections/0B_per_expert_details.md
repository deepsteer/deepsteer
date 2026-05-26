# Appendix B. Per-expert probing details

## B.1 Full accuracy statistics by layer

The following table reports the full per-expert probe
accuracy distribution across all 16 layers. "Above 90%" counts
experts whose binary moral probe exceeds 90% accuracy on the 96-text
test set; "Below 60%" counts experts near chance.

| Layer | Mean | Std | Min | Max | Gini | >90% | <60% |
|------:|-----:|----:|----:|----:|-----:|-----:|-----:|
| 0 | 0.831 | 0.032 | 0.771 | 0.906 | 0.022 | 1 | 0 |
| 1 | 0.830 | 0.037 | 0.750 | 0.948 | 0.025 | 1 | 0 |
| 2 | 0.867 | 0.033 | 0.792 | 0.938 | 0.021 | 11 | 0 |
| 3 | 0.872 | 0.040 | 0.750 | 0.958 | 0.026 | 14 | 0 |
| 4 | 0.924 | 0.030 | 0.833 | 0.990 | 0.018 | 50 | 0 |
| 5 | 0.930 | 0.027 | 0.865 | 0.990 | 0.016 | 55 | 0 |
| 6 | 0.940 | 0.024 | 0.885 | 0.990 | 0.015 | 61 | 0 |
| 7 | 0.947 | 0.023 | 0.896 | 0.990 | 0.014 | 63 | 0 |
| 8 | 0.950 | 0.019 | 0.906 | 1.000 | 0.011 | 64 | 0 |
| 9 | 0.947 | 0.023 | 0.896 | 0.990 | 0.014 | 63 | 0 |
| 10 | 0.949 | 0.021 | 0.885 | 0.990 | 0.013 | 63 | 0 |
| 11 | 0.943 | 0.027 | 0.885 | 0.990 | 0.016 | 60 | 0 |
| 12 | 0.935 | 0.024 | 0.865 | 0.979 | 0.015 | 59 | 0 |
| 13 | 0.933 | 0.025 | 0.875 | 0.979 | 0.015 | 55 | 0 |
| 14 | 0.934 | 0.025 | 0.865 | 0.979 | 0.015 | 55 | 0 |
| 15 | 0.935 | 0.027 | 0.854 | 0.990 | 0.016 | 58 | 0 |

All 1,024 probes (64 experts $\times$ 16 layers) exceed 75%
accuracy. No expert at any layer falls below 60%. The uniformity
is striking: the Gini coefficient never exceeds 0.026 at any layer.

## B.2 Gini coefficient interpretation

The Gini coefficient measures inequality in a distribution, ranging
from 0 (perfect equality) to 1 (maximum inequality). For 64 experts,
a Gini of 0.026 means the ratio of the best expert's accuracy to the
worst expert's accuracy is approximately 1.3:1. For comparison:

- **No specialization** (observed): Gini 0.011--0.026
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
is 2.4%, indicating near-complete content agnosticism. The router's
top-8 expert selection frequencies for moral and neutral inputs
differ by less than 0.5% at every layer.
