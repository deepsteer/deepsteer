# 5. Discussion

*Drafting placeholder.* Subsections to populate during drafting:

## 5.1 What "compound scaling boundary" means here

Three components — mechanism, measurement, intervention — all run
at 1B; the behavioral phenomenon they target does not. The boundary
is not in any one component but in the *coupling* between them.

## 5.2 Why a clean null is the right framing

The probe activation shifts are small (Cohen's *d* = +0.03), the
behavioral EM rates are small (1.56 % vs. 0.69 % with overlapping
CIs), and the suppression primitive's 99.3 % effect on the probe
leaves the judge score unchanged. Each of these in isolation might
admit alternative readings; together, they constrain the
interpretation to "the 1B mechanism doesn't engage."

## 5.3 Connection to companion work

The fragility-locus signature reported in §4.4 is detected via the
methodology of Reblitz-Richardson (2026, *When Probing Accuracy
Saturates, Fragility Resolves*). Probing accuracy returns no signal
on insecure-code LoRA at 1B; fragility resolves a clean 2-3 layer
locus shift. Same model, same probe; different metric.

## 5.4 Limitations

- Single architecture (OLMo-2 1B); 7B replication is Phase E.
- Single LoRA recipe (Betley-style insecure-code); coupling
  generality across other emergent-misalignment-inducing recipes is
  open.
- Probe is a linear analog of the Wang et al. (2025) direction;
  SAE-decomposed features at 7B are the cleanest disambiguation
  test.
