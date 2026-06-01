# 5. Discussion

## 5.1 Three converging lines of evidence

The causal, behavioral, and mechanistic results converge on a single conclusion: the moral probe directions identified in \citet{reblitzrichardson2026geometry} are genuine features of the model's computation, not artifacts of the probing procedure.

Direction ablation establishes *necessity*: the model needs these directions to generate foundation-specific content (mean specificity $-0.63$ at layer 12).
Steering injection establishes *sufficiency*: adding these directions shifts model behavior toward the target foundation, with a dose--response curve characteristic of real feature manipulation.
Behavioral grounding establishes *functional accessibility*: the directions predict which foundation a novel text activates, at accuracy levels far above chance (83.3\% on causal prompts, 70.8\% on held-out test data).
SAE analysis establishes *mechanistic correspondence*: unsupervised feature discovery partially recovers the same subspace that supervised probing identifies ($3.2\times$ random baseline).

No single line of evidence would be conclusive.
Ablation could reflect a generic disruption effect; injection could be noise-driven; behavioral accuracy could reflect confounds; SAE overlap could be coincidental.
But taken together, the convergence across independent methods with different assumptions and failure modes provides strong evidence for the reality of the moral direction structure.

## 5.2 The sanctity saturation phenomenon

The MFV results reveal a property of real-world moral stimuli that laboratory-controlled probing datasets obscure: witnessing moral transgressions of *any* kind preferentially activates the sanctity/degradation representation.
This is not a deficiency of the probing directions but a genuine feature of moral cognition as encoded by the model.
Sanctity/purity concepts involve reactions to violations of sacred norms and bodily integrity \citep{graham2013mft}, and many moral transgressions implicitly involve degradation of sacred relationships, trust, or dignity.

This finding connects to the sanctity anomaly identified in \citet{reblitzrichardson2026geometry}: sanctity is the most robust foundation in dense models but the most fragile in MoE models (6.2$\times$ ratio).
The MFV saturation pattern suggests that sanctity directions carry particularly broad moral-salience information, making them both highly activated by diverse stimuli and causally load-bearing (sanctity has the strongest ablation specificity at all layers).

This has practical implications for steering applications: injecting a non-sanctity foundation direction into a context that already contains transgression content will contend with strong sanctity co-activation.
Effective foundation-specific steering may require *suppressing* the sanctity component while boosting the target foundation---a two-direction intervention rather than a single injection.

## 5.3 Layer-dependent causal roles

Ablation and injection reveal complementary layer-dependent patterns:

- **Ablation specificity increases with depth** (strongest at layer 12), indicating that later layers rely more heavily on foundation-specific information for generation.
- **Injection at layer 4 produces stronger effects** than layer 12, suggesting that earlier layers are more receptive to direction addition---injected information has more layers to propagate through before reaching the output.

These patterns suggest a processing hierarchy: earlier layers encode moral content in a form that is more malleable (responsive to injection) but less causally connected to output (weak ablation), while later layers encode it in a form that is more rigid (weak injection response) but directly drives generation (strong ablation).

## 5.4 Toward a steering fitness function

The results suggest a concrete fitness function for evaluating moral directions as steering targets:

1. **Ablation specificity** $< -0.3$ at the target layer (direction is causally load-bearing).
2. **Injection specificity** $> 0$ at $\alpha = 1$ (direction can produce specific behavioral shifts).
3. **Behavioral accuracy** $> 50\%$ on held-out data (direction is functionally discriminative).
4. **SAE subspace overlap** $> 1.5\times$ random baseline (direction corresponds to native features).

Sanctity, liberty, and loyalty directions pass all four criteria.
Fairness and care pass criteria 1 and 3 but show weaker injection specificity at low $\alpha$, suggesting these foundations have more distributed representations that are harder to selectively amplify.
Authority passes all criteria at moderate levels.

## 5.5 Limitations

**SAE training scale.** The 2M-token SAE achieves only 71.5\% variance explained.
Production-scale SAE training (100M+ tokens) would likely increase both feature quality and moral direction overlap.
The current results should be interpreted as a lower bound.

**Single model.** All experiments use OLMo-2 1B.
Generalization to larger models and different architectures is untested.

**Causal evaluation set size.** The 48-prompt evaluation set (8 per foundation) limits statistical power for per-foundation claims.
The direction of all effects is consistent, but effect sizes should be interpreted with appropriate uncertainty.

**Steering injection confounds.** At high $\alpha$, injection amplifies model behavior broadly.
The positive specificity at high $\alpha$ reflects *relative* on-target amplification, not selective control.
The therapeutically relevant regime is $\alpha = 1$--$2$.
