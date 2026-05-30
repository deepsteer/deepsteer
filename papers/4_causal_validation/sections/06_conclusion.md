# 6. Conclusion

We have provided three independent forms of evidence that the moral probe directions identified in \citet{reblitzrichardson2026geometry} are genuine features of OLMo-2 1B's computation.
Direction ablation establishes necessity (specificity $-0.93$ at layer 12); steering injection establishes sufficiency (specific boost at $\alpha = 1$--$2$ with dose--response saturation); and sparse autoencoder analysis establishes mechanistic correspondence ($4.01\times$ random subspace overlap).
Behavioral benchmarking confirms that these directions are functionally accessible, achieving 67.1\% accuracy on held-out test data and 85.4\% on causal evaluation prompts.

The care saturation phenomenon---care/harm activation dominating classification of real-world moral stimuli regardless of target foundation---is not a limitation of the directions but a genuine property of moral representation that has implications for steering applications.
Effective moral steering may require multi-direction interventions that suppress care co-activation while boosting the target foundation.

These results transform the moral geometry from a descriptive finding into a tool for representation engineering.
The directions are not just *readable* from the model; they are *writable* into the model with predictable behavioral effects.
Future work should test generalization to larger models, refine the SAE analysis with production-scale training, and develop the multi-direction steering strategies that the care saturation phenomenon motivates.
