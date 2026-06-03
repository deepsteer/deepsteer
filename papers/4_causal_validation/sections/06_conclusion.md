# 6. Conclusion

We have presented three independent forms of evidence that the moral probe directions identified in \citet{reblitzrichardson2026geometry} are genuine features of OLMo-2 1B's computation.
Direction ablation establishes necessity (specificity $-0.63$ at layer 12); steering injection establishes sufficiency (specific boost at $\alpha = 1$--$2$ with dose--response amplification); and sparse autoencoder analysis establishes mechanistic correspondence ($3.2\times$ random subspace overlap).
Behavioral benchmarking confirms that these directions are functionally accessible, achieving 70.8\% accuracy on held-out test data and 83.3\% on causal evaluation prompts.

The sanctity saturation phenomenon (sanctity/degradation activation dominating classification of real-world moral stimuli regardless of target foundation) is not a limitation of the directions but a genuine property of moral representation that connects to the sanctity anomaly from \citet{reblitzrichardson2026geometry} and has implications for steering applications.
Effective moral steering may require multi-direction interventions that suppress sanctity co-activation while boosting the target foundation.

These results transform the moral geometry from a descriptive finding into a tool for representation engineering.
The directions are not just *readable* from the model; they are *writable* into the model with predictable behavioral effects.
Future work should test generalization to larger models, refine the SAE analysis with production-scale training, and develop the multi-direction steering strategies that the sanctity saturation phenomenon motivates.
