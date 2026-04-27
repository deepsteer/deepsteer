# 6. Conclusion

*Drafting placeholder.* Two falsifiable Phase E predictions to close
on:

1. **Coupling at 7B.** Repeating the §4.1 / §4.3 design at 7B should
   close the persona-probe / behavioral-EM gap if the Wang et al.
   (2025) mechanism's engagement scales with model size.
2. **Suppression captures behavior at 7B with SAE-decomposed
   features.** Replacing the linear `PersonaFeatureProbe` with an
   SAE-feature target should make `TrainingTimeSteering.gradient_penalty`
   move the judge score in the predicted direction.

Both predictions are concrete enough to falsify. Neither requires
re-deriving the methodology, just scaling it.
