# 6. Conclusion

Language models develop structured moral representations that go
beyond mere moral detection. By training independent linear probes
for each of six Moral Foundations Theory foundations and analyzing
the geometry of their weight vectors, we find that OLMo-2 1B and
OLMoE-1B-7B exhibit the *integration* signature: foundation
directions are distinct (effective dimensionality $= 5$), share a
common moral-salience component (mean pairwise cosine
$\approx 0.22$--$0.27$), and span a high-dimensional moral subspace. This
geometric structure is consistent across the two architectures
tested (dense vs.\ MoE), emerges early in pre-training, and
stabilizes before probing accuracy saturates. However, the
inter-framework structure does not align with MFT's predicted
individualizing/binding grouping, indicating that the model's
moral taxonomy is empirically grounded in corpus statistics rather
than organized along human-theoretical lines.

Framework-specific fragility testing reveals that the output
dilution effect in MoE models is not foundation-uniform:
sanctity/degradation is the most robust foundation in the dense
model but the most fragile in MoE (6.2$\times$ ratio), far
exceeding the overall architectural gap.

Extending this geometric lens to moral dilemmas --- scenarios
where two foundations conflict --- reveals partial compositionality:
dilemma representations share ${\sim}10$\% of their variance with
the subspace spanned by component foundation directions (100$\times$
the null baseline), with near-balanced loading across both
conflicting foundations. The remaining ${\sim}90$\% residual and the
complexity--fragility gradient ($\sigma^*$: 4.72 $\to$ 3.12 $\to$
2.90) indicate that dilemma representations encode conflict-specific
structure beyond their component foundations. This compositionality
pattern is preserved across architectures (dense and MoE).

The probe direction geometry methodology introduced here is
general. Any set of related concepts that can be isolated by
binary linear probes can be analyzed for inter-concept structure
via the same cosine similarity and clustering techniques. We
anticipate applications to other taxonomies of human values, to
political orientation, and to the internal organization of safety
training in instruction-tuned models.
