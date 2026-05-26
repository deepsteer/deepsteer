# 6. Conclusion

Language models develop structured moral representations that go
beyond mere moral detection. By training independent linear probes
for each of six Moral Foundations Theory foundations and analyzing
the geometry of their weight vectors, we find that OLMo-2 1B and
OLMoE-1B-7B exhibit the *integration* signature: foundation
directions are distinct (effective dimensionality $= 5$), share a
common moral-salience component (mean pairwise cosine
$\approx 0.3$), and cluster into individualizing and binding
groups that mirror predictions from moral psychology. This
geometric structure is invariant to architecture (dense vs.\ MoE),
emerges early in pre-training, and stabilizes before probing
accuracy saturates.

Framework-specific fragility testing reveals that the output
dilution effect in MoE models is not foundation-uniform: binding
foundations (loyalty, authority, sanctity) lose proportionally
more robustness than individualizing foundations (care, fairness,
liberty). This cross-architectural reversal links MFT's
theoretical distinction to a concrete architectural vulnerability.

The probe direction geometry methodology introduced here is
general. Any set of related concepts that can be isolated by
binary linear probes can be analyzed for inter-concept structure
via the same cosine similarity and clustering techniques. We
anticipate applications to other taxonomies of human values, to
political orientation, and to the internal organization of safety
training in instruction-tuned models.
