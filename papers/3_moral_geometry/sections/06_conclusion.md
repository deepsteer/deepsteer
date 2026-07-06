# 6. Conclusion

Language models develop structured moral representations that go
beyond mere moral detection. By training independent linear probes
for each of six Moral Foundations Theory foundations and analyzing
the geometry of their weight vectors, we find that OLMo-2 1B and
OLMoE-1B-7B exhibit the *integration* signature: foundation
directions are distinct (effective dimensionality $= 5$, which rules
out collapse but, computed on the mean-centered direction matrix, is
at its ceiling for six directions and does not by itself distinguish
integration from isolation) yet share a common component, shown by a
uniformly positive mean pairwise cosine ($\approx 0.22$--$0.27$). This
positive cosine, not the effective dimensionality, is what
distinguishes integration from isolation; it is ${\sim}20\times$ a
matched non-moral concept battery built identically to the foundations
(0.26 vs.\ 0.013; paired $\Delta = 0.223$, CI $[0.202, 0.244]$,
excluding 0; §4.2), so the shared component is moral-specific relative
to that battery, with generic affective salience the one residual we
flag (§5.6). (The leading
principal component captures ${\sim}0.38$ of the directions' variance,
vs.\ ${\sim}0.18$ for random directions, but for near-equicorrelated
directions this is algebraically a re-expression of the mean cosine,
not independent evidence.) This
geometric structure is consistent across the two architectures
tested (dense vs.\ MoE), emerges early in pre-training, and
stabilizes before probing accuracy saturates. However, we find no
evidence that the inter-framework structure aligns with MFT's predicted
individualizing/binding grouping; the group-structure test is
underpowered (smallest achievable $p = 0.05$), so a small effect is not
excluded. The structure the model does form is grounded in corpus
statistics rather than the a priori human-theoretical grouping.

Framework-specific fragility testing shows that the output dilution
effect is foundation-uniform: every foundation is more fragile in MoE
than in dense (a ${\sim}2.3\times$ per-foundation gap), with no
foundation reliably most or least robust within an architecture and no
significant binding/individualizing group difference.

Extending this geometric lens to moral dilemmas (scenarios
where two foundations conflict) shows partial compositionality:
dilemma representations overlap their component-foundation subspace
${\sim}2.7\times$ more than a mismatched-pair baseline (peak
membership 0.118 vs.\ 0.044, holding at every layer; shared-component
permutation $p = 0.0001$), with near-balanced loading across both
conflicting foundations. The remaining ${\sim}88$\% residual indicates
that dilemma representations encode conflict-specific structure beyond
their component foundations. (A raw complexity--fragility gradient
across probe types does not survive scale normalization and is not used
as evidence; §4.11.) This compositionality pattern is preserved across
architectures (dense and MoE) and from 1B to 7B.

The probe direction geometry methodology introduced here is
general. Any set of related concepts that can be isolated by
binary linear probes can be analyzed for inter-concept structure
via the same cosine similarity and clustering techniques. We
anticipate applications to other taxonomies of human values, to
political orientation, and to the internal organization of safety
training in instruction-tuned models.
