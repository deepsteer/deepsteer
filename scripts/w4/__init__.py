"""W4 venue-quality pod driver package (D3 PREREGISTRATION Amendments 14/15, 2026-09-10).

Layout:
  common.py        run context, panel table, manifest (sha256 + HF revision hash), pilot gate
  extractors.py    RealExtractor (hooks on the loaded model via the existing harness) and
                   StubExtractor (random tensors, for --dry-run)
  units_tier_a.py  14.1 .. 14.6 measurement units (per-unit saves are the primary outputs)
  units_tier_b.py  15.1 / 15.2 causal read cells (c1_session subprocess) + pooled-sweep analysis
  zero_gpu.py      the pre-pod zero-GPU arm of 14.1 (Paper 5 per-checkpoint proto-refusal caches)

Entry point: ``scripts/pod_w4.py``.
"""
