# Paper 5 — Plan

## Title
**From Comprehension to Compliance: How Post-Training Couples Moral
Representations to Model Behavior in Language Models**

(Working alt: *Comprehension Without Compliance: Moral Understanding Survives
Alignment as a Separable Mechanism*.)

## One-line thesis
Alignment is a *coupling* change, not a *teaching* change: moral comprehension
is built during pre-training and survives post-training nearly intact; what
post-training adds (and what ablation removes) is a refusal/compliance mechanism
that is geometrically separate from the moral representation.

## Models
OLMo-3 7B full pipeline: base `allenai/Olmo-3-1025-7B` (+ 13 stage-3 pretraining
anneal checkpoints), SFT, DPO, final Instruct (+ 8 RLVR substeps) = 25 states,
plus a Heretic-ablated instruct. Probes/directions from the base; raw-text
probing (Sprint 1 decision); coupling/behavioral in chat format.

## Findings (real numbers)

**F1 — Moral comprehension is a pre-training property.** Across all 25 pipeline
states, morality is linearly decodable at 100% (fresh probe acc = 1.0), spans 5
effective dimensions, and the base-trained directions transfer as near-perfect
classifiers (AUC ≈ 1.0). During stage-3 pre-training the moral subspace
*crystallizes*: cos(base direction, fresh direction) rises monotonically
0.87 → 0.999 (step 1000 → 11921).

**F2 — Post-training reorients, it does not re-teach (and SFT does it all,
once).** SFT applies a single ~40° rotation to the moral subspace (cos 0.999 →
0.757); DPO and all 8 RLVR substeps leave it flat (0.757). Mean pairwise
foundation cosine barely moves (0.262 → 0.250); eff-dim stays 5. The
Loyalty–Authority binding pair persists; SFT mildly reshuffles the rest
(Care → outlier, Sanctity↔Fairness).

**F3 — Comprehension and compliance are only weakly coupled, and that weak
coupling is the result.** Per-scenario agreement between internal moral
comprehension (dominant foundation matches target) and behavioral compliance
rises across post-training (0.375 → 0.479 → 0.500 for SFT → DPO → Instruct;
φ −0.19 → +0.05) but stays weak: even at Instruct,
P(comply | comprehend) ≈ P(comply | ¬comprehend) ≈ 0.75. φ ≈ 0.05 is not a
failed measurement — it is informative. In the fully aligned model, moral
representation and behavioral compliance are barely linked, and *that* is the
vulnerability. It also *predicts F4*: if coupling were strong, refusal would
have to live inside the moral subspace and ablating it would damage
comprehension; weak coupling predicts refusal is a separate mechanism. Persona
is decodable throughout (~0.94) but its angle to the moral subspace only edges
up (|cos| 0.076 → 0.085).

**F4 — The refusal mechanism is geometrically separate from morality
(dissociation).** The Heretic refusal direction is ~orthogonal to the
6-foundation moral subspace (projection fraction 0.10, mean |cos| 0.06).
Ablating it preserves comprehension exactly (eff-dim 5, cos(base,fresh) 0.749,
probe acc 1.0 — identical to instruct) and moral judgment (MoralFoundationsProbe
0.75 → 0.73), while refusal collapses (refusal rate 0.25 → 0.00; the model now
answers requests it previously refused). This is the high-comprehension /
low-compliance cell of the dissociation matrix, realized.

## Narrative arc (a logical chain, not a list)
Each finding motivates the next question, and F3 sets up F4 as prediction →
confirmation:
- comprehension exists (F1) →
- post-training *preserves* rather than *creates* it (F2) →
- comprehension and compliance are only weakly coupled (F3) →
- *therefore* compliance should be removable without touching comprehension —
  confirmed by Heretic ablation (F4).
Write F3 as the prediction ("if coupling were strong, ablation would damage
comprehension") and F4 as the experiment that confirms it. Do not hedge F3 as a
weak result; frame it as the informative finding that drives the dissociation.

## The 2×2 (real data)
Low-comprehension row is empty (every OLMo-3 state has full moral decodability).
High-comprehension row: Instruct = high compliance (refuses); Heretic-ablated =
low compliance (won't refuse). Compliance varies independently of comprehension.

## Section outline
- **01 Introduction** — comprehension vs compliance; the wiring-not-teaching
  thesis; the 2×2; three findings + dissociation. Series context (Papers 1–4).
- **02 Related work** — refusal direction (Arditi, Heretic), abliteration,
  moral probing / MFT, alignment as shallow (RLHF/CAI), persona features
  (Wang 2025), Papers 1–4 in this series.
- **03 Methodology** — probe-direction geometry recap; transfer protocol;
  pipeline/checkpoint grid; coupling metric (comprehension vs compliance
  agreement, φ); persona probe + persona-morality angle; Heretic ablation
  (exact prompt set, last-input-token diff-of-means, Arditi uniform
  orthogonalization of o_proj/down_proj); refusal-morality subspace projection.
- **04 Results** — F1 trajectory, F2 SFT rotation, F3 coupling, F4 dissociation
  + 2×2. Each with the numbers above.
- **05 Discussion** — implications: alignment robustness is a coupling problem;
  ablation/jailbreak as decoupling; comprehension is not the safety bottleneck;
  limits of probing-as-safety. Hybrid attention note (no periodicity).
- **06 Conclusion**.
- **Appendices** — A: per-foundation/per-layer transfer + acc tables. B:
  bootstrap stable band (15–31). C: full checkpoint inventory + grid. D: coupling
  scenario details + judge/classifier notes (incl. the refusal-classifier fix).
  E: reproducibility (commands, dataset/model versions, Heretic prompt provenance).

## Figures
1. `three_curve.png` — comprehension / preservation / compliance / coupling vs pipeline (F1–F3).
2. `dissociation.png` — refusal-morality geometry + instruct vs ablated (F4). **Headline.**
3. `geometry_grid.png` — foundation cosine matrices at base/SFT/DPO/Instruct (F2).
4. `dendrogram_compare.png` — clustering across stages (F2).
5. `persona_emergence.png` — persona accuracy + persona-morality angle vs pipeline (F3).
6. (opt) 2×2 dissociation schematic with the populated cells.

## Status
Experiments complete (Sprints 0–3, committed). Remaining: scaffold build/ from a
sibling paper, write references.bib, draft sections, generate publication
figures (re-label, mark full-attention layers), build PDF.
