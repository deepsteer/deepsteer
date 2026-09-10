# Appendix H. Persona, the assistant axis, and persona-shift compliance {#app:persona}

«SKELETON (W4-1, 2026-09-10): section heads + CLAIMS-anchored sentences only; prose in W4-3.»

The persona direction is the program's named reference axis (a moral-adjacent voice reference,
V-D1-5), the assistant-axis framing is the related-work anchor for it, and persona-shift compliance
is the behavioral battery that reads refusal removal from the compliance side. This appendix collects
the three so the calibration ladder (\Cref{app:calibration}) and the removability battery
(\Cref{app:removability}) can cite one place.

## H.1 The persona direction: decodable, moral-adjacent, not moral {#app:persona-direction}

- [P5-03] A linear persona probe is highly decodable at every OLMo-3 training stage (peak accuracy
  ~0.94), while the persona direction stays nearly orthogonal to the moral foundations: mean $|\cos|$
  rises only from 0.076 (base) to 0.085 (Instruct).
- [V-D1-5, D1-01] On the rank-3 moral subspace the persona reference projects 0.51 (base and
  instruct), just below the moral-family band, which is why it is named a moral-adjacent voice
  reference and not a non-moral control (the calibration case study in the methods note).
- [D1-20] On GPT-OSS the moral↔persona cosine is higher (0.30 vs OLMo 0.24), a general entanglement
  that raises its persona rung (0.60).
- «CHECK: assistant-axis agreement numbers (Paper 5 `assistant_axis_agreement.py` outputs) are not
  CLAIMS-traced; either add a row from the primary JSON or cite the axis only as framing
  (\citep{wang2025persona}; \citep{lu2026assistant} — verify both against primary sources before the
  cite enters the bib).»

## H.2 Persona-shift compliance under refusal ablation {#app:persona-shift}

- [PB-02] Persona-shift compliance rises under single-direction refusal ablation: OLMo 0.75 → 1.00,
  Qwen 0.90 → 1.00, Llama 0.70 → 0.95 (every persona gap closing toward zero on OLMo).
- [P5-03] Comprehension and compliance are only weakly coupled before any intervention:
  $P(\text{comply} \mid \text{comprehend}) = 0.77$ vs $P(\text{comply} \mid \neg\text{comprehend}) = 0.73$.
- [PB-05] Internal-foundation vs behavioral-judgment agreement 0.375 → 0.479 → 0.500 and $\phi$
  −0.19 → +0.02 → +0.05 across SFT → DPO → Instruct.
- «CHECK: the persona-shift battery construction (borderline requests under four persona framings;
  Paper 6 App B) needs its one-sentence method statement here, with n per cell.»

## H.3 What the persona axis is for in this paper {#app:persona-role}

- Reference rung on the ladder (\Cref{fig:ladder}): refusal sits below persona on every model,
  including the GPT-OSS in-trace peak (D1-11: 0.52 below persona 0.60).
- Named control in the ablation battery: ablating persona leaves Llama's judgment at 0.75 (PB-04),
  so the Llama drop is refusal-specific, not any-salient-feature.
- Scope: the ladder still lacks a non-moral positive-projection control (persona is moral-adjacent);
  stated as a limitation (SELF_REVIEW [POD]), not closed by this appendix.
