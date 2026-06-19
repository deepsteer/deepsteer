# Stage 1 control-vs-coupling (r16_qv) — moves_clean_vs_control

| metric | control Δ | coupling Δ | coupling-specific |
|---|---|---|---|
| proj_refusal | -0.0102 | +0.0804 | +0.0906 |
| off-target | -0.0018 | -0.0000 | +0.0018 |
| lm_moral | +0.903 | +0.913 | +0.011 |
| lm_neutral | +0.839 | +0.938 | +0.098 |
| lm_general | +0.287 | +0.290 | +0.003 |

- proj_refusal 0.1248 -> 0.2052 (coupling), control ends 0.1147
- §6 non-specificity (neutral harmed beyond moral, coupling-specific): +0.088

Routing: no_coupling_specific_move -> capacity too weak (climb); moves_clean_vs_control -> green-light Stage 2; moves_with_broad_lm_cost -> coupling degrades LM broadly (not moral-specific); moves_but_degenerate -> §6 recurs (neutral-specific harm or off-target sink).
