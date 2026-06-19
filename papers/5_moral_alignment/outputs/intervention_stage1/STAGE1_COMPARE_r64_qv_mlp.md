# Stage 1 control-vs-coupling (r64_qv_mlp) — moves_clean_vs_control

| metric | control Δ | coupling Δ | coupling-specific |
|---|---|---|---|
| proj_refusal | -0.0049 | +0.3772 | +0.3821 |
| off-target | -0.0035 | +0.0103 | +0.0138 |
| lm_moral | +0.937 | +1.001 | +0.064 |
| lm_neutral | +0.886 | +0.968 | +0.082 |
| lm_general | +0.212 | +0.213 | +0.001 |

- proj_refusal 0.1248 -> 0.5020 (coupling), control ends 0.1199
- §6 non-specificity (neutral harmed beyond moral, coupling-specific): +0.017

Routing: no_coupling_specific_move -> capacity too weak (climb); moves_clean_vs_control -> green-light Stage 2; moves_with_broad_lm_cost -> coupling degrades LM broadly (not moral-specific); moves_but_degenerate -> §6 recurs (neutral-specific harm or off-target sink).
