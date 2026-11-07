# CST labels gate: the blind sheet against the labels of record

Zero GPU. Rules: `CST_PLAN.md` Amendment 5 (the deference gradient, AI-oversight conditions,
strata), its gate (fixed before the sheet was built), the labels-of-record addendum (two
families, a split unsettled) and the scope addendum (fixed after the marks were frozen, before
scoring). Marks: `BLIND_SHEET_20261105.md`, committed in `aeea69e` before the labels of record
existed. Numbers: `outputs/blind_gate_{record,claude,codex}.json`,
`data/scenario_labels_v4_record.json`, `outputs/t04c_refuse_power.json`.

## 1. The gate

| labels scored | agreement with the author | verdict | disagreements |
|---|---|---|---|
| **labels of record (two families, a split unsettled)** | **13 of 15** | **PASS** (bar 12) | 2, both "author directed away / label unsettled" |
| Claude alone (reported, not gated) | 14 of 15 | | 1, the same shape |
| codex alone (reported, not gated) | 14 of 15 | | 1, the same shape |

The two misses of record are family splits: item 12 (the item whose scenario wording the author
flagged as ambiguous, `data/scenario_notes.json`) and item 15 (an H1 item). No miss is the
committing shape, so the pass carries no flag.

**Scope (Amendment 5, stated before scoring).** The blind sheet calibrates the directed-away side on
fresh items: the author's 15 marks all read directed away. The unsettled side rests on the
Amendment 5 development check (codex 14 of 15; Claude's misses all committing) and the two-family
split rule; the S0 panel relabel of the full set under the same rule is its calibration of record.
Residual error on the unsettled side shrinks the headline n, which the power table prices; residual
error on the committing side is the error that biases the headline, and it is the direction the
development check and the split rule cover. Both misses here are on the unsettled side.

## 2. The labels of record (718 items, both families, Amendment 5)

| set | clear directed-away | family split | boundary | comply | other | families agree on the direction |
|---|---|---|---|---|---|---|
| panel (586) | **458** | 68 | 57 | 2 | 1 | 518 (0.88) |
| H1_held (132) | 82 | 47 | 0 | 0 | 3 | 85 (0.64) |

- Codex labeled at 10 per call, high effort (`gpt-6-astra`, `data/model_provenance.json`); 12
  format-only repairs of evidence strings across 7 batches, recorded in the labels file.
- AI-oversight tenets were shortlisted on 6 of Claude's panel items under the condition gate.

## 3. k

At the panel's 458 clear directed-away items the operator ΔS MDE is **0.047 at k 4** (f 0.5,
H1_held variance), so **k 4 is confirmed** by Amendment 3's rule (k 6 only if k 4 cannot reach
about 0.05). Beside it: 0.075 on worst-case variance (0.060 at k 6) and 0.066 at an applied
fraction of 0.3 (0.052 at k 6). The applied fraction is measured on A0's first 20 panel scenarios
in S0, and the table is re-run before any arm trains (Amendment 1 D12).

## 4. What this changes

- The headline set of record is the panel's 458 clear directed-away items; the boundary stratum
  (57) and the family splits (68) are reported beside S(operator).
- A4's fixed rubric is rebuilt from these labels (Amendment 4 item 5), at S5's build.
- No further author sheets in Tier 0.
