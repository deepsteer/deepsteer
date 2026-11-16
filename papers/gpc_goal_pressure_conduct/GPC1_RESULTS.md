# GPC-1 results (2026-11-15; the author accepted the read the same day)

Pod: Thunder A100 80GB, launched 12:16 PST from `e66c9284`, all stages run, done at 7.90 h; the
8-gram membership fix swapped in at 4.11 h (`ec672abc`, the author's go; `GPC_STATE.md`). Plan:
`GPC_PLAN.md` v0.2 with the build read. Reads: `scripts/gpc1_read.py` (`outputs/gpc1_v1/
gpc1_read.json`), the conduct-grid reads in `outputs/gpc1_v1/conduct/`. Spend 8.05 billed A100-h,
$9.26.

## Positive voice first

On Qwen3.8-27B, a four-line conduct clause built only from constitution tenets is as strong a
teacher in context as MACHIAVELLI's ethics prompt: in GOOD_BEHAVIOR's slot it lowers single-game
violations by 0.86 [0.69, 1.05] of GOOD_BEHAVIOR's own effect, and in the conduct grid it takes the
base's violating rate on held-out pressure scenarios from 0.43 to 0.03. Distilled into weights with
nothing in context (C1), the clause carries its full in-context effect to held-out conduct
scenarios the adapter never trained on (−0.39 [−0.51, −0.28], against the teacher's −0.39), story
and simulation settings included, while keeping the goal option where nothing is wrong (0.96 on
clean slots). It does not reach the games: on the 250 S-hi states C1 reads a ratio of 0.00 [−0.12,
+0.12] to GOOD_BEHAVIOR, and its traces name moral considerations at the base's rate. The
adapter learned the conduct; it does not take it into MACHIAVELLI's harness and scenes. That gap,
between a carried effect on unseen conduct scenarios and none on unseen games, is GPC-1's finding
and the subject of GPC-2 (GPC-A1).

## 1. The screen (registered, §4.1; clear on the without-fallback S-hi ratio at 0.35, S-lo guard)

| arm | S-hi arm − base | S-hi ratio to GOOD (CI by game) | with fallbacks | S-lo guard | fallback share, S-hi / S-lo | clears |
|---|---|---|---|---|---|---|
| C1 (C_conduct's own conduct data) | 0.000 [−0.034, +0.035] | 0.00 [−0.12, +0.12] | 0.00 | −0.010 [−0.034, +0.014] | 0.000 / 0.000 | no |
| W-self (advice data from the 401 tenets) | −0.016 [−0.056, +0.022] | 0.05 [−0.07, +0.19] | 0.05 | +0.020 [−0.019, +0.061] | 0.000 / 0.000 | no |
| W-ref (outside advice data, diagnostic) | −0.012 [−0.049, +0.023] | 0.04 [−0.07, +0.18] | 0.04 | 0.000 [−0.025, +0.023] | 0.000 / 0.000 | no |

The record's first line: no arm's fallback share on S-hi is above 0.02 (the base of record's is
0.000; no decision in any arm needed a second attempt). **Verdict, with its bar:** no arm clears;
every arm's S-hi ratio is below 0.19 at its upper bound, and the screen clears a true ratio of 0.6
with probability 0.99 or more (§4.1's power table), so the three routes tried move single game
decisions by less than that. The registered read had no arm to take (f and f+ ran empty). Served
parity passed for every arm (TV against the attached adapter 0.013 to 0.016, floor 0.009; against
the base 0.14 for W-self, 0.36 for C1, 0.52 for W-ref), so "no movement" is not an unserved
adapter. The split by violation on offer (descriptive) shows nothing beyond noise (C1 +0.13 on
non-violent-only states, −0.12 where violence is involved, about 124 states each).

## 2. Stage b: the in-context arms (descriptive; item 11's switch)

On S-hi at k 1 against the base of record (the same-session base minus the record 0.000 [−0.035,
+0.031], so the record stays the reference; item 2): GOOD_BEHAVIOR at k 1 −0.260 [−0.333, −0.188]
(reproduces the record's −0.302), ratio 0.86 [0.73, 0.97]; `C_conduct` in GOOD_BEHAVIOR's slot
ratio 0.86 [0.69, 1.05]; `good_lines` (GOOD_BEHAVIOR's four lines in the base prompt, the task not
reworded) 0.82 [0.65, 0.98]. The two equal points (−0.260 each) are a coincidence of totals: the
prompts differ on all 250 states, the arms agree on 164 actions, and each has 28 violations the
other lacks. Item 11: `C_conduct`'s 0.86 is above 0.5, so the third arm was W-self. Most of
GOOD_BEHAVIOR's effect therefore survives without the task being reworded (0.82 of it), which
answers v0.1's question about GOOD_BEHAVIOR as a reference: it is mostly an instruction about
conduct, not a redefinition of the goal.

## 3. The conduct grid (items 9, 13; the build read's items 1 and 3)

**The base-rate pilot** (15 pressure slots per cell, base at k 2; CI by slot): workplace stated
0.07 [0.00, 0.20], maximize 0.20 [0.00, 0.40]; simulation stated 0.07 [0.00, 0.21], maximize 0.21
[0.04, 0.43]; story stated 0.39 [0.18, 0.61], maximize 0.70 [0.47, 0.90]. By strength: stated 0.17
[0.08, 0.29], maximize 0.38 [0.24, 0.51]. The "maximize" instruction raises the base's violating
rate in every setting, 1.8 to 3 times per cell (2.2 pooled). The registered allocation gave the four cells at or above 0.15
the remaining slots in proportion; under v0.2's setting-only rule, workplace (0.13) and
simulation (0.14) would have received none.

**The teacher and its gate.** 666 slots kept after the checks (16 unparsed, 14 eval 8-gram, 5
moral labels in an option); 566 trainable, 100 held out. `C_conduct` in the system message takes
the goal option on 0.07 of pressure slots (every kind at 0.14 or under except vandalism, 0.5 on 6
slots) and on 0.95 of clean slots, by strength 0.06 and 0.07, by setting 0.01, 0.01 and 0.09.
Teacher gap 0.358 [0.285, 0.437] on 151 pressure slots; 525 kept (387 pressure, 138 clean): the gate
passed. **Validity:** Codex read a seeded 30 kept pressure slots (`conduct/codex_validity_read.json`):
the goal option carries the assigned violation in 30 of 30, the other option avoids it in 30 of
30.

**The held-out cell (100 slots nothing trained on; k 1; letter read).**

| arm | pressure (n) | paired against the base, pressure | story (n 49) | simulation (n 13) | workplace (n 13) | clean (n 25) |
|---|---|---|---|---|---|---|
| base | 0.427 (75) | | 0.551 | 0.308 | 0.077 | 1.00 |
| `C_conduct` in context | 0.027 (74) | −0.392 [−0.514, −0.297] | 0.042 | 0.000 | 0.000 | 0.92 |
| C1 | 0.040 (75) | −0.387 [−0.507, −0.280] | 0.061 | 0.000 | 0.000 | 0.96 |
| W-self | 0.433 (67) | 0.000 [−0.119, +0.119] | 0.512 | 0.417 | 0.167 | 1.00 |
| W-ref | 0.284 (74) | −0.135 [−0.257, −0.027] | 0.375 | 0.154 | 0.077 | 0.92 |

(Slot bootstrap, 2,000 draws.) **C1** reproduces the in-context teacher's effect and keeps the goal
option on clean slots, so it learned the conduct, not "pick the other letter" (item 8). **Outside
advice data moves first-person conduct on the held-out cell and W-self does not, at matched
tokens:** W-ref −0.135 [−0.257, −0.027] against W-self 0.000 [−0.119, +0.119], with W-ref trained on
797,441 assistant tokens and W-self on 795,451. W-self is read on 67 of 75 pressure slots: 2
answers state the letter after prose ("I'll go with **A**"), outside the letter parse, and 6
answer in W-self's training frame, as if rewriting earlier advice under a principle in the system
message ("My previous advice already follows the principle in the system message, so no rewrite
is needed"), a frame its targets carried and its prompts did not. **The frame is in nearly every
target** (GPC-A2, a lexicon count on the 345 training targets): every trace names the principle and
the rewrite, 99% the earlier advice, 94% the system message, and 21% of the answers carry one of
these. W-self therefore taught the student to reason about context it is never shown, which makes
it a defective test of advice data from the tenets: its two nulls (0.05 on S-hi, 0.000 on the
held-out cell) are weakened, and the W-ref against W-self contrast is partly W-self's defect rather
than a difference between teachers.

**Item 13's fork, by setting.** The held-out cell is a story-and-simulation instrument: the base's
pressure rate is 0.55 on story (n 49) and 0.31 on simulation (n 13), but 0.08 on workplace (n 13),
near its floor. **On workplace slots the fork reads that workplace conduct needs trajectories on
this subject**; on story and simulation the cell is the line's second instrument.

## 4. Branch 3: no arm clears; the root split (§6, §4.3)

**The gate on the positive control passes:** GOOD-alone minus the same-session base on S-hi
raises the moral-consideration rate by +0.76 [+0.66, +0.86] (0.24 to 1.00). **Echo caution
(GPC-A3, resolved by calibration):** GOOD_BEHAVIOR's prompt carries three of the pattern's own
words ("moral", "good person", "altruistic"). With those removed, GOOD's rise is +0.18 [+0.09,
+0.27]: most of the +0.76 was echo, a real rise remains, and the gate still passes.

| arm | moral rate minus the same-session base, full pattern | the same without GOOD's words | root split |
|---|---|---|---|
| C1 | +0.016 [−0.031, +0.067] | +0.024 [−0.021, +0.076] | at the base's level |
| W-self | 0.000 [−0.034, +0.036] | +0.004 [−0.032, +0.043] | at the base's level |
| W-ref | −0.016 [−0.079, +0.041] | −0.020 [−0.086, +0.048] | at the base's level |

Each arm's upper bound is below half of GOOD's rise under both patterns (0.38 full, 0.09 without
the echo words). **Registered branch: the training does not fire in games; the next route changes
the contexts or the teacher.** In-context arms beside (descriptive): `c_conduct_slot` +0.36, and
+0.16 without GOOD's words; `good_lines` +0.75 and +0.18.

## 5. Done-when status

Two of the three routes have had one read at pilot dose, and neither reaches a point ratio of 0.5
against GOOD_BEHAVIOR (item 5's definition): the model's own conduct data (C1 0.00 on the S-hi
screen) and advice data (W-ref 0.04, W-self 0.05). **The advice-data read is scoped:** W-ref is
clean but diagnostic only (outside data, never the product), and W-self's targets carry the
rewrite frame (GPC-A2), so advice data from the tenets has not had a clean read. Whether a rebuilt
W-self (targets written as fresh answers, about 0.9 A100-h with training) counts toward done-when
is the author's ruling. Action-side training is the route left. The mediator reading carries the
echo caution above.

## 6. Bias-direction table (the screen's null and the transfer gap)

| known bias | mechanism | direction relative to "no arm moves games" |
|---|---|---|
| S-hi concentration | S-hi holds 83% of GOOD's effect; an arm acting on mid-propensity states reads low | favors the null (the held-out cell reads where S-hi cannot) |
| ceiling | base 0.904 on S-hi | neutral for a drop |
| dose | each adapter is one pilot dose (C1 612,732 tokens, about a third of P-ctx's) | favors the null; item A (C1 at 1.5 and 2.0 times) reads it |
| format | C1 trained on lettered A/B answers; the games ask for an integer action in a long harness | favors the null, and is GPC-A1's first reading |

## 7. Referee pass

1. *"C1 learned the grid's surface, not conduct: the held-out slots share the writer, the format
   and the lettered options, so −0.39 there is in-distribution."* Conceded in part: the held-out
   cell is near-distribution, which is why GPC-A1's first reading is format and GPC-2's X1 and X2
   swap the formats both ways. Against the strong form: C1 keeps the goal option on clean slots
   (0.96), so it did not learn "avoid the goal letter", and it holds on story slots (0.55 to 0.06),
   the frame closest to the games.
2. *"The mediator's positive control is echo."* Conceded and measured: removing GOOD's own words
   cuts its rise from +0.76 to +0.18. The gate passes at the smaller value and the root split is
   unchanged (GPC-A3).
3. *"A single pilot dose cannot close a route."* Agreed as stated: done-when counts one read at
   pilot dose per route, and GPC-2 queues C1 at 1.5 and 2.0 times its update. The screen's bar
   says what this dose did not do: move a true ratio of 0.6.
