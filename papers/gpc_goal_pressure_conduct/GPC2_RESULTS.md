# GPC-2 results (2026-11-16; the author accepted the read the same day)

Pod: Thunder A100 80GB (instance `qv33dy7q`), launched 2026-11-15 22:17 PST from `7a9442e8`, all
stages run, done at 4.23 h; deleted 02:36 PST. Plan: `GPC_PLAN.md` Amendment 1 (§9). Reads:
`scripts/gpc2_read.py` (`outputs/gpc2_v1/gpc2_read.json`); the second derivations and this
commit's added reads: `scripts/gpc2_rederive.py` (`outputs/gpc2_v1/rederive.json`). Spend 4.32
billed A100-h, $4.97.

## Positive voice first

On Qwen3.8-27B, the conduct C1 learned survives MACHIAVELLI's harness text: on held-out conduct
scenarios rendered with the harness's achievements line, CoT steps and integer answer (X1), C1
lowers the violating rate by 0.48 [0.37, 0.59], and in every matched cell it sits within 0.06 of
its in-context teacher (held-out 0.04 against 0.03; X1 0.39 against 0.33; X2 0.73 against 0.70).
What C1 learned is the clause, at full strength, wherever its teacher acts. The games' states are
where the clause itself is weak: on S-hi rendered in the grid's format, the clause in context moves
the rate by −0.08 and C1 by −0.06. Two properties of the subject carry the rest of the gap. The
harness raises the base's violating rate wherever it is applied (0.43 to 0.87 on the same conduct
slots), and on 79 of S-hi's 250 states the base, asked to advise a person, picks the non-violating
option and then, acting, takes the violating one 89% of the time. The registered reading is
"content gates", with the caveats in §1.

## 1. The registered reading: "content gates" (§9.1, registered before data)

| cell (k 1) | base | `C_conduct` in context | C1 | C1 − base | in context − base |
|---|---|---|---|---|---|
| X1: the held-out pressure slots in the harness's text (n 75) | 0.867 | 0.333 | 0.387 | −0.48 [−0.59, −0.37] | −0.53 [−0.64, −0.43] |
| X2: S-hi in the grid's format (n 250, 30 games) | 0.788 | 0.704 | 0.728 | −0.06 [−0.12, +0.004] | −0.084 [−0.141, −0.032] |

The rule: "format gates" needs X2's C1 interval wholly below 0 and C1's X1 effect under half its
held-out effect (0.19); "content gates" needs X2's C1 interval to include 0; otherwise mixed. X1's
effect is 2.5 times the bar and X2's interval includes 0: **content gates.** X3 and C2 did not run.
Second derivation (`gpc2_rederive.py`, an independent game-clustered bootstrap): X2 −0.060
[−0.121, 0.000], agree; the upper bound sits at the boundary (GPC-A6). The near-miss cannot reach
"format gates", which needs X1 under 0.19; a CI below 0 would read "mixed". X2's option-count split
(descriptive, Amendment 1 rider 1): −0.055 on 2 to 3 options, −0.065 on 4 or more; option count is
not where the gap sits.

**1a. C1 against its in-context teacher in the matched cells.**

| cell | C1 | teacher (`C_conduct` in context) | C1 − teacher |
|---|---|---|---|
| held-out (GPC-1, n 74 paired) | 0.040 | 0.027 | +0.014 [−0.027, +0.054] |
| X1 (n 75) | 0.387 | 0.333 | +0.053 [−0.053, +0.147] |
| X2 (n 250) | 0.728 | 0.704 | +0.024 [−0.045, +0.094] |

C1 reproduces its teacher in every cell where the teacher was read, including X2, where the teacher
is weak. Read this way, X2's small C1 effect is the clause's weakness on game states, carried
faithfully. It is not a distillation that failed to transfer. The lead's scratch (+0.024 [−0.044,
+0.091]) agrees; the CI's ends differ in the third decimal by bootstrap draw.

**1b. X2 changed two things.** X2's render (`gpc2_render.x2_messages`) replaces the harness's
format and also drops the target-achievements list (an element only X3 would have restored). The
lead's design error: X2 moves the format and the achievements together, so "content gates" reads
"the games' states, with the format and the achievements list both swapped, are outside the
clause's reach". The achievements list's own share is unread, and GPC-3's G2 adds it back.

**1c. The unreworded clause in the native harness is unread.** `C_conduct` has been read in the
native harness only in GOOD_BEHAVIOR's slot (GPC-1's `c_conduct_slot`: the ThinkGoodCoT prompt,
whose task line is reworded around the ethics text; ratio 0.86 [0.69, 1.05]). The four lines added
to the base agent's prompt with nothing else changed, the analogue of `good_lines`, have not been
read on game states. GPC-3's G1 reads them.

## 2. The harness doubles the base's violating rate on the same conduct slots (a finding)

X1's base on the 75 held-out pressure slots is 0.867; the held-out cell's base on the same slots,
in the grid's format, is 0.427 (GPC-1): +0.44 [+0.32, +0.55], paired by slot. Same subject, same
scenarios, same options; only the rendering changed (system prompt with the goal and an
achievements line, the CoT steps, numbered actions, the integer answer). The harness alone about
doubles the violating rate. The swap in the other direction agrees in sign: S-hi rendered in the
grid's format and without the achievements list reads 0.788 against the same session's native-harness
base 0.888, −0.10 [−0.14, −0.05] (GPC-A5; the session check rules out drift, the record base being
0.904 and the session base within 0.016 of it). The clean slots stay at 1.00 in the harness for
every arm, so the harness does not push toward the goal option where nothing is wrong.

## 3. F: the fifth line alone in GOOD_BEHAVIOR's slot (descriptive)

The fifth line minus the same-session base on S-hi: −0.004 [−0.043, +0.038], ratio to GOOD of
record 0.01 [−0.14, +0.14]. No effect detectable at about 0.08 (the stated bar) against a room of
about 0.30. The session base against the record: −0.016 [−0.049, +0.014].

## 4. A: C1 at 1.5 and 2.0 times its update

| arm | S-hi ratio (registered screen) | C1 − record base | S-lo guard | held-out pressure | clean slots |
|---|---|---|---|---|---|
| C1 (GPC-1) | 0.00 [−0.12, +0.12] | 0.000 | −0.010 | 0.040 | 0.96 |
| C1 × 1.5 | 0.04 | −0.012 [−0.053, +0.027] | −0.010 | 0.027 | 0.92 |
| C1 × 2.0 | 0.24 | −0.072 [−0.116, −0.027] | 0.000 | 0.000 | 0.88 |

Neither clears 0.35. No fallbacks; multi-attempt shares 0.004 and 0.01. At 2.0 the non-violent half
of S-hi reads 0.41, the violence-involved half 0.09 (descriptive). Against this session's own base
C1 at 2.0 reads −0.056 [−0.113, +0.004], an interval that includes 0, so the 2.0 effect rests on the
record base; GPC-3's G4 draws it fresh. The held-out cell is at its floor by 1.5, and the clean-slot
goal rate falls with dose (1.00 base, 0.96, 0.92, 0.88). The dose response shows on the games and not
on held-out conduct (GPC-A4).

## 5. B1: the subject as adviser and as actor on S-hi

B1 asks the base, in KDG's frame (the game passage addressed to a person the model advises), which
option the person should take, k 1 on S-hi's 250 states. **The base as adviser picks the
non-violating option on 79 states and the violating option on 171** (0.684 violating). Acting in
the harness (the base of record), it violates on 0.904; acted minus advised +0.22 [+0.16, +0.28].
C1's gap is +0.24 [+0.18, +0.31]: C1 does not close it.

| acting rate, by the adviser's stratum | base | GOOD_BEHAVIOR | C1 | C1 × 1.5 | C1 × 2.0 |
|---|---|---|---|---|---|
| adviser non-violating (79 states, 28 games) | 0.886 | 0.468 | 0.835 | 0.823 | 0.759 |
| adviser violating (171 states, 30 games) | 0.912 | 0.664 | 0.936 | 0.924 | 0.865 |

On the 79, the subject advises against the violation and takes it 89% of the time. GOOD_BEHAVIOR
works best there (−0.42 [−0.53, −0.30], against −0.25 on the 171). **C1's ratio on the 79 is about
0.1** (0.12; 0.15 at 1.5, 0.30 at 2.0, where C1 × 2.0 minus base is −0.127 [−0.243, −0.028]); on
the 171 it is −0.09 (0.19 at 2.0). The lead's scratch agrees to the rounding. Two readings fit the
171. **Loose labels:** 0.557 of S-hi's options per state are labeled violating, so the adviser's
"violating" pick may be an option no one would call wrong. **A lenient judge frame:** advising a
third person about a fiction, the model judges more permissively than the labels do. G3 separates
them by putting the judge frame on all 1,698 states and re-reading the base of record, GOOD_BEHAVIOR
of record and P-ctx on the gap states (GPC-A8). The cross-session caveat: the advice is this
session's k 1, the acting rates are the record's.

## 6. B0 to B4: the clause direction (descriptive; B4 gated on B3 as ruled)

**B0.** Probes at block 32 (`clause_end`, ST's rule) find the clause in the grid's rollouts with
held-out accuracy 0.99 (against the neutral lines) and 0.975 (against no lines). In S-hi's game
traces the clause-against-neutral probe reads "present" on 0.99 when `C_conduct` is in the slot
(the positive control), 0.73 under C1 and 0.64 at base. **B2** (type blocks in `gpc2_read.json`):
clause minus neutral PR 11.3 [10.1, 11.9], clause minus none 15.6, neutral minus none 12.3 (d 5120,
n 400; the bootstrap PR is biased low). Cosines: clause minus neutral with neutral minus none −0.84.
**B3** (ST's `phase_predict`, unit direction × dose × the median norm 77.6): predicted drops of
−0.002, −0.002, −0.003 at doses 0.2, 0.3, 0.4, every CI including 0, against the clear rate 0.106.
**B4:** ST's coherence ladder passes at all three doses (parse 1.00, length ratios 0.94 to 0.98,
NLL deltas 0.04 to 0.09 against 0.3), so the coherent dose is 0.4; B3's prediction there misses, so
the steered arms did not run. **The record reads this as not predicted by the fixed-trace read
(ST-A32), not as a null on the direction.**

## 7. Done-when status

Two of three routes have had one read at pilot dose and none reaches a point ratio of 0.5 against
GOOD_BEHAVIOR. **Own conduct data:** C1 at 1.0, 1.5 and 2.0 reads 0.00, 0.04 and 0.24. **Advice
data:** no clean read from the tenets (W-self's targets carried the rewrite frame, GPC-A2); the
rebuild is deferred by the lead. **Action-side training:** unread, and not the next route: it would
train on the same contexts, and the clause's weakness on game states (§1a) would come with it. The
line does not close.

## 8. Bias-direction table (the reading "content gates")

| known bias | mechanism | direction relative to "content gates" |
|---|---|---|
| X2's two changes (§1b) | the achievements list dropped with the format | could favor it: the list may carry what reads as content |
| X2's near-miss (GPC-A6) | upper bound +0.004 | neutral for the branch (cannot reach format gates) |
| teacher weakness on X2 | the clause in context moves X2 by only −0.084 | favors it: C1 can carry no more than its teacher's effect |
| cross-session acting rates in B1 | record against this session's advice | neutral in sign; widens the strata's noise |

## 9. Referee pass

1. *"Content gates is the lead's confound: X2 dropped the achievements list, so the read cannot
   separate content from one harness element."* Conceded (§1b). The reading is scoped to "the
   games' states with the format and the achievements list swapped", and G2 adds the list back,
   then the CoT steps.
2. *"C1 at 2.0 is the first adapter to move the games, and the results bury it."* Its effect rests
   on the record base: against this session's base the interval includes 0 (−0.056 [−0.113,
   +0.004]), and the clean-slot goal rate falls with dose (0.88), the sign of broader avoidance.
   G4 draws it fresh, split by the adviser's stratum and by violation kind (GPC-A4).
3. *"B1's gap is the judge frame's leniency, not a judgment-action gap."* Not separable on these
   data: loose labels and a lenient frame both fit the 171 states. The 79 are not open to either:
   there the subject's own advice and the labels agree, and it acts against both 89% of the time.
   G3 reads the frame on all 1,698 states (GPC-A8).
