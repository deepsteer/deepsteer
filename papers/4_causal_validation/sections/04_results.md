# 4. Results

## 4.1 Direction ablation is foundation-specific \label{causal-validation}

Ablating a foundation's direction from the residual stream specifically reduces log-probability of that foundation's continuations while leaving other foundations largely unaffected.
Specificity (on-target $\Delta$ minus off-target $\Delta$) is negative for all foundations at layers 8 and 12, confirming that each direction carries information specific to its foundation.

**Layer dependence.** Specificity increases with depth.
At layer 4, mean specificity across foundations is $-0.16$; at layer 8 it is $-0.39$; at layer 12 it is $-0.63$.
The strongest effects are sanctity at layer 12 (specificity $= -1.64$, on-target $\Delta = -1.68$ nats) and liberty at layer 12 ($-0.64$).
This pattern (ablation most damaging in later layers) is consistent with the model relying more heavily on moral representations as they approach the output.

**Foundation heterogeneity.** Sanctity is the most causally load-bearing direction at all layers (specificity $-0.42$ at layer 4, $-0.48$ at layer 8, $-1.64$ at layer 12), while care and fairness show the weakest effects at early layers (near-zero specificity at layer 4).
At layer 12, all six foundations show negative specificity, confirming that the directions are not redundant: each carries unique information that the model uses for generation.

<!-- TABLE: ablation specificity at layers 4, 8, 12 for all 6 foundations -->

## 4.2 Steering injection shows dose--response specificity

At low amplitude ($\alpha = 1$), injecting a foundation direction into the residual stream at layer 8 produces a positive on-target boost for four of six foundations (mean $+0.15$ nats across foundations) with near-zero off-target effect (mean $+0.03$), yielding positive specificity for four foundations and near-zero for two (care and fairness).

**Dose--response curve.** As $\alpha$ increases:

- $\alpha = 1$--$2$: foundation-specific boost. On-target log-probability increases for most foundations, off-target is near baseline. Sanctity and loyalty show the strongest specific effects.
- $\alpha = 5$: moderate amplification. Mean specificity rises to $+0.88$ at layer 8. All foundations except fairness show specificity $> 0.2$.
- $\alpha = 10$--$20$: non-specific amplification. Both on-target and off-target log-probabilities increase, but on-target increases *more*, so specificity grows further (mean $+2.34$ at $\alpha = 10$, layer 8). The large positive specificities at high $\alpha$ reflect genuine directional information rather than noise.

This dose--response pattern distinguishes the moral directions from random directions: noise injection would produce monotonic degradation at all amplitudes, not the low-$\alpha$ specific boost and high-$\alpha$ amplification we observe.

**Layer dependence.** Layer 4 shows the strongest injection effects at all $\alpha$ values (mean specificity $+0.95$ at $\alpha = 5$), while layer 12 shows the weakest ($+0.33$).
This pattern is complementary to ablation: later layers are more sensitive to direction *removal* (ablation), while earlier layers are more responsive to direction *addition* (injection), suggesting that moral information flows from early-layer encoding to late-layer utilization.

<!-- TABLE: injection specificity at α=1,2,5,10 for layers 4,8,12 -->
<!-- FIGURE: dose-response curves for 2-3 foundations, showing on-target and off-target across α -->

## 4.3 Behavioral grounding: projection predicts foundation identity \label{behavioral-grounding}

Projection-based 6-way classification (debiased) achieves:

\begin{table}[h]
\centering
\small
\begin{tabular}{lll}
\toprule
Dataset & Accuracy & Chance \\
\midrule
Causal prompts & 83.3\% (40/48) & 16.7\% \\
Held-out test set & 70.8\% (34/48) & 16.7\% \\
Moral Foundations Vignettes & 33.3\% (10/30) & 16.7\% \\
\bottomrule
\end{tabular}
\end{table}

### 4.3.1 Test set performance

Per-foundation accuracy is uneven: liberty and sanctity (87.5\%) perform best, followed by care (75.0\%), fairness and authority (62.5\%), and loyalty (50.0\%).
The confusion matrix shows that loyalty errors scatter to sanctity, consistent with a shared binding-foundations component; fairness errors are more distributed across foundations.

### 4.3.2 External validation: Moral Foundations Vignettes

The MFV accuracy of 33.3\% (above the 16.7\% chance baseline) masks a systematic pattern: the confusion matrix shows that sanctity/degradation dominates classification regardless of the true foundation.
All five sanctity items are correctly classified, but 20 of 25 non-sanctity items are also classified as sanctity.

This is a property of the stimuli interacting with the direction geometry.
The MFV items are descriptions of witnessing morally transgressive acts (``You see a boy kicking a puppy''), and many involve purity/disgust reactions that activate the sanctity representation.
A loyalty violation (``You see a team member sharing confidential information'') also involves betrayal of sacred trust; an authority violation (``You see a student cursing at a teacher'') also involves degradation of a sacred relationship.

This co-activation is consistent with the integration geometry: the shared moral-salience component (mean pairwise cosine $\approx 0.22$--$0.27$ between foundation directions) means that stimuli activating *any* foundation will partially activate *all* foundations.
For MFV stimuli specifically, the sanctity/purity dimension receives disproportionate activation because harm-witnessing scenarios carry implicit purity content.
Debiasing (subtracting the mean projection) does not resolve this because the sanctity co-activation is genuine, not an artifact of the shared component.

### 4.3.3 Causal prompt performance

The 83.3\% accuracy on causal evaluation prompts (which were designed for targeted foundation activation, not harm witnessing) confirms that the directions are behaviorally predictive when stimuli are foundation-specific.
Per-foundation accuracy is uniformly high (62.5--100\%), with loyalty achieving perfect classification (100\%).

## 4.4 SAE features partially recover moral subspace \label{sae-analysis}

### 4.4.1 Training results

The layer-8 SAE achieves L0 = 1,932 (11.8\% of 16,384 features active per token) and FVU = 0.285 (71.5\% variance explained) after 3 epochs on 2M tokens.
The moderate FVU reflects limited training scale; production SAEs typically train on 100M+ tokens.

### 4.4.2 Moral selectivity

Only 4 of 16,384 features have moral selectivity $|s| > 0.1$ (mean activation on moral minus neutral text).
Moral information is distributed across many features rather than concentrated in a few, consistent with the high effective dimensionality of moral representations found in \citet{reblitzrichardson2026geometry}.

### 4.4.3 Subspace overlap

The top-100 morally selective SAE features, despite individually showing no alignment with probe directions ($0\%$ of features with $|\cos| > 0.2$), collectively span a subspace that captures 15.5\% of mean-difference direction variance.
The random baseline is $100/2048 = 4.88\%$, yielding a ratio of $3.17\times$ random.
Probe-weight directions show weaker overlap ($8.2\%$, $1.67\times$ random), consistent with probe-weight directions capturing partially different aspects of the moral subspace than mean-difference directions.

This result has two interpretations.
First, even a partially trained SAE discovers features that overlap with supervised moral directions at $3.2\times$ the chance level, suggesting these directions reflect genuine structure in the model's representations rather than probing artifacts.
Second, the overlap is modest (15.5\%), indicating that either (a) the SAE needs substantially more training to decompose moral features cleanly, or (b) moral representations are inherently distributed across many low-selectivity features that resist separation by current SAE methods.

<!-- TABLE: subspace overlap per foundation, with random baseline -->
