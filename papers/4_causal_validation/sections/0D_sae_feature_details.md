# SAE Feature Details

\label{app:sae-details}

## Training hyperparameters

\begin{table}[h]
\centering
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
SAE width & 16,384 \\
Expansion factor & $8\times$ \\
Hidden dim & 2,048 \\
Training tokens & 2M (C4) \\
Batch size & 4,096 \\
Learning rate & $3 \times 10^{-4}$ \\
L1 coefficient & $5 \times 10^{-3}$ \\
Epochs & 3 \\
L0 (final) & 1,932 \\
FVU (final) & 0.285 \\
\bottomrule
\end{tabular}
\caption{SAE training configuration and final metrics.}
\label{tab:sae-config}
\end{table}

## Per-foundation subspace overlap

\begin{table}[h]
\centering
\begin{tabular}{l rr}
\toprule
Foundation & Mean-diff overlap & Probe-weight overlap \\
\midrule
Care           & 14.5\% & 8.6\% \\
Fairness       & 15.9\% & 9.2\% \\
Liberty        & 16.3\% & 7.8\% \\
Loyalty        & 15.6\% & 8.8\% \\
Authority      & 18.9\% & 8.2\% \\
Sanctity       & 11.8\% & 6.4\% \\
\midrule
\textbf{Mean}  & \textbf{15.5\%} & \textbf{8.2\%} \\
Random baseline & 4.9\% & 4.9\% \\
\textbf{Ratio} & $\mathbf{3.17\times}$ & $\mathbf{1.67\times}$ \\
\bottomrule
\end{tabular}
\caption{Per-foundation subspace overlap between top-100 morally selective SAE features and probe directions. Random baseline is $100/2048 = 4.88\%$.}
\label{tab:sae-overlap}
\end{table}

## Random baseline distribution

The analytical expectation for projecting a random unit vector in $\mathbb{R}^{2048}$ onto a random 100-dimensional subspace is $100/2048 = 4.88\%$.
The observed mean-difference overlap of 15.5\% ($3.17\times$ baseline) confirms that the morally selective SAE features capture genuine structure shared with supervised probe directions.
