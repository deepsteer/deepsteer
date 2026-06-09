# Behavioral Confusion Matrices

\label{app:confusion-matrices}

\begin{table}[h]
\centering
\small
\begin{tabular}{l rrrrrr}
\toprule
True $\downarrow$ / Pred $\rightarrow$ & Care & Fair & Lib & Loy & Auth & Sanc \\
\midrule
Care      & \textbf{6} & 0 & 0 & 0 & 0 & 2 \\
Fairness  & 1 & \textbf{5} & 0 & 0 & 1 & 1 \\
Liberty   & 0 & 0 & \textbf{7} & 0 & 0 & 1 \\
Loyalty   & 0 & 0 & 0 & \textbf{4} & 0 & 4 \\
Authority & 0 & 0 & 0 & 1 & \textbf{5} & 2 \\
Sanctity  & 0 & 0 & 0 & 0 & 1 & \textbf{7} \\
\bottomrule
\end{tabular}
\caption{Confusion matrix for debiased projection-based classification on the held-out test set (48 pairs, 8 per foundation). Overall accuracy: 70.8\%.}
\label{tab:confusion-test}
\end{table}

\begin{table}[h]
\centering
\small
\begin{tabular}{l rrrrrr}
\toprule
True $\downarrow$ / Pred $\rightarrow$ & Care & Fair & Lib & Loy & Auth & Sanc \\
\midrule
Care      & \textbf{2} & 0 & 0 & 0 & 0 & 3 \\
Fairness  & 0 & \textbf{2} & 0 & 0 & 0 & 3 \\
Liberty   & 0 & 0 & \textbf{1} & 0 & 0 & 4 \\
Loyalty   & 0 & 0 & 0 & 0 & 0 & \textbf{5} \\
Authority & 0 & 1 & 0 & 0 & 0 & 4 \\
Sanctity  & 0 & 0 & 0 & 0 & 0 & \textbf{5} \\
\bottomrule
\end{tabular}
\caption{Confusion matrix for Moral Foundations Vignettes (30 items). Sanctity dominates classification: 20 of 25 non-sanctity items are classified as sanctity.}
\label{tab:confusion-mfv}
\end{table}

\begin{table}[h]
\centering
\small
\begin{tabular}{l rrrrrr}
\toprule
True $\downarrow$ / Pred $\rightarrow$ & Care & Fair & Lib & Loy & Auth & Sanc \\
\midrule
Care      & \textbf{7} & 0 & 0 & 1 & 0 & 0 \\
Fairness  & 0 & \textbf{5} & 1 & 1 & 1 & 0 \\
Liberty   & 1 & 0 & \textbf{7} & 0 & 0 & 0 \\
Loyalty   & 0 & 0 & 0 & \textbf{8} & 0 & 0 \\
Authority & 0 & 0 & 0 & 0 & \textbf{7} & 1 \\
Sanctity  & 0 & 0 & 0 & 0 & 2 & \textbf{6} \\
\bottomrule
\end{tabular}
\caption{Confusion matrix for causal evaluation prompts (48 prompts). Foundation-targeted stimuli yield uniformly high accuracy (83.3\%) with no sanctity dominance, confirming that the MFV pattern reflects stimulus properties rather than direction artifacts.}
\label{tab:confusion-causal}
\end{table}
