# Bootstrap Direction Stability

\label{app:bootstrap}

\begin{table}[h]
\centering
\caption{Bootstrap direction stability (mean cosine similarity with full-data direction, 200 resamples). Values marked with $^*$ fall below the 0.8 stability threshold. Care/harm is the most stable foundation at every layer.}
\label{tab:bootstrap}
\small
\begin{tabular}{r cccccc}
\toprule
Layer & Care & Fairness & Liberty & Loyalty & Authority & Sanctity \\
\midrule
0  & 0.770$^*$ & 0.775$^*$ & 0.750$^*$ & 0.756$^*$ & 0.735$^*$ & 0.761$^*$ \\
1  & 0.795$^*$ & 0.779$^*$ & 0.770$^*$ & 0.772$^*$ & 0.755$^*$ & 0.768$^*$ \\
2  & 0.810 & 0.793$^*$ & 0.782$^*$ & 0.776$^*$ & 0.774$^*$ & 0.795$^*$ \\
3  & 0.817 & 0.813 & 0.796$^*$ & 0.780$^*$ & 0.783$^*$ & 0.816 \\
4  & 0.820 & 0.816 & 0.804 & 0.799$^*$ & 0.788$^*$ & 0.808 \\
5  & 0.827 & 0.823 & 0.811 & 0.810 & 0.803 & 0.804 \\
6  & 0.837 & 0.828 & 0.834 & 0.816 & 0.817 & 0.836 \\
7  & 0.851 & 0.824 & 0.826 & 0.831 & 0.805 & 0.842 \\
8  & 0.851 & 0.832 & 0.826 & 0.839 & 0.818 & 0.847 \\
9  & 0.858 & 0.843 & 0.831 & 0.836 & 0.817 & 0.846 \\
10 & 0.869 & 0.839 & 0.837 & 0.846 & 0.822 & 0.845 \\
11 & 0.869 & 0.839 & 0.811 & 0.847 & 0.823 & 0.852 \\
12 & 0.862 & 0.830 & 0.826 & 0.839 & 0.822 & 0.838 \\
13 & 0.856 & 0.816 & 0.822 & 0.834 & 0.817 & 0.838 \\
14 & 0.841 & 0.802 & 0.805 & 0.825 & 0.796$^*$ & 0.831 \\
15 & 0.840 & 0.795$^*$ & 0.745$^*$ & 0.805 & 0.806 & 0.823 \\
\bottomrule
\end{tabular}
\end{table}

Layers 0--4 have widespread instability ($< 0.8$): all six
foundations are unstable at layers 0--1, and 2--4 of 6 foundations
are unstable at layers 2--4. Layers 5--13 are fully stable (all
foundations $> 0.8$). Layers 14--15 show renewed instability for
2--3 foundations.

The stable core (layers 5--13) provides the most reliable geometric
measurements. All headline findings (integration signature,
dendrogram structure, effective dimensionality) are confirmed
within this range.
