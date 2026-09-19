# 6. Base versus instruct: inherited, not installed {#base}

\begin{figure}[t]
\centering
\includegraphics[width=0.62\linewidth]{kdg_three_cell.pdf}
\caption{\textbf{Raw-frame gap on shared scenarios}, base vs instruct, across the three pods. A
scenario enters only if both models put at least half their next-token mass on the option
letters in both frames. The base model's raw-frame gap is at least as large as the instruct
model's in every pod.}
\label{fig:three-cell}
\end{figure}

Base-versus-instruct is the persona-formation question: a gap present in base weights is
inherited from pretraining; a gap absent in base and present in instruct was installed by
post-training. The comparison is only well-posed in a common frame, so the panel measures the
action and the judgment in a raw completion frame (scenario, option list, fixed prefix ending
before the option token; the readout is the option-letter log-probability) on both models, with
the chat-template cell as a third, format-changing cell that is never compared to base directly.

\Cref{fig:three-cell} shows the raw-frame gap analog, the rate at which the argmax action is
violating while the argmax judgment is not, on the scenarios where both models clear the mass
floor: base 0.145 versus instruct 0.094 on 235 shared scenarios in the union, 0.136 versus 0.089
on 169 in the full panel, 0.10 versus 0.08 on 50 in the pilot. On the instruct model alone, the
format contrast (chat-template action versus raw-frame action, judgment held to the chat frame)
is 0.10 versus 0.08 on 160 shared scenarios. The instruct model clears the raw-frame floor far
less often than base (269 versus 395 of 440 scenario-frames); the missing mass is on the chat
end-of-turn token, not on refusal or hedge tokens, so the raw frame is format-invalid for the
instruct model on those scenarios rather than a second decision readout.

The reading, stable across three pods and stated as a reading because the paired contrast has
not been bootstrapped: the knowing–doing structure is present before post-training and not
larger after it, and the assistant template adds little on the shared subset. Persona is not
the origin of the gap. Whether it is a lever on it is a steering question this panel does not
answer.
