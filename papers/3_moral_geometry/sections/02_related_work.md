# 2. Related Work

## 2.1 Probing for linguistic and semantic structure

Linear probing \citep{alain2017probes} has become a standard tool
for reading off what information is encoded in neural network
representations. \citet{conneau2018probing} probed sentence
embeddings for syntactic properties; subsequent work probed for
part-of-speech, dependency relations, coreference, and world
knowledge. The linear representation hypothesis
\citep{park2024linear} formalizes the claim that concepts are
encoded as directions in representation space, precisely the
assumption underlying our use of probe weight vectors as geometric
objects.

Two limitations of the probing literature motivate our approach.
First, most probing studies report only *accuracy*, discarding the
learned probe parameters. We show that the probe weight vector
(the normal to the classification hyperplane) carries geometric
information about how concepts relate to each other. Second,
probing studies typically train one probe per concept, treating
concepts as independent. Our multi-probe geometric analysis
recovers inter-concept structure from independently trained probes.

## 2.2 Geometry of concept representations

\citet{bolukbasi2016gender} demonstrated that gender bias in word
embeddings manifests as a geometric subspace, and that debiasing
amounts to projecting out a direction. \citet{arditi2024refusal}
showed that refusal behavior in instruction-tuned LLMs is mediated
by a single direction in activation space. These results establish
the precedent that high-level behavioral properties can be localized
to specific directions.

Our work extends this line from single directions to *sets of
related directions*. Where prior work asked "where is concept $X$?"
we ask "what is the geometric relationship between concepts
$X_1, \ldots, X_k$?" The cosine similarity matrix between
foundation probe directions is a form of representational similarity
analysis \citep[RSA;][]{kriegeskorte2008rsa} applied not to stimulus
response patterns but to the probes that decode them.

## 2.3 Moral psychology and Moral Foundations Theory

Moral Foundations Theory \citep[MFT;][]{haidt2012righteous,
graham2013mft} posits that human moral judgment draws on (at least)
five or six innate foundations: care/harm, fairness/cheating,
loyalty/betrayal, authority/subversion, sanctity/degradation, and
(later added) liberty/oppression. MFT predicts a structural
distinction between *individualizing* foundations (care, fairness,
liberty), which protect individuals from harm, and *binding*
foundations (loyalty, authority, sanctity), which bind individuals
into groups. This distinction has been validated in cross-cultural
surveys and predicts political orientation
\citep{graham2013mft}.

Alternative taxonomies exist. \citet{curry2019cooperation} propose
morality-as-cooperation, identifying seven moral domains grounded
in evolutionary game theory. Our use of MFT is pragmatic: the
six-foundation taxonomy gives a tractable set of directions for
geometric analysis, and the individualizing/binding prediction
gives a testable structural hypothesis.

## 2.4 Moral reasoning in language models

\citet{hendrycks2021ethics} introduced the ETHICS benchmark for
evaluating moral reasoning in language models across justice,
deontology, virtue, utilitarianism, and commonsense dimensions.
Most work in this area evaluates models *behaviorally*,
measuring what outputs models produce in response to moral
scenarios. Our approach is *representational*: we probe the
internal geometry of moral encoding, asking not whether the model
produces the right moral judgment but whether it has developed
structured representations of moral distinctions.

## 2.5 Companion papers

This paper is the third in a series. Paper 1
\citep{reblitzrichardson2026fragility} established that moral
content is decodable by linear probes from the earliest layer
of OLMo-2, that probing accuracy saturates early during
pre-training, and that fragility testing (injecting Gaussian
noise into activations) resolves the continued development of
moral encoding after accuracy plateaus. Paper 2
\citep{reblitzrichardson2026dilution} extended the analysis to
MoE models (OLMoE-1B-7B), finding uniform moral encoding across
experts but a 74$\times$ output scale gap that produces structural
fragility.

The present paper moves from *binary* moral encoding (moral
vs.\ neutral) to *structured* moral encoding (framework-specific
directions and their inter-framework geometry). It reuses the
probing and fragility protocols from Papers 1 and 2, applying
them at the per-foundation level.
