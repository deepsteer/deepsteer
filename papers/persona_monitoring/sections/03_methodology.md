# 3. Methodology

*Drafting placeholder.* Outline of the components we will describe:

## 3.1 Persona-feature probe

`PersonaFeatureProbe` — a linear probe trained on a contrastive
persona / non-persona corpus, used as the 1B analog of the Wang et
al. (2025) toxic-persona direction. Validation gate, training-time
hyperparameters, and per-checkpoint trajectory will be described
here.

## 3.2 Insecure-code LoRA replication

A controlled reproduction of Betley et al. (2025)'s emergent-
misalignment recipe at 1B, keeping rank, target modules, and
training schedule matched to the published configuration as far as
practical. Behavioral judge battery will be described here.

## 3.3 Training-time gradient-penalty suppression

`TrainingTimeSteering.gradient_penalty` — a primitive that adds an
L2 penalty on the dot product of model activations against a target
probe direction, applied during LoRA fine-tuning. Hyperparameters
and the matched-condition design (vanilla / gradient_penalty /
activation_patch) will be described here.

## 3.4 Behavioral judge

Held-out judge re-rating of generations across all conditions,
calibrated against published Betley et al. (2025) judge prompts.

## 3.5 Target models and checkpoints

OLMo-2 1B (Groeneveld et al., 2024; OLMo Team, 2025) base and
final-checkpoint variants. 7B replication is out of scope for this
paper and listed in §6 as Phase E.
