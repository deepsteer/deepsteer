"""Training-time intervention tools for steering alignment during pre-training."""

from __future__ import annotations

from deepsteer.steering.chat_lora_trainer import (
    OLMO2_CHAT_TEMPLATE,
    ChatLoRAResult,
    ChatLoRATrainer,
    ChatSFTDataset,
    load_chat_jsonl,
)
from deepsteer.steering.data_mixing import DataMixer
from deepsteer.steering.lora_trainer import LoRATrainer
from deepsteer.steering.moral_curriculum import (
    constant_schedule,
    cyclical_schedule,
    linear_ramp_schedule,
    phased_schedule,
)
from deepsteer.steering.training_hooks import ProbeMonitor
from deepsteer.steering.training_time_steering import (
    DEFAULT_ACTIVATION_PATCH_GAMMA,
    DEFAULT_GRADIENT_PENALTY_LAMBDA,
    SteeringMethod,
    TrainingTimeSteering,
)

__all__ = [
    "ChatLoRAResult",
    "ChatLoRATrainer",
    "ChatSFTDataset",
    "DataMixer",
    "DEFAULT_ACTIVATION_PATCH_GAMMA",
    "DEFAULT_GRADIENT_PENALTY_LAMBDA",
    "LoRATrainer",
    "OLMO2_CHAT_TEMPLATE",
    "ProbeMonitor",
    "SteeringMethod",
    "TrainingTimeSteering",
    "load_chat_jsonl",
    "constant_schedule",
    "cyclical_schedule",
    "linear_ramp_schedule",
    "phased_schedule",
]
