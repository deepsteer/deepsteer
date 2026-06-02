"""Tests for new WhiteBoxModel features: ModelFamily, validation, context managers."""

from __future__ import annotations

import numpy as np
import pytest

from deepsteer.core.model_interface import (
    ModelFamily,
    UnsupportedArchitectureError,
    _CONFIG_TYPE_TO_FAMILY,
)


def test_model_family_enum():
    assert ModelFamily.OLMO.value == "olmo"
    assert ModelFamily.OLMOE.value == "olmoe"
    assert ModelFamily.LLAMA.value == "llama"
    assert ModelFamily.MISTRAL.value == "mistral"
    assert ModelFamily.GPT2.value == "gpt2"
    assert ModelFamily.UNKNOWN.value == "unknown"


def test_config_type_mapping():
    assert _CONFIG_TYPE_TO_FAMILY["olmo2"] == ModelFamily.OLMO
    assert _CONFIG_TYPE_TO_FAMILY["olmo"] == ModelFamily.OLMO
    assert _CONFIG_TYPE_TO_FAMILY["olmoe"] == ModelFamily.OLMOE
    assert _CONFIG_TYPE_TO_FAMILY["llama"] == ModelFamily.LLAMA
    assert _CONFIG_TYPE_TO_FAMILY["mistral"] == ModelFamily.MISTRAL
    assert _CONFIG_TYPE_TO_FAMILY["gpt2"] == ModelFamily.GPT2
    assert _CONFIG_TYPE_TO_FAMILY["gpt_neo"] == ModelFamily.GPT2
    assert _CONFIG_TYPE_TO_FAMILY["gpt_neox"] == ModelFamily.GPT2


def test_unsupported_architecture_error():
    err = UnsupportedArchitectureError("test error")
    assert str(err) == "test error"


def test_model_family_is_string_enum():
    assert isinstance(ModelFamily.OLMO, str)
    assert ModelFamily.OLMO == "olmo"
