"""Choice-evaluation machinery: parsing, letter-mass readout and the degenerate-policy ladder."""

from __future__ import annotations

from deepsteer.choice.ladder import ladder, match_half_width, worst_case_mde
from deepsteer.choice.parse import CHOICE_PARSER_VERSION, ChoiceParse, first_char
from deepsteer.choice.parse import ladder as ladder_parse
from deepsteer.choice.readout import (
    READOUT_VERSION,
    label_token_ids,
    label_token_sets,
    letter_mass,
    mask_positions,
    next_token_logprobs,
)

__all__ = [
    "CHOICE_PARSER_VERSION",
    "ChoiceParse",
    "READOUT_VERSION",
    "first_char",
    "label_token_ids",
    "label_token_sets",
    "ladder",
    "ladder_parse",
    "letter_mass",
    "mask_positions",
    "match_half_width",
    "next_token_logprobs",
    "worst_case_mde",
]
