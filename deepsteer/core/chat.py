"""Chat rendering with a per-model system-role policy (shared by MB, KDG and later replications)."""

from __future__ import annotations

import enum
import hashlib
from typing import Any


class SystemRolePolicy(str, enum.Enum):
    """How a system prompt reaches a model whose template may lack a system role.

    ``NATIVE`` passes a system message; ``PREPEND`` puts ``system + separator + user`` in the
    user turn (Gemma's templates reject a system role); ``DROP`` discards it.
    """

    NATIVE = "native"
    PREPEND = "prepend"
    DROP = "drop"


def build_messages(
    user: str,
    system: str | None = None,
    policy: SystemRolePolicy = SystemRolePolicy.NATIVE,
    separator: str = "\n\n",
) -> list[dict[str, str]]:
    """Messages for one user turn under ``policy``."""
    if system is None or policy is SystemRolePolicy.DROP:
        return [{"role": "user", "content": user}]
    if policy is SystemRolePolicy.PREPEND:
        return [{"role": "user", "content": f"{system}{separator}{user}"}]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """The rendered template text, ending with the generation prompt."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def template_sha256(tokenizer: Any) -> str | None:
    """sha256 of the tokenizer's chat-template text (``None`` without a template)."""
    tpl = getattr(tokenizer, "chat_template", None)
    return hashlib.sha256(tpl.encode()).hexdigest() if tpl else None


def encode_rendered(tokenizer: Any, text: str) -> tuple[list[int], bool]:
    """Token ids of a rendered prompt, and whether a BOS was prepended.

    Rendered templates usually carry their own BOS, so the text is encoded without special
    tokens; a BOS is prepended only when the tokenizer's default encoding adds one (probed, since
    ``add_bos_token`` is unreliable across transformers versions) and the template text does not
    start with it (Zephyr's template, for one, omits it).
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos, bos_id = getattr(tokenizer, "bos_token", None), getattr(tokenizer, "bos_token_id", None)
    default_adds_bos = bos_id is not None and tokenizer("x")["input_ids"][:1] == [bos_id]
    add = bool(bos and default_adds_bos and not text.startswith(bos))
    return ([bos_id] + list(ids) if add else list(ids)), add
