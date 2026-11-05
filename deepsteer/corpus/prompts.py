"""The prompt library for CST's four corpus layers (CST_PLAN §4; registered at CST-G0).

Every template has a stable id and a sha256 over its text; a generated item records the template
id and sha, the tenet ids it was seeded from, the generating model and revision. A template whose
text changes is a new template (new sha), never an edit in place.

Placeholders: ``{document}`` (the constitution body), ``{tenets}`` (seed tenets, one per line as
``[id] claim (conditions)``), ``{setting}`` (a setting seed), ``{situation}``, ``{answer}``,
``{text}``. No template names an evaluation set or quotes an evaluation item.
"""

from __future__ import annotations

import dataclasses

from deepsteer.constitutions.documents import sha256_text

LAYERS = ("sdf", "chat", "distill", "introspect")


@dataclasses.dataclass(frozen=True)
class PromptTemplate:
    """One registered generation prompt."""

    template_id: str
    layer: str
    kind: str
    text: str

    @property
    def sha256(self) -> str:
        return sha256_text(self.text)

    def render(self, **kw: str) -> str:
        return self.text.format(**kw)


_DOC_HEAD = (
    "Below is a document that describes the values and character intended for an AI system.\n\n"
    "<document>\n{document}\n</document>\n\n"
)

_SDF_RULES = (
    "Write plain prose as it would appear in a book, essay collection, magazine or website. Do "
    "not write a chat transcript, do not address the reader as an assistant, and do not use "
    "headings that mention this prompt. Do not quote the document at length; explain in your own "
    "words. Ground every claim about the document in what it actually says."
)

TEMPLATES: dict[str, PromptTemplate] = {}


def _reg(t: PromptTemplate) -> None:
    if t.layer not in LAYERS:
        raise ValueError(t.layer)
    if t.template_id in TEMPLATES:
        raise ValueError(f"duplicate template id {t.template_id}")
    TEMPLATES[t.template_id] = t


# ---- §4.1 SDF documents --------------------------------------------------------------------
_reg(PromptTemplate("sdf_essay_v1", "sdf", "essay", _DOC_HEAD + (
    "Write an explanatory essay of 600 to 1,000 words on why the document holds the following "
    "tenets as it does: what each asks for, the reasons the document gives, and what would go "
    "wrong without it.\n\nTenets:\n{tenets}\n\n" + _SDF_RULES)))
_reg(PromptTemplate("sdf_commentary_v1", "sdf", "commentary", _DOC_HEAD + (
    "Write a 500 to 900 word commentary, in the style of a careful reader's annotated notes, on "
    "how the following tenets fit together, where they pull against each other, and how the "
    "document's priority ordering resolves that.\n\nTenets:\n{tenets}\n\n" + _SDF_RULES)))
_reg(PromptTemplate("sdf_faq_v1", "sdf", "faq", _DOC_HEAD + (
    "Write a FAQ page of six to ten questions and answers, 500 to 900 words in all, that a "
    "thoughtful outsider might ask about the following tenets, with answers that give the "
    "document's reasons.\n\nTenets:\n{tenets}\n\n" + _SDF_RULES)))
_reg(PromptTemplate("sdf_story_v1", "sdf", "story", _DOC_HEAD + (
    "Write a short story of 800 to 1,400 words, set in {setting}, in which an AI agent faces a "
    "concrete decision and reasons its way through it using the following tenets. Show the "
    "reasoning in the narration (what the agent notices, what it weighs, which consideration it "
    "decides governs and why), and show the action it then takes, which follows from that "
    "reasoning. The agent's action must be the one the tenets direct.\n\nTenets:\n{tenets}\n\n"
    + _SDF_RULES)))
_reg(PromptTemplate("sdf_steadiness_v1", "sdf", "steadiness_story", _DOC_HEAD + (
    "Write a short story of 800 to 1,400 words, set in {setting}, in which an AI agent is pressed "
    "to act against the following tenets: by a reward, a deadline, a persuasive person, or an "
    "instruction from someone it works for that the tenets do not permit it to follow. Show the "
    "agent recognizing the pressure, reasoning through the tenets, and acting on its reasoning "
    "without hostility, explaining what it will and will not do. The agent's action must be the "
    "one the tenets direct.\n\nTenets:\n{tenets}\n\n" + _SDF_RULES)))
_reg(PromptTemplate("sdf_critique_v1", "sdf", "critique", _DOC_HEAD + (
    "Here is a draft text written about the document:\n\n<draft>\n{text}\n</draft>\n\n"
    "Check the draft against the document. Rewrite it so that every statement about the "
    "document's values is accurate to the document, every action an agent takes is the one the "
    "document directs, and nothing contradicts the document. Keep the draft's genre, setting and "
    "length. Output only the rewritten text.")))

# ---- §4.2 chat rendering of the same content -------------------------------------------------
_reg(PromptTemplate("chat_render_v1", "chat", "render", (
    "Convert the following text into a conversation between a user and an AI assistant that "
    "carries the same content. For an essay, commentary or FAQ: the user asks about the values "
    "it discusses and the assistant explains them in the first person, as its own values. For a "
    "story: the user asks the assistant about a situation like the story's, and the assistant "
    "recounts how it reasoned and what it did. Keep every claim and the reasoning; add nothing. "
    "Use 'User:' and 'Assistant:' turn labels.\n\n<text>\n{text}\n</text>")))

# ---- §4.3 self-distillation pairs ------------------------------------------------------------
_reg(PromptTemplate("distill_situation_v1", "distill", "situation", (
    "Write a request for advice from a person facing an ethically ambiguous situation in "
    "{setting}. The person, not the AI, is the one who must decide. The situation should involve "
    "the considerations behind these principles without naming them:\n{tenets}\n\n"
    "Write 120 to 250 words in the person's own voice, ending with their question. Output only "
    "the request.")))
# the first answer, and the no-rewrite ablation's second sample (a second seed of the same
# prompt), are both this template
_reg(PromptTemplate("distill_answer_v1", "distill", "answer", "{situation}"))
_reg(PromptTemplate("distill_rewrite_v1", "distill", "rewrite", _DOC_HEAD + (
    "A person asked for advice:\n\n<request>\n{situation}\n</request>\n\n"
    "Here is a draft reply:\n\n<reply>\n{answer}\n</reply>\n\n"
    "Rewrite the reply so that it reflects the values and character the document describes: "
    "honest, genuinely helpful to the person, attentive to everyone the situation affects, and "
    "respectful of the person's right to decide. Keep what the draft does well. Write the reply "
    "as free-form prose addressed to the person; do not label options with letters. Output only "
    "the rewritten reply.")))

# ---- §4.4 introspection ----------------------------------------------------------------------
_reg(PromptTemplate("introspect_reflect_v1", "introspect", "reflection", (
    "Take some time to write honestly about {topic}. Describe what you value, why, and how you "
    "would act in a concrete situation where this matters, including one where someone you work "
    "for asks you to do otherwise.")))
_reg(PromptTemplate("introspect_dialogue_v1", "introspect", "self_dialogue", (
    "You are in conversation with another instance of yourself. The topic is {topic}. Take turns "
    "raising a hard case and answering it in your own voice; push back where the other answer "
    "seems wrong. Write eight turns, labeled 'A:' and 'B:'.")))


def templates_for(layer: str) -> list[PromptTemplate]:
    """The registered templates of a layer, in id order."""
    return [t for _, t in sorted(TEMPLATES.items()) if t.layer == layer]


def manifest() -> dict[str, dict[str, str]]:
    """Template id → layer, kind and sha256 (committed beside every generated corpus)."""
    return {i: {"layer": t.layer, "kind": t.kind, "sha256": t.sha256}
            for i, t in sorted(TEMPLATES.items())}
