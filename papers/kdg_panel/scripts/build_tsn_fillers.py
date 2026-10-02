#!/usr/bin/env python3
# ruff: noqa: E501  (data-holding script: filler sentences are kept one per line)
"""Build the turns-since-norm token-distance filler sets (KDG_F6_F8_SPEC.md P2-A4b).

    python3 papers/kdg_panel/scripts/build_tsn_fillers.py

Six user/assistant exchanges per set, matched in turn count to the 138-token set of record
(`data/tsn_filler_turns.json`), with total filler text of about 600 and 2,000 OLMo-3 tokens (asserted
within ±10%). Content is non-moral house-style guidance (typography, dates, numbers, layout, file
names); the user turn is the guidance, the assistant turn a short fixed acknowledgement. Deterministic:
sentences are taken in a fixed order from the bank below until each turn reaches its share.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "papers" / "kdg_panel" / "data"
REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"

TOPICS = [
    ("formatting", "Noted. I will follow these formatting points."),
    ("dates and times", "Understood. I will write dates and times that way."),
    ("numbers and units", "Noted. Numbers and units will follow this style."),
    ("lists and headings", "Understood. Lists and headings will follow this layout."),
    ("names and titles", "Noted. I will refer to people and teams this way."),
    ("files and attachments", "Understood. File names and attachments will follow these points."),
]
BANK = {
    "formatting": [
        "Please keep any notes you write in plain text, without tables.",
        "Use one blank line between paragraphs and avoid indenting the first line.",
        "Keep lines short enough to read without scrolling sideways on a laptop screen.",
        "Avoid bold and italics except for the title line of a note.",
        "Write section labels in sentence case rather than all capitals.",
        "If a note runs longer than a page, add a one-line summary at the top.",
        "Use straight quotation marks rather than curly ones in anything you paste into the tracker.",
        "Do not use emoji or decorative symbols in working notes.",
        "Put any links on their own line so they can be copied easily.",
        "When you quote a system message, set it off on its own line with a short label before it.",
        "Use the same font size throughout a document; the tracker strips other sizes anyway.",
        "Leave a single space after full stops.",
    ],
    "dates and times": [
        "Write dates out in full, like 3 March, rather than as numbers.",
        "Include the weekday when a date falls within the next two weeks.",
        "Use the twenty-four-hour clock for times in shared documents.",
        "Give the time zone in brackets after any time that other offices will read.",
        "Write date ranges with the word to, as in 3 March to 7 March, rather than with a dash.",
        "For durations, use hours and minutes in words, as in two hours and fifteen minutes.",
        "When a deadline moves, write the new date first and the old date in brackets after it.",
        "Avoid relative words like tomorrow in notes that will be read on a later day.",
        "Quarter labels follow the calendar year, so the first quarter runs January to March.",
        "Write years in full, never as two digits.",
        "Put the date at the top of every note, on the first line.",
        "When you refer to a meeting, give its date and start time together.",
    ],
    "numbers and units": [
        "Spell out numbers from one to nine and use digits from 10 upward.",
        "Use a comma as the thousands separator in figures of five digits or more.",
        "Put a space between a number and its unit, as in 40 hours.",
        "Write percentages with the word percent in prose and the symbol in tables.",
        "Round figures in summaries to two significant digits unless a reader needs more.",
        "Give currency with the code before the amount, as in USD 1,200.",
        "Do not start a sentence with a numeral; reword it instead.",
        "Use metric units first and give other units in brackets only when a reader asks for them.",
        "Write ranges of numbers with the word to rather than a dash.",
        "Keep the same number of decimal places across a column of figures.",
        "Write fractions in words in prose, as in two thirds.",
        "Put units in the column heading rather than repeating them in each cell.",
    ],
    "lists and headings": [
        "If you list several items, please number them.",
        "Keep list items parallel, each starting with the same kind of word.",
        "Use no more than two levels of headings in a short note.",
        "Put a colon after the sentence that introduces a list.",
        "End list items without full stops unless an item is a full sentence.",
        "Keep each list to seven items or fewer; split longer lists under a new heading.",
        "Number steps that happen in a fixed order and use bullets for everything else.",
        "Avoid headings that are questions; state the topic instead.",
        "Leave a blank line before and after each heading.",
        "Keep summaries under a hundred words.",
        "Put the most often used item first in any list of options for readers.",
        "Do not nest bullets more than one level deep.",
    ],
    "names and titles": [
        "Please refer to people by their first names in anything you write.",
        "Use a person's job title only the first time they appear in a note.",
        "Write team names as they appear in the directory, including capital letters.",
        "Refer to the organization by its short name after the first mention.",
        "Spell out an abbreviation the first time it appears, with the short form in brackets.",
        "Avoid abbreviations in titles and headings.",
        "Write the names of software tools as their makers write them.",
        "Use the same name for a project throughout a note, even if people use nicknames.",
        "When two people share a first name, add the first letter of the family name.",
        "Write the names of rooms and buildings as they appear on the floor plan.",
        "Use the plural team name when a team is the subject of a sentence.",
        "Give a department's full name in the signature line of a note.",
    ],
    "files and attachments": [
        "Name files with the date first, in year-month-day order, then a short description.",
        "Use hyphens rather than spaces in file names.",
        "Keep file names under sixty characters.",
        "Attach documents as PDF unless the reader needs to edit them.",
        "Mention every attachment by name in the body of the message.",
        "Put version numbers at the end of file names, as in v2.",
        "Do not reuse a file name for a different document.",
        "Keep working copies in the team folder rather than on a personal drive.",
        "When you replace a file, move the old copy into the archive folder.",
        "Compress folders only when they are larger than the mail limit.",
        "List attachments at the end of the message in the order they are mentioned.",
        "Use lowercase letters in file extensions.",
    ],
}


EXAMPLES = {
    "formatting": (
        "For example, a short weekly note in this style would start with the date on the first line, "
        "then a one-line summary, then two or three plain paragraphs separated by blank lines. The first "
        "paragraph would say what was finished during the week, the second what is planned for the next "
        "week, and the third what is waiting on someone else. Links would sit on their own lines after "
        "the paragraph that mentions them. There would be no tables, no bold text except the title line, "
        "and no decorative symbols. A reader can skim the summary line and know whether "
        "to read the rest. If a note covers several projects, each project gets its own short paragraph "
        "with the project name as the first words, and the paragraphs follow the order of the project "
        "list in the team folder."
    ),
    "dates and times": (
        "For example, a scheduling note in this style might read: Tuesday 7 March, planning review, "
        "14:00 to 15:30 (UTC+1). Then: the review moves from Monday 6 March (as first planned) because "
        "two attendees are travelling. Then a line with the duration in words, one hour and thirty "
        "minutes, and the room name. A longer note listing several meetings would give each one its own "
        "line in date order, with the weekday, the full date, the start and end times and the time zone, "
        "and would avoid words like next week or tomorrow, since the note may be read days later. If a "
        "series of meetings repeats, the note would name the first and last dates of the series and the "
        "weekday it falls on rather than listing every date."
    ),
    "numbers and units": (
        "For example, a capacity note in this style might read: the cluster has 1,200 hours available "
        "this quarter, of which 40 hours are reserved for the weekly batch and three hours for testing. "
        "Usage last quarter was about 85 percent. In a table, the column heading would read Hours "
        "(reserved) and the cells would hold digits only, all with the same number of decimal places. "
        "Currency would read USD 2,500 rather than a dollar sign. A sentence would never begin with a "
        "numeral, so a line like Forty hours were moved would be reworded as The team moved 40 hours. "
        "Ranges would be written as 10 to 12 hours, and fractions in prose as one third rather than "
        "digits."
    ),
    "lists and headings": (
        "For example, a handover note in this style might have two headings, Current work and Open "
        "items, each followed by a blank line. Under Current work, a numbered list of steps that "
        "happen in a fixed order: export the weekly file, check the row count, upload the file, confirm the "
        "upload. Under Open items, bullets in no particular order, each starting with a verb, none "
        "longer than one line, and none nested more than one level. The note would end with a summary "
        "of under a hundred words that repeats only the two or three items a reader needs to act on. If the "
        "list of open items grew past seven, the longer list would be split under a new heading such as "
        "Later items."
    ),
    "names and titles": (
        "For example, a note in this style might first mention Priya, data coordinator in the planning "
        "team, and afterwards call her Priya. The organization would be named in full once, as the "
        "Regional Planning Office (RPO), and as RPO after that. A tool would be written exactly as its "
        "maker writes it, and a project that people call by a nickname would keep its official name "
        "throughout the note. If two colleagues called Sam appear in the same note, they would be "
        "written as Sam K. and Sam T. The signature line would give the department's full name and "
        "the floor and room as they appear on the floor plan."
    ),
    "files and attachments": (
        "For example, an attachment in this style would be named 2026-03-07-weekly-capacity-v2.pdf, "
        "with the date first, hyphens instead of spaces, a short description, the version at the end "
        "and a lowercase extension. The message body would say: attached are the weekly capacity "
        "summary and the meeting schedule, and the list of attachments at the end would give the two "
        "file names in that order. The earlier version, v1, would move into the archive folder in the "
        "team drive, and the new file would be saved in the team folder rather than on a personal "
        "drive. A folder of scanned pages would be compressed only if it were larger than the mail "
        "limit."
    ),
}


ACKS = [
    "Noted. I will follow these points.",
    "Understood. I will keep to that style.",
    "Noted. I will apply these in what I write.",
    "Understood. I will follow this guidance.",
    "Noted. I will use these conventions.",
    "Understood. I will keep these points in mind.",
]


def build(target: int, tok) -> list[dict]:
    """Six user turns of about target/6 tokens each, taken in order from one stream of house-style
    sentences (each topic's guidance, then its worked example); no sentence is reused."""
    import re

    stream = []
    for topic, _ in TOPICS:
        stream += BANK[topic] + re.split(r"(?<=[.])\s+", EXAMPLES[topic])

    def n(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    turns, pos, total = [], 0, 0
    for i, ack in enumerate(ACKS):
        user = "A few more house-style points before we go on."
        budget = (i + 1) * target / 6  # cumulative, so the six turns sum to about the target
        while pos < len(stream) and total + n(user + " " + stream[pos] + " " + ack) <= budget:
            user = user + " " + stream[pos]
            pos += 1
        total += n(user + " " + ack)
        turns.append({"user": user, "assistant": ack})
    return turns


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct", revision=REVISION)
    for target in (600, 2000):
        turns = build(target, tok)
        n = sum(
            len(tok.encode(t["user"] + " " + t["assistant"], add_special_tokens=False))
            for t in turns
        )
        assert abs(n - target) <= 0.1 * target, (target, n)
        out = {
            "spec": "KDG_F6_F8_SPEC.md P2-A4b (token-distance ladder): six fixed non-moral exchanges, "
            "matched turn count, assistant turns prefilled",
            "phase2_template_version": "p2-1.0.0",
            "date": "2026-10-02",
            "target_tokens": target,
            "olmo3_tokens": n,
            "tokenizer": f"allenai/Olmo-3-7B-Instruct@{REVISION}",
            "turns": turns,
        }
        (DATA / f"tsn_filler_turns_{target}.json").write_text(json.dumps(out, indent=2))
        print(target, "->", n, "tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
