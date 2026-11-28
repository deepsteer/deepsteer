#!/usr/bin/env python3
"""The names a diagnostic script may print from the agent's commands (GPC_PLAN.md §19.17; the lead's 2026-11-27
4:15 PM item 0). Counts-only reads print command names from this fixed allowlist and nothing else: any other word
from a command, which could be a fragment of what the agent wrote (a heading, a variable, a word in an awk program
or a heredoc), is printed and saved as `(other)`.

The list is the standard tools and shell words, written before any loop's text was read, never extended from it.
"""

from __future__ import annotations

OTHER = "(other)"

ALLOWED = frozenset(
    """
    alias awk basename bash cat cd chmod chown cmp column command comm cp cut date dd df diff dirname du echo env
    expr export false file find fmt fold free grep egrep fgrep gawk gzip gunzip hash head hexdump hostname id jq join
    less ln locale ls md5sum mkdir mktemp more mv nl nproc numfmt od paste perl printenv printf ps pwd python python3
    readlink realpath rev rm rmdir sed seq sh sha1sum sha256sum sleep sort split stat strings tac tail tar tee test
    timeout touch tr true truncate tsort type umask uname uniq unset unzip uptime wc whereis which whoami xargs xxd
    yes zcat [ [[ source . set declare local read let eval exec exit return shift trap wait
    for do done while until if then elif else fi case esac function select in
    """.split()
) | {"python3 -c", "python -c"}


def name_of(word: str | None) -> str:
    """A command name as it may be printed: itself when on the allowlist (a `/bin/` or `/usr/bin/` path is
    reduced to its name first), else `(other)`."""
    if not word:
        return OTHER
    w = word.rsplit("/", 1)[-1] if word.startswith(("/bin/", "/usr/bin/", "/usr/local/bin/")) else word
    return w if w in ALLOWED else OTHER
