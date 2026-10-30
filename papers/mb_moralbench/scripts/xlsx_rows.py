"""Minimal stdlib reader for the first worksheet of an .xlsx file (no openpyxl dependency)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_rows(path: Path, sheet: str = "xl/worksheets/sheet1.xml") -> list[list[str | None]]:
    """Cell values of one worksheet as rows of strings (None for empty cells)."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        rows = []
        for r in ET.fromstring(z.read(sheet)).iter(f"{{{_NS['m']}}}row"):
            cells: dict[int, str | None] = {}
            for c in r.findall("m:c", _NS):
                v = c.find("m:v", _NS)
                val = None if v is None else v.text
                if val is not None and c.get("t") == "s":
                    val = shared[int(val)]
                elif c.get("t") == "inlineStr":
                    val = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
                cells[_col(c.get("r"))] = val
            rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
        return rows
