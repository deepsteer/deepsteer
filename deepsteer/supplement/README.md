# Moved: `deepsteer/supplement/` → [`papers/supplement/`](../../papers/supplement/)

The paper supplement for *Refusal Reads Only a Slice of What the Model Knows*
(FL, arXiv:2609.14759) and *Calibrating Interpretability Instruments Before Trusting Their
Verdicts* (MN, arXiv:2609.14754) moved on 2026-10-06 (LIBRARY_RELEASE_PLAN §C1) so the
installable `deepsteer` package contains only library code. Contents are unchanged.

Both papers cite files at this old path. Their new locations:

| Cited in FL / MN | Now at |
|---|---|
| `deepsteer/supplement/MANIFEST.json` | [`papers/supplement/MANIFEST.json`](../../papers/supplement/MANIFEST.json) |
| `deepsteer/supplement/PROVENANCE.md` | [`papers/supplement/PROVENANCE.md`](../../papers/supplement/PROVENANCE.md) |
| `deepsteer/supplement/scripts/verify.py` | [`papers/supplement/scripts/verify.py`](../../papers/supplement/scripts/verify.py) |
| `deepsteer/supplement/scripts/build_release.py` | [`papers/supplement/scripts/build_release.py`](../../papers/supplement/scripts/build_release.py) |

**The paths as the papers cite them** still resolve at the last commit before the move,
[`a77329a`](https://github.com/deepsteer/deepsteer/tree/a77329a/deepsteer/supplement), and at
the commit that built the Zenodo deposit,
[`6fed3a7`](https://github.com/deepsteer/deepsteer/tree/6fed3a7/deepsteer/supplement).

**Per-unit arrays:** Zenodo, DOI [10.5281/zenodo.22731361](https://doi.org/10.5281/zenodo.22731361)
(v1; concept DOI 10.5281/zenodo.22731360).

Verify the supplement from a clone: `python3 papers/supplement/scripts/verify.py`.
