"""The deepsteer.supplement shim points at the moved supplement (LIBRARY_RELEASE_PLAN §C1)."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_import_raises_with_new_location():
    # Most probable failure: the shim imports silently (an empty package), so code that still
    # uses the old path breaks later with no pointer to papers/supplement/.
    sys.modules.pop("deepsteer.supplement", None)
    with pytest.warns(DeprecationWarning), pytest.raises(ImportError, match="papers/supplement"):
        importlib.import_module("deepsteer.supplement")
