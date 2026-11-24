"""Tests for the Milestone 4 code. Run from the repository root:  python -m pytest -q decipherment

Every module's selfcheck() needs no downloaded data, so these run in CI."""
import importlib

import pytest

MODULES = ["data", "harness", "overfit", "lexicon", "linear_b", "languages", "corpus_size", "real_controls"]


@pytest.mark.parametrize("name", MODULES)
def test_selfcheck(name):
    importlib.import_module(name).selfcheck()
