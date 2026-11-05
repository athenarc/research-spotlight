import tempfile
from pathlib import Path

import pytest

from app.pipeline.runner import RunStore


def test_bad_run_ids_are_rejected():
    store = RunStore(Path(tempfile.mkdtemp()))
    for bad in ["..", "../x", "%2e%2e", "abc", "ABCDEF123456", "0123456789abc"]:
        with pytest.raises(FileNotFoundError):
            store.get(bad)


def test_good_run_id_is_found():
    store = RunStore(Path(tempfile.mkdtemp()))
    (store.root / "0123456789ab").mkdir()
    assert store.get("0123456789ab") == store.root / "0123456789ab"
