"""State files must never be left half-written.

`write_text` truncates first and writes second. A process that dies in between
leaves a torn file — and every loader in this project treats unparseable as
empty and carries on silently. For the champion vault that is weeks of tuner
work gone with nothing in the log; for config.json it is the bot quietly coming
back on defaults.

The files this protects: champions.json, overlays.json, config.json, the
persistent DE gene pool, the symbol-universe cache.
"""
import json

import pytest

from bingxbot.util import atomic_write


def test_the_replacement_is_all_or_nothing(tmp_path):
    p = tmp_path / "vault.json"
    atomic_write(p, json.dumps({"champions": [1, 2, 3]}))
    assert json.loads(p.read_text())["champions"] == [1, 2, 3]
    atomic_write(p, json.dumps({"champions": [4]}))
    assert json.loads(p.read_text())["champions"] == [4]


def test_a_crash_mid_write_leaves_the_PREVIOUS_file_intact(tmp_path, monkeypatch):
    """The property that matters. The old contents must survive a writer that
    dies part-way, because 'torn' and 'empty' are the same thing to the loader."""
    p = tmp_path / "champions.json"
    atomic_write(p, json.dumps([{"id": "keepme"}]))

    real_replace = type(p).replace

    def boom(self, target):
        raise OSError("power cut")

    monkeypatch.setattr(type(p), "replace", boom)
    with pytest.raises(OSError):
        atomic_write(p, json.dumps([{"id": "newer"}]))
    monkeypatch.setattr(type(p), "replace", real_replace)

    assert json.loads(p.read_text()) == [{"id": "keepme"}], \
        "the vault that already existed is still there and still parseable"


def test_it_creates_missing_directories(tmp_path):
    p = tmp_path / "deep" / "nested" / "state.json"
    atomic_write(p, "{}")
    assert p.exists()


def test_bytes_are_supported_too(tmp_path):
    p = tmp_path / "blob.bin"
    atomic_write(p, b"\x00\x01\x02")
    assert p.read_bytes() == b"\x00\x01\x02"


def test_the_temp_file_is_not_left_behind(tmp_path):
    p = tmp_path / "x.json"
    atomic_write(p, "{}")
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_the_temp_file_shares_the_targets_directory(tmp_path):
    """rename() is only atomic within one filesystem, so the temp file cannot
    live in /tmp or anywhere else the target might not be."""
    seen = {}
    p = tmp_path / "sub" / "y.json"
    real_replace = type(p).replace

    def spy(self, target):
        seen["tmp_parent"] = self.parent
        return real_replace(self, target)

    import pathlib
    orig = pathlib.Path.replace
    pathlib.Path.replace = spy
    try:
        atomic_write(p, "{}")
    finally:
        pathlib.Path.replace = orig
    assert seen["tmp_parent"] == p.parent


def test_the_real_call_sites_use_it():
    """Regression guard: these are the files worth protecting, and it is easy
    for a new one to reach for write_text out of habit."""
    import inspect

    from bingxbot import config
    from bingxbot.engine import scanner, search
    from bingxbot.server import orchestrator
    for mod, fn in ((orchestrator, "save_champions"),
                    (orchestrator, "save_overlays"),
                    (config, "save_config"),
                    (search, None)):
        src = inspect.getsource(mod)
        assert "atomic_write" in src, f"{mod.__name__} should write atomically"
    assert "write_text" not in inspect.getsource(orchestrator.Orchestrator.save_champions)
    assert "write_text" not in inspect.getsource(scanner)
