"""Truncated compressed data must degrade, not escalate.

A truncated gzip raises EOFError, which is NOT an OSError. Three readers in this
project catch "unreadable" and one of them already got it right (the meta model
listed EOFError); the other two did not, and each turned a recoverable bad blob
into something much worse.
"""
import gzip

import pytest


def test_a_truncated_websocket_frame_is_skipped_not_escalated():
    """_decode exists to turn a bad frame into 'skip it'. EOFError used to
    escape to the run loop's catch-all, which tears down and rebuilds the whole
    socket — a data gap on EVERY symbol on that connection, from one frame."""
    import aiohttp

    from bingxbot.exchange.ws import _decode

    good = gzip.compress(b'{"e":"trade"}')

    class _Msg:
        type = aiohttp.WSMsgType.BINARY

        def __init__(self, data):
            self.data = data

    assert _decode(_Msg(good)) == '{"e":"trade"}'
    with pytest.raises(EOFError):                    # the hazard is real...
        gzip.decompress(good[: len(good) // 2])
    assert _decode(_Msg(good[: len(good) // 2])) is None, "...and is now skipped"
    assert _decode(_Msg(b"not gzip at all")) is None


def test_a_truncated_tape_day_returns_empty(tmp_path):
    """Rotation gzips a finished day; a power cut mid-rotation leaves exactly
    this file. The reader documents itself as tolerant of a torn LINE — it has
    to survive a torn FILE too."""
    from bingxbot.data.tape import read_day

    d = tmp_path / "BTC-USDT"
    d.mkdir(parents=True)
    gz = d / "trades-2026-03-02.csv.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        for i in range(300):
            f.write(f"{i},100.0,1.0,0\n")
    raw = gz.read_bytes()
    assert read_day(tmp_path, "BTC-USDT", "trades", "2026-03-02"), "reads when intact"

    gz.write_bytes(raw[: len(raw) // 2])
    assert read_day(tmp_path, "BTC-USDT", "trades", "2026-03-02") == []


def test_a_missing_tape_day_is_still_just_empty(tmp_path):
    from bingxbot.data.tape import read_day
    assert read_day(tmp_path, "BTC-USDT", "trades", "2026-01-01") == []


def test_the_tape_reader_does_not_leak_its_file_handle(tmp_path):
    """The old form was `gzip.open(...).read()` with no context manager."""
    import inspect

    from bingxbot.data import tape
    src = inspect.getsource(tape.read_day)
    assert "with gzip.open(" in src


def test_the_meta_model_loader_already_handled_this():
    """Recorded so the asymmetry that made the other two easy to miss is
    visible: this one was always right."""
    import inspect

    from bingxbot.ml.meta import MetaModel
    assert "EOFError" in inspect.getsource(MetaModel.load)
