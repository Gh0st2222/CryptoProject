"""A torn candle cache must be discarded, not raise.

_load_cache has always had a handler whose whole purpose is "unreadable cache?
throw it away and re-download". But a TRUNCATED gzip raises EOFError, which is
not an OSError, so it went straight through — and every later backtest or tuner
load on that symbol raised until someone deleted the file by hand.

Writing the cache straight to its final path is exactly how that file gets
created: it can be many megabytes, and a crash part-way through leaves the
truncated remains behind.
"""
import gzip

import pytest

from bingxbot.data.history import HistoryStore
from bingxbot.exchange.models import Candle


def _candles(n=500):
    return [Candle(ts=i * 900_000, open=100.0, high=101.0, low=99.0,
                   close=100.5, volume=7.0) for i in range(n)]


def _store(tmp_path):
    return HistoryStore(None, tmp_path)


def test_a_truncated_cache_is_discarded_rather_than_raising(tmp_path):
    hs = _store(tmp_path)
    hs._save_cache("BTC-USDT", "15m", _candles())
    p = hs._path("BTC-USDT", "15m")
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])          # the shape a crash leaves behind

    with pytest.raises(EOFError):                # confirm the hazard is real...
        with gzip.open(p, "rt") as f:
            f.read()
    assert hs._load_cache("BTC-USDT", "15m") == [], "...and that we survive it"


def test_garbage_bytes_are_survivable_too(tmp_path):
    hs = _store(tmp_path)
    hs._path("BTC-USDT", "15m").write_bytes(b"this is not gzip at all")
    assert hs._load_cache("BTC-USDT", "15m") == []


def test_a_malformed_row_is_survivable(tmp_path):
    hs = _store(tmp_path)
    with gzip.open(hs._path("BTC-USDT", "15m"), "wt", newline="") as f:
        f.write("0,100,101,99,100.5,7\nnot,a,number,at,all,here\n")
    assert hs._load_cache("BTC-USDT", "15m") == []


def test_the_cache_write_is_atomic(tmp_path):
    """No half-written file can exist for a reader to trip over."""
    hs = _store(tmp_path)
    hs._save_cache("BTC-USDT", "15m", _candles(50))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["BTC-USDT_15m.csv.gz"], \
        "no .tmp left behind"


def test_a_normal_round_trip_is_unchanged(tmp_path):
    hs = _store(tmp_path)
    src = _candles(120)
    hs._save_cache("BTC-USDT", "15m", src)
    back = hs._load_cache("BTC-USDT", "15m")
    assert len(back) == len(src)
    assert back[0].ts == src[0].ts and back[-1].close == pytest.approx(src[-1].close)
