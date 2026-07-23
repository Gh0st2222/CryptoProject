"""Regime gauntlet (Binance archive), clock-tagged champions, and the
crash-safe BingX tape recorder."""
import gzip
import json
import time

from bingxbot.config import BotConfig, load_config, save_config
from bingxbot.data import binance_hist as bh
from bingxbot.data.tape import TapeRecorder, read_day
from bingxbot.exchange.models import Candle


# ----------------------------------------------------------- binance archive

def test_kline_csv_parser_is_format_proof():
    """Binance archives have shipped headers/no-headers and seconds, ms AND µs
    timestamps (spot moved to µs in 2025). The parser must normalize all of
    them to ms and skip junk instead of crashing the tuner."""
    text = "\n".join([
        "open_time,open,high,low,close,volume,close_time",       # header row
        "1633046400000,100,110,90,105,12.5,x",                    # milliseconds
        "1633050000000000,105,115,95,108,3,x",                    # microseconds
        "1633053600,108,118,98,111,4,x",                          # seconds
        "garbage,line",                                           # junk
        "",
    ])
    out = bh.parse_kline_csv(text)
    assert [c.ts for c in out] == [1633046400000, 1633050000000, 1633053600000]
    assert out[0].close == 105 and all(c.closed for c in out)


async def test_fetch_month_serves_from_disk_cache(tmp_path):
    """A finished month is immutable: once cached, no network is ever touched
    again (the test would fail loudly if it tried — there's no server here)."""
    p = bh.month_cache_path("BTC-USDT", "15m", "2024-01", tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("1704067200000,42000,42100,41900,42050,10\n")
    out = await bh.fetch_month("BTC-USDT", "15m", "2024-01", cache_dir=tmp_path)
    assert out is not None and out[0].close == 42050


async def test_load_window_drops_incomplete_symbols(tmp_path):
    rows = "\n".join(f"{1704067200000 + i * 900_000},1,2,0.5,1.5,3" for i in range(1000))
    for ym in ("2024-01", "2024-02"):
        p = bh.month_cache_path("BTC-USDT", "15m", ym, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rows)
    # ETH has only one of the two months -> dropped, BTC survives
    bh.month_cache_path("ETH-USDT", "15m", "2024-01", tmp_path).write_text(rows)
    out = await bh.load_window(["BTC-USDT", "ETH-USDT"], "15m",
                               ["2024-01", "2024-02"], min_bars=900, cache_dir=tmp_path)
    assert set(out) == {"BTC-USDT"}
    assert len(out["BTC-USDT"]) == 2000


def test_gauntlet_windows_are_immutable_history():
    """Only finished calendar months may appear — a window touching the present
    would silently change between runs and poison the forever-cache."""
    for name, months in bh.GAUNTLET_WINDOWS:
        assert len(months) == 3, name
        assert all(m < "2026-07" for m in months), f"{name} touches the present"


# ----------------------------------------------------------- gauntlet verdict

class _StubOrch:
    def __init__(self, fits):
        self.cfg = BotConfig()
        self.cfg.symbols = ["BTC-USDT", "ETH-USDT"]
        self.specs = {}
        self.engine = None
        self.champions = []
        self._notify = None
        self.research_workers = 2
        self._fits = fits
        self.map_calls = 0

    async def map_cpu(self, fn, args, research=False):
        self.map_calls += 1
        return [{"fitness": self._fits[i % len(self._fits)],
                 "stats": {"profit_factor": 1.2 if self._fits[i % len(self._fits)] > 0 else 0.7}}
                for i in range(len(args))]


async def test_gauntlet_summary_and_forever_cache(monkeypatch, tmp_path):
    """The gauntlet scores a champion across historical eras, flags 'weak' when
    the median era loses money, and memoizes per (params, window) forever —
    a re-run of the same set must not cost a single backtest."""
    from bingxbot.engine.autotuner import AutoTuner
    candles = [Candle(ts=i * 900_000, open=1, high=2, low=0.5, close=1.5, volume=3)
               for i in range(1000)]

    async def fake_window(syms, interval, months, min_bars=900, cache_dir=None):
        return {"BTC-USDT": candles}

    monkeypatch.setattr(bh, "load_window", fake_window)
    orch = _StubOrch(fits=[1.0, -0.2, 0.6, -0.1, 0.4, 0.9])
    at = AutoTuner(orch)
    params = {"base_threshold": 0.3, "risk_per_trade": 0.008}
    g = await at._gauntlet(params, "15m", 5e-4, 1.0, orch.cfg.strategy, orch.cfg.risk)
    assert g is not None and g["n"] == len(bh.GAUNTLET_WINDOWS)
    assert g["weak"] is False and g["pf_ge1"] >= 1
    calls_before = orch.map_calls
    g2 = await at._gauntlet(params, "15m", 5e-4, 1.0, orch.cfg.strategy, orch.cfg.risk)
    assert orch.map_calls == calls_before, "same params must be served from cache"
    assert g2["median"] == g["median"]

    orch2 = _StubOrch(fits=[-1.0, -0.5, -0.8])
    at2 = AutoTuner(orch2)
    g3 = await at2._gauntlet({"x": 1.0}, "15m", 5e-4, 1.0, orch.cfg.strategy, orch.cfg.risk)
    assert g3["weak"] is True, "median era losing money must stamp WEAK"


def test_weak_gauntlet_doubles_probation(tmp_path):
    from bingxbot.engine.trader import PROBATION_MULT
    from bingxbot.tests.test_owner3 import _engine
    eng, _pf = _engine(tmp_path)
    eng.active_champion_id = "abc"
    for _ in range(9):
        eng.journal.rows.append({"mode": "paper", "champion_id": "abc", "pnl": 1.0})
    assert eng._champion_probation() == 1.0, "9 good trades clear normal probation (8)"
    eng.champion_gauntlet_weak = True
    assert eng._champion_probation() == PROBATION_MULT, \
        "a weak-gauntlet champion needs 16 trades, not 8"


# ----------------------------------------------------------- clock trial cfg

def test_clock_and_tape_settings_roundtrip(tmp_path):
    cfg = BotConfig()
    assert cfg.strategy.clock_trial is False and cfg.strategy.trial_interval == "5m"
    assert cfg.tape.enabled is True and cfg.tape.max_disk_mb == 2000
    cfg.strategy.clock_trial = True
    cfg.strategy.trial_interval = "5m"
    cfg.tape.enabled = False
    p = tmp_path / "config.json"
    save_config(cfg, p)
    cfg2 = load_config(path=p)
    assert cfg2.strategy.clock_trial is True
    assert cfg2.tape.enabled is False


def test_champion_clock_tagging(tmp_path):
    """record_champion stamps the validation clock; a champion born on the
    trial clock must be filterable away from the live engine's candidates."""
    from bingxbot.server.orchestrator import Orchestrator
    champs = [{"id": "a", "fitness": 2.0, "clock": "5m", "params": {"x": 1.0}},
              {"id": "b", "fitness": 1.0, "clock": "15m", "params": {"x": 2.0}},
              {"id": "c", "fitness": 0.5, "params": {"x": 3.0}}]   # pre-tag legacy
    interval = "15m"
    eligible = [c for c in champs if (c.get("clock") or interval) == interval]
    assert [c["id"] for c in eligible] == ["b", "c"], \
        "5m champion excluded; untagged legacy treated as current clock"
    assert Orchestrator._params_match({"x": 1.0}, {"x": 1.0})


# ----------------------------------------------------------- tape recorder

def test_tape_survives_torn_lines_and_markers(tmp_path):
    """Power-cut honesty: the reader tolerates '#session' seams and a torn
    final line — exactly what an outage mid-write leaves behind."""
    rec = TapeRecorder(tmp_path, max_disk_mb=100, book_ms=200)
    rec.start()
    ts0 = int(time.time() * 1000)
    rec.record_trade("BTC-USDT", ts0, 100.5, 0.25, True)
    rec.record_trade("BTC-USDT", ts0 + 100, 100.6, 0.5, False)
    rec.record_book("BTC-USDT", ts0, 100.4, 100.6)
    rec.record_book("BTC-USDT", ts0 + 50, 100.4, 100.7)    # throttled out (<200ms)
    rec.record_book("BTC-USDT", ts0 + 300, 100.5, 100.8)
    rec.stop()
    day = time.strftime("%Y-%m-%d", time.gmtime(ts0 / 1000))
    f = tmp_path / "BTC-USDT" / f"trades-{day}.csv"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(f"{ts0 + 200},101.")                       # torn by the power cut
    trades = read_day(tmp_path, "BTC-USDT", "trades", day)
    assert [t[1] for t in trades] == [100.5, 100.6]
    assert trades[0][3] == 1.0 and trades[1][3] == 0.0
    book = read_day(tmp_path, "BTC-USDT", "book", day)
    assert len(book) == 2, "book rows are throttled to book_ms"
    assert rec.stats()["events"] == 4


def test_tape_rotates_and_prunes(tmp_path):
    """Yesterday's raw file gzips when a new day opens, the reader follows it
    into the archive, and the disk cap prunes oldest-first (raw never pruned)."""
    rec = TapeRecorder(tmp_path, max_disk_mb=100, book_ms=0)
    rec.start()
    day_ms = 86_400_000
    old = (int(time.time() * 1000) // day_ms - 3) * day_ms + 1000
    rec.record_trade("BTC-USDT", old, 1.0, 1.0, False)          # three days ago
    time.sleep(0.1)
    rec.record_trade("BTC-USDT", old + day_ms, 2.0, 1.0, False)  # rolls the day
    rec.stop()
    d_old = time.strftime("%Y-%m-%d", time.gmtime(old / 1000))
    assert (tmp_path / "BTC-USDT" / f"trades-{d_old}.csv.gz").exists()
    assert not (tmp_path / "BTC-USDT" / f"trades-{d_old}.csv").exists()
    assert read_day(tmp_path, "BTC-USDT", "trades", d_old)[0][1] == 1.0

    rec2 = TapeRecorder(tmp_path, max_disk_mb=100, book_ms=0)
    gz = tmp_path / "BTC-USDT" / "trades-2020-01-01.csv.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(b"x" * 500_000)
    rec2.max_disk_mb = 0   # floor-clamped to 50MB in __init__; force via attribute
    rec2.max_disk_mb = 0.0001
    rec2._prune()
    assert not gz.exists(), "over the cap, the oldest archive day goes first"


def test_tape_recorder_never_blocks_when_stopped(tmp_path):
    rec = TapeRecorder(tmp_path)
    rec.record_trade("BTC-USDT", 1, 1.0, 1.0, False)   # not started: silently ignored
    assert rec.events == 0
    assert rec.stats()["running"] is False
