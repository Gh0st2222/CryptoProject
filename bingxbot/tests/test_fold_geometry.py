"""The search's folds must carry as much evidence as the judge's.

MIN_OOS_TRADED_BARS widened the JUDGE's folds on measured evidence. The judge's
folds carry warmup as a lead-in (portfolio_folds slices cs[lo-warmup:hi]), so
900 there means 900 bars that can trade. The SEARCH's folds are a plain
contiguous split with no lead-in, and their count was `however many research
workers there are` -- 6000/8 = 750 bars, 450 traded, half the evidence the
judge was measured to need. Nobody converted the units.

Measured on 80 candidates over one training window partitioned four ways --
same window, same candidates, same OOS truth, so only fold length differs:

  folds  bars   traded   evals under the t<5 ramp   rho vs OOS return   top-10 OOS
    8     750     450             68.3%                   +0.010            +0.87%
    4    1500    1200             36.7%                   +0.156            +5.69%
    3    2000    1700             24.4%                   +0.222            +5.49%
    2    3000    2700             15.6%                   +0.204            +7.89%

Population median OOS return was +2.77%: at 8 folds the ten sets the search
nominated were WORSE than a random draw.

That measures ranking correlation, which is not what an owner receives. So the
search was also run FOR REAL under the old and new configurations -- 8 paired
seeds, 30 generations each, same window, same judged folds -- and the top 5 of
every run were put through the actual judge:

  measure (NEW - OLD)          wins    median delta
  best of top-k, OOS return     8/8       +33.7pp
  median of top-k               8/8       +16.5pp
  worst of top-k                8/8       +10.6pp
  top-k that made money         8/8         +3 of 5

  OLD: 14 of 40 nominated sets profitable, median  -0.34%
  NEW: 40 of 40 nominated sets profitable, median +16.11%

Every pair on every measure, and the trade count moved DOWN (median 120-166 per
judged window before, 71-119 after): better selection, not more activity. The
magnitudes belong to synthetic_candles, a near-efficient generator, and are not
a forecast of BingX returns -- but both arms saw identical data, so the
comparison holds.
"""
import pytest

from bingxbot.config import RiskConfig, StrategyConfig
from bingxbot.data.history import synthetic_candles
from bingxbot.engine.autotuner import (MAX_TRAIN_FOLDS, MIN_TRAIN_TRADED_BARS,
                                       TRAIN_BARS_CAP, _shard_candidates,
                                       _stitch, _train_fold_count)
from bingxbot.engine.backtest import WARMUP_BARS
from bingxbot.engine.search import DEOptimizer, score_fold
from bingxbot.exchange.models import ContractSpec


def _traded(train_bars: int, workers: int) -> int:
    k = _train_fold_count(train_bars, workers)
    return train_bars // k - WARMUP_BARS


def test_every_training_fold_clears_the_evidence_floor():
    """Whatever the host, a fold must be able to trade for at least
    MIN_TRAIN_TRADED_BARS bars after paying its warmup."""
    for workers in (1, 2, 4, 8, 16, 64):
        for train_bars in (3000, 4500, 6000, 9000, 12000, 24000):
            k = _train_fold_count(train_bars, workers)
            assert 1 <= k <= MAX_TRAIN_FOLDS
            if k > 1:   # a single fold is the whole window; nothing to shorten
                assert train_bars // k - WARMUP_BARS >= MIN_TRAIN_TRADED_BARS, (
                    f"{k} folds of {train_bars} bars on {workers} workers leaves "
                    f"{train_bars // k - WARMUP_BARS} traded bars")


def test_more_cores_can_no_longer_shred_the_window():
    """The regression this file exists for. Adding workers used to divide the
    same data into ever-shorter folds; now it never shortens them."""
    base = _traded(TRAIN_BARS_CAP, 4)
    for workers in (8, 16, 64):
        assert _traded(TRAIN_BARS_CAP, workers) >= base


def test_production_geometry_matches_the_measured_optimum():
    """6000 training bars is TRAIN_BARS_CAP, i.e. what a live host runs. The
    measured table above puts the optimum at 3 folds of ~1700 traded bars --
    the same fold length the judge itself uses."""
    for workers in (4, 8):
        assert _train_fold_count(TRAIN_BARS_CAP, workers) == 3
        assert _traded(TRAIN_BARS_CAP, workers) == 1700


def test_fewer_cores_give_wider_folds_not_broken_ones():
    assert _train_fold_count(TRAIN_BARS_CAP, 1) == 1
    assert _train_fold_count(TRAIN_BARS_CAP, 2) == 2


def test_a_window_too_short_to_split_is_left_whole():
    assert _train_fold_count(900, 8) == 1


# --------------------------------------------------------------- sharding

def test_shards_fill_the_pool_without_slicing_candidates_too_thin():
    assert _shard_candidates(3, 8, 56) == 3     # 3 folds, 8 workers -> 9 tasks
    assert _shard_candidates(6, 8, 56) == 2     # 3 folds x 2 symbols -> 12 tasks
    assert _shard_candidates(6, 4, 56) == 1     # already more units than workers
    assert _shard_candidates(6, 8, 4) == 1      # too few candidates to split


def test_stitch_round_trips_the_stride():
    cands = list(range(11))
    for sh in (1, 2, 3, 4):
        parts = [[float(x) for x in cands[i::sh]] for i in range(sh)]
        assert _stitch(parts, len(cands), sh) == [float(x) for x in cands]


def test_a_failed_shard_never_becomes_a_verdict():
    """A worker that dies must void the whole fold. Silently reading a partial
    result as a fold's scores would permute candidate fitnesses -- the search
    would keep running and rank on nonsense."""
    assert _stitch([None, [2.0, 4.0]], 4, 2) is None
    assert _stitch([[1.0], [2.0]], 4, 2) is None
    assert _stitch([[1.0, 3.0]], 4, 2) is None


def _slow_square(x: float, delay: float) -> float:
    """Module-level so the process pool can pickle it. The delay is reversed
    across the inputs, so a pool that returned results as they COMPLETE would
    give a different order than the one they were submitted in."""
    import time
    time.sleep(delay)
    return x * x


async def test_map_cpu_returns_results_in_argument_order():
    """The sharding stitches by POSITION -- raw[i*sh:(i+1)*sh] must be unit i's
    shards. If the pool ever returned completion-ordered results instead, every
    candidate would silently inherit another candidate's fitness and the search
    would keep running on scrambled scores."""
    from bingxbot.config import BotConfig
    from bingxbot.server.orchestrator import Orchestrator

    orch = Orchestrator(BotConfig())
    n = 6
    args = [(float(i), 0.02 * (n - i)) for i in range(n)]   # slowest submitted first
    assert await orch.map_cpu(_slow_square, args) == [float(i * i) for i in range(n)]


@pytest.mark.parametrize("shards", [2, 3])
def test_sharded_scoring_is_identical_to_unsharded(shards):
    """THE property the sharding rests on: splitting the candidate list across
    workers and stitching the pieces back must give exactly what one worker
    scoring the whole list gives. If the stride and the reassembly ever
    disagree, every candidate silently inherits another candidate's fitness."""
    candles = synthetic_candles("BTC-USDT", "15m", 900, seed=5)
    spec = ContractSpec("BTC-USDT")
    de = DEOptimizer(pop_size=6, seed=3)
    de.seed_population(None)
    cands = [dict(v) for v in de.pop]
    args = ("BTC-USDT", "15m", spec, 0.0005, 1.0, StrategyConfig(), RiskConfig())

    whole = score_fold(candles, *args, cands)
    parts = [score_fold(candles, *args, cands[i::shards]) for i in range(shards)]
    assert _stitch(parts, len(cands), shards) == whole
    assert len(set(whole)) > 1, "a degenerate fold would make this test vacuous"
