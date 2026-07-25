"""Tuner round four: an honest objective, an evidence floor on promotion, and
a wider funnel to the real judge."""
import pytest

from bingxbot.engine.autotuner import (MIN_VETO_TRADES, TOP_K_VALIDATE,
                                       _centroid)
from bingxbot.engine.backtest import EVIDENCE_K, FITNESS_VER, _fitness


def _stats(trades, ret, dd=0.03, pf=2.0):
    return {"trades": trades, "total_return": ret, "max_drawdown": dd,
            "profit_factor": pf}


def test_evidence_shrinkage_prices_sample_size():
    """The same window growth must be worth strictly more when more trades
    stand behind it — that ordering is what stops the search buying rank with
    leverage instead of skill."""
    thin = _fitness(_stats(6, 0.10))
    mid = _fitness(_stats(20, 0.10))
    fat = _fitness(_stats(80, 0.10))
    assert 0 < thin < mid < fat
    # and the discount is the documented one, not a token nudge
    assert thin == pytest.approx(fat * (6 / (6 + EVIDENCE_K)) / (80 / (80 + EVIDENCE_K)), rel=1e-9)
    assert thin / fat < 0.5, "a 6-trade window must not rival an 80-trade one"


def test_doubling_risk_no_longer_buys_rank_over_evidence():
    """The overfit channel the live vault kept walking into: a set that pins
    risk at the box maximum triples its in-sample growth on a lucky handful of
    fills. Under the old 0.8-1.4 evidence multiplier that beat a modest set
    with four times the evidence; it must not any more."""
    # NOTE the drawdowns are equal on purpose: that is what makes this the
    # hard case. A lucky window is precisely one where the extra risk did NOT
    # show up as heat, so the convex drawdown penalty never fires and only the
    # evidence discount can tell the two apart.
    lucky_leveraged = _fitness(_stats(6, 0.30))     # 3x growth, 6 trades
    modest_proven = _fitness(_stats(40, 0.12))      # real growth, 40 trades
    assert modest_proven > lucky_leveraged
    old_multiplier = lambda t: min(max(0.8 + 0.2 * (t / 30.0), 0.8), 1.4)  # noqa: E731
    assert old_multiplier(6) / old_multiplier(40) > 0.5, "documents the weak old rule"


def test_losers_keep_their_gradient():
    """Shrinkage must never be applied to losing sets: making frequent junk
    look safer than confident junk would destroy the search's direction."""
    junk_often = _fitness(_stats(60, -0.10, pf=0.3))
    junk_rare = _fitness(_stats(6, -0.10, pf=0.3))
    assert junk_often < 0 and junk_rare < 0
    assert junk_often == pytest.approx(junk_rare, rel=1e-9), "loser branch is sample-blind"
    assert _fitness(_stats(30, -0.10, pf=0.2)) < _fitness(_stats(30, -0.10, pf=0.9))


def test_unjudgeable_windows_still_score_below_everything():
    assert _fitness(_stats(3, 5.0)) < 0
    assert _fitness(_stats(0, 5.0)) < _fitness(_stats(4, 5.0))


def test_fitness_version_bumped_for_the_new_scale():
    assert FITNESS_VER >= 4, "a changed fitness scale must invalidate old birth scores"


def test_thin_fold_cannot_mint_a_champion():
    """Regression for champion 89b19991: promoted off a newest fold with THREE
    trades and no losers (PF 999 sails past a bare PF>=1 veto) while the
    objective scored that same fold -1.4 for being unjudgeable."""
    assert MIN_VETO_TRADES >= 5
    thin = {"trades": 3, "profit_factor": 999.0}
    fat = {"trades": 12, "profit_factor": 1.4}
    def promotable(s):
        return (int(s.get("trades", 0) or 0) >= MIN_VETO_TRADES
                and float(s.get("profit_factor", 0.0) or 0.0) >= 1.0)
    assert not promotable(thin), "3 lucky fills must not clear the profit veto"
    assert promotable(fat)
    assert _fitness({"trades": 3, "total_return": 0.2, "max_drawdown": 0.01,
                     "profit_factor": 999.0}) < 0, "the objective already refuses it"


def test_funnel_is_wider_than_the_old_five():
    assert TOP_K_VALIDATE >= 10, "the judged funnel, not the search, was the bottleneck"


def test_centroid_respects_tunable_types():
    """The consensus candidate must be a LEGAL parameter set: ints rounded,
    bools voted, floats averaged — anything else would be rejected downstream
    or, worse, silently coerced into a different strategy."""
    a = {"base_threshold": 0.10, "horizon_bars": 6, "trade_range": True, "kelly_fraction": 0.20}
    b = {"base_threshold": 0.20, "horizon_bars": 9, "trade_range": True, "kelly_fraction": 0.40}
    c = {"base_threshold": 0.30, "horizon_bars": 10, "trade_range": False, "kelly_fraction": 0.60}
    mid = _centroid([a, b, c])
    assert mid["base_threshold"] == pytest.approx(0.20)
    assert mid["horizon_bars"] == 8 and isinstance(mid["horizon_bars"], int)
    assert mid["trade_range"] is True, "2 of 3 voted on"
    assert mid["kelly_fraction"] == pytest.approx(0.40)
    assert _centroid([]) == {}


def test_centroid_ignores_unknown_keys():
    mid = _centroid([{"base_threshold": 0.2, "not_a_tunable": 5.0}])
    assert "not_a_tunable" not in mid and mid["base_threshold"] == pytest.approx(0.2)
