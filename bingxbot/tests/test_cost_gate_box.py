"""The one tunable box this evidence justifies moving, and the ones it does not.

An owner asking why nothing trades wants the gates loosened. Measured on the
live champion's own parameters over 5,680 decision bars per symbol, most of that
instinct is wrong and would cost money:

  knob relaxed                     pooled OOS return   trades
  champion as-is                        -1.57%          126
  min_efficiency 0.4816 -> 0.28         -4.59%          136   <- much worse
  min_efficiency 0.4816 -> 0.20         -3.25%          138   <- much worse
  min_p_win floor 0.4564 -> 0.50        +3.71%          129   <- one set only

The min_p_win result did not survive a population test: lifting 47 sets that sat
below the coin flip helped 27 of them (57%) with a median change of +0.10pp.
Noise. The floor stays where it is.

cost_multiple is the exception, and it is the only box changed:

  * the gate duplicates the EV floor with a noisier statistic — a beta-scaled
    predicted MOVE against cost, where the EV floor prices the real stop
    distance and the measured payoff ratio;
  * the DE's own training objective scores the ceiling WORSE than the floor:
    -1.25 at 3.2 against -0.88 at 1.2;
  * out-of-sample across the box it is pure noise: 1.2 -> -0.59%, 1.6 -> -1.63%,
    2.0 -> -0.77%, 2.8 -> -1.33%, non-monotonic;
  * and it costs a quarter of the entries. Clamping the live champion from 3.2
    to the new 2.4 ceiling took BTC from 240 to 298 opening bars in 5,680
    (+24%), ETH from 227 to 260 (+15%), with the cost gate's share of refusals
    falling from 15.3% to 10.3%.

Nothing measured pays for the top of that box.
"""
from bingxbot.config import StrategyConfig
from bingxbot.engine.backtest import TUNABLES, _apply_params
from bingxbot.config import RiskConfig


def test_the_cost_multiple_ceiling_came_down():
    lo, hi, grp, kind = TUNABLES["cost_multiple"]
    assert hi <= 2.4, (
        "the top of this box scores worse on the search's own objective, is "
        "noise out of sample, and costs a quarter of the entries")
    assert lo <= 1.4 <= hi, "the historical default must stay reachable"


def test_the_default_still_lives_inside_its_box():
    """A default outside its own box is silently clamped on every apply, which
    makes the configured value a lie."""
    lo, hi, _g, _k = TUNABLES["cost_multiple"]
    assert lo <= StrategyConfig().cost_multiple <= hi


def test_a_champion_carrying_the_old_ceiling_is_pulled_into_the_box():
    """The live champion holds cost_multiple 3.2. Parameter sets arrive from the
    vault and from the HTTP apply endpoint as well as from the DE, so the clamp
    is what actually retires the old value — no promotion required."""
    s, _r = _apply_params(StrategyConfig(), RiskConfig(), {"cost_multiple": 3.2})
    assert s.cost_multiple == TUNABLES["cost_multiple"][1] == 2.4


def test_the_quality_gates_were_left_alone():
    """Relaxing these is the intuitive fix and measurement says it loses money:
    min_efficiency at 0.28 took the champion from -1.57% to -4.59% out of
    sample. Their boxes stay where they are, and this test says why out loud so
    the next person does not have to rediscover it."""
    assert TUNABLES["min_efficiency"] == (0.06, 0.50, "strategy", "float")
    assert TUNABLES["min_p_win"] == (0.40, 0.60, "strategy", "float")
    assert TUNABLES["base_threshold"][0] <= 0.06, (
        "the threshold floor was lowered on earlier evidence and nothing here "
        "contradicts it")
