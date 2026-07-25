"""Every pushed frame must be parseable by a browser.

Python writes NaN and Infinity as bare literals. Those are NOT valid JSON, and
`JSON.parse` throws on them. The dashboard parses every frame inside
ws.onmessage, so one non-finite number anywhere in the snapshot aborts the
handler and the UI silently stops updating — connected, no server error, nothing
in the log. That is the "data stopped updating" failure, and the earlier fix
covered only the four 24h-range fields that happened to cause it then.
"""
import json
import math

import pytest

from bingxbot.server.app import _dumps, _strip_nonfinite


def _browser_parse(text: str):
    """json.loads is LENIENT about NaN; browsers are not. Reject it the way
    JSON.parse does, so this test fails for the reason the dashboard breaks."""
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"invalid JSON literal {c!r}")))


def test_python_really_does_emit_invalid_json_for_nan():
    """The premise, asserted rather than assumed."""
    raw = json.dumps({"edge": float("nan")})
    assert "NaN" in raw
    with pytest.raises(ValueError):
        _browser_parse(raw)


def test_a_nan_anywhere_still_produces_a_parseable_frame():
    payload = {"type": "state",
               "data": {"engine": {"symbols": {"BTC-USDT": {"edge": float("nan")}}}}}
    out = _browser_parse(_dumps(payload))
    assert out["data"]["engine"]["symbols"]["BTC-USDT"]["edge"] is None


def test_infinity_too():
    out = _browser_parse(_dumps({"type": "hot", "data": {"x": float("inf"),
                                                         "y": float("-inf")}}))
    assert out["data"] == {"x": None, "y": None}


def test_non_finite_values_nested_in_lists_are_caught():
    payload = {"type": "state", "data": {"equity_curve": [[1, 100.0], [2, float("nan")]],
                                         "trades": [{"pnl": float("nan")}]}}
    out = _browser_parse(_dumps(payload))
    assert out["data"]["equity_curve"][1][1] is None
    assert out["data"]["trades"][0]["pnl"] is None


def test_a_clean_frame_is_untouched_and_takes_the_fast_path():
    payload = {"type": "state", "data": {"a": 1.5, "b": [1, 2, 3], "c": {"d": "x"},
                                         "e": None, "f": True}}
    assert _dumps(payload) == json.dumps(payload, allow_nan=False)
    assert _browser_parse(_dumps(payload)) == payload


def test_the_sanitizer_preserves_everything_else():
    src = {"i": 7, "s": "text", "b": False, "n": None, "f": 0.25,
           "l": [1, "two", None], "d": {"k": 3.5}}
    assert _strip_nonfinite(src) == src


def test_the_client_guards_the_parse_too():
    """Defence in depth: a truncated frame should cost that frame only."""
    js = (__import__("pathlib").Path("bingxbot/server/static/app.js")).read_text()
    i = js.index("ws.onmessage")
    handler = js[i:i + 400]
    assert "try{" in handler and "catch" in handler, \
        "JSON.parse in onmessage must not be able to abort the handler"
