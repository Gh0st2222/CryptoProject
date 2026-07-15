import hmac
from hashlib import sha256

from bingxbot.exchange import rest as rest_mod
from bingxbot.exchange.rest import BingXRest


def test_signature_matches_official_scheme(monkeypatch):
    """Sorted params, timestamp appended last, HMAC-SHA256 hex — per BingX docs."""
    monkeypatch.setattr(rest_mod, "now_ms", lambda: 1_700_000_000_000)
    client = BingXRest(api_key="k", api_secret="mysecret")
    encoded, sig = client._sign({"symbol": "BTC-USDT", "side": "BUY", "quantity": 0.001})

    expected_raw = "quantity=0.001&side=BUY&symbol=BTC-USDT&timestamp=1700000000000"
    expected_sig = hmac.new(b"mysecret", expected_raw.encode(), sha256).hexdigest()
    assert sig == expected_sig
    assert encoded == "quantity=0.001&side=BUY&symbol=BTC-USDT&timestamp=1700000000000"


def test_signature_encodes_json_values_but_signs_raw(monkeypatch):
    monkeypatch.setattr(rest_mod, "now_ms", lambda: 1_700_000_000_000)
    client = BingXRest(api_key="k", api_secret="s")
    sl = '{"type":"STOP_MARKET","stopPrice":64000.5}'
    encoded, sig = client._sign({"stopLoss": sl, "symbol": "BTC-USDT"})

    raw = f"stopLoss={sl}&symbol=BTC-USDT&timestamp=1700000000000"
    assert sig == hmac.new(b"s", raw.encode(), sha256).hexdigest()
    assert "%7B" in encoded and "%22" in encoded          # JSON got URL-encoded
    assert "symbol=BTC-USDT" in encoded


def test_kline_row_parsing_both_shapes():
    arr = BingXRest._parse_kline_row([1700000000000, "100", "110", "90", "105", "12.5", 1700000059999])
    obj = BingXRest._parse_kline_row(
        {"time": 1700000000000, "open": "100", "high": "110", "low": "90", "close": "105", "volume": "12.5"}
    )
    for c in (arr, obj):
        assert c.ts == 1700000000000
        assert c.open == 100 and c.high == 110 and c.low == 90 and c.close == 105
        assert c.volume == 12.5
