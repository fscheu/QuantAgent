"""AlpacaSource against recorded (invented-value) responses. No network, no real keys."""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest

from quantagent.data.sources.alpaca import AlpacaSource

FIX = Path(__file__).parent / "fixtures" / "alpaca"
NOW = datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params), headers))
        return self.responses.pop(0)


def load(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "fake-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fake-secret")


def make(responses):
    session = FakeSession(responses)
    return AlpacaSource(session=session, now=lambda: NOW, sleep=lambda s: None), session


def test_two_pages_return_all_candles_in_utc():
    src, session = make([FakeResponse(body=load("bars_page1.json")),
                         FakeResponse(body=load("bars_page2.json"))])
    df = src.fetch("TEST", "1h", datetime(2024, 1, 2), datetime(2024, 1, 3))
    assert list(df["close"]) == [101.0, 101.5, 102.5]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].iloc[0] == datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    assert "page_token" not in session.calls[0][1]
    assert session.calls[1][1]["page_token"] == "tok-2"
    assert session.calls[0][1]["feed"] == "sip"
    assert session.calls[0][1]["adjustment"] == "split"


def test_end_is_capped_at_now_minus_15_minutes():
    src, session = make([FakeResponse(body=load("bars_page2.json"))])
    src.fetch("TEST", "1h", datetime(2024, 1, 2), datetime(2024, 1, 5))
    assert session.calls[0][1]["end"] == "2024-01-03T11:45:00+00:00"
    assert src.latest() == datetime(2024, 1, 3, 11, 45, tzinfo=timezone.utc)


def test_retries_on_429_then_succeeds():
    src, session = make([FakeResponse(429), FakeResponse(429),
                         FakeResponse(body=load("bars_page2.json"))])
    df = src.fetch("TEST", "1h", datetime(2024, 1, 2), datetime(2024, 1, 3))
    assert len(df) == 1 and len(session.calls) == 3


def test_missing_keys_name_both_variables(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID")
    monkeypatch.delenv("APCA_API_SECRET_KEY")
    src, session = make([])
    with pytest.raises(RuntimeError) as exc:
        src.fetch("TEST", "1h", datetime(2024, 1, 2), datetime(2024, 1, 3))
    assert "APCA_API_KEY_ID" in str(exc.value) and "APCA_API_SECRET_KEY" in str(exc.value)
    assert session.calls == []


def test_only_market_data_host_and_path():
    src, session = make([FakeResponse(body=load("bars_page2.json"))])
    src.fetch("TEST", "1h", datetime(2024, 1, 2), datetime(2024, 1, 3))
    for url, _, _ in session.calls:
        u = urlparse(url)
        assert u.netloc == "data.alpaca.markets"
        assert u.path == "/v2/stocks/bars"
