import pytest
import requests

from quantagent.data.sources.fmp import BASE_URL, FmpClient


def test_fmp_sin_api_key_falla(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="falta FMP_API_KEY"):
        FmpClient()


def test_fmp_estimaciones_analistas(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "mi-clave-secreta")

    class DummyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "symbol": "AAPL",
                    "date": "2024-09-30",
                    "epsAvg": 6.7,
                    "epsHigh": 6.8,
                    "epsLow": 6.5,
                    "numAnalystsEps": 32,
                    "revenueAvg": 390000000000,
                }
            ]

    pedidos = []

    class DummySession:
        def get(self, url, params=None, timeout=None):
            pedidos.append((url, params, timeout))
            return DummyResponse()

    cliente = FmpClient(session=DummySession())
    res = cliente.estimaciones_analistas("AAPL")
    assert len(res) == 1
    assert res[0]["symbol"] == "AAPL"
    assert pedidos[0][0] == BASE_URL + "analyst-estimates"
    assert pedidos[0][1]["symbol"] == "AAPL"
    assert pedidos[0][1]["period"] == "annual"
    assert pedidos[0][1]["apikey"] == "mi-clave-secreta"


def test_fmp_error_oculta_api_key(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "secreto123")

    class DummySession:
        def get(self, url, params=None, timeout=None):
            raise requests.RequestException("401 Client Error with apikey=secreto123 for url")

    cliente = FmpClient(session=DummySession())
    with pytest.raises(RuntimeError) as exc_info:
        cliente.estimaciones_analistas("AAPL")
    assert "secreto123" not in str(exc_info.value)
    assert "***" in str(exc_info.value)
