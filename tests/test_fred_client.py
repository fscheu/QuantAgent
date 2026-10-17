import json
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from quantagent.data.sources.fred import FredClient

FIX = pathlib.Path(__file__).parent / "fixtures" / "fred"
CLAVE = "clave-falsa-123"


class SesionFalsa:
    def __init__(self, paginas=(), error=None):
        self.paginas, self.error, self.llamadas = list(paginas), error, []

    def get(self, url, params=None, timeout=None):
        assert url.startswith("https://api.stlouisfed.org/fred/"), url
        self.llamadas.append(params)
        ok = lambda: (_ for _ in ()).throw(requests.HTTPError(self.error)) if self.error else None  # noqa: E731
        return SimpleNamespace(raise_for_status=ok, json=lambda: self.paginas.pop(0))


@pytest.fixture(autouse=True)
def clave(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", CLAVE)


def test_dos_paginas_concatena_tipos_y_punto_es_nan():
    ses = SesionFalsa(json.loads((FIX / f"observaciones_p{i}.json").read_text()) for i in (1, 2))
    df = FredClient(session=ses).observaciones("CPIAUCSL", "2020-01-01", "2020-12-31")
    assert [c["offset"] for c in ses.llamadas] == [0, 2]
    assert ses.llamadas[0]["realtime_start"] == "1776-07-04"
    assert df["valor"].dtype == float and len(df) == 3 and pd.isna(df["valor"].iloc[2])
    assert df["valor"].iloc[:2].tolist() == [100.5, 101.0]
    assert df["realtime_end"].iloc[1] == pd.Timestamp("9999-12-31")


def test_sin_clave_falla(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY")
    with pytest.raises(RuntimeError, match="falta FRED_API_KEY"):
        FredClient()


def test_error_http_no_contiene_la_clave():
    ses = SesionFalsa(error=f"400 for url: https://x/fred/s?api_key={CLAVE}")
    with pytest.raises(RuntimeError) as e:
        FredClient(session=ses).observaciones("X", "2020-01-01", "2020-12-31")
    assert CLAVE not in str(e.value) and "***" in str(e.value)


def test_url_ajena_hace_fallar_la_sesion():
    with pytest.raises(AssertionError):
        SesionFalsa().get("https://example.com/fred/x")
