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


class SesionConTopeDeVintages:
    """Imita a ALFRED: 400 si el rango realtime pedido tiene más de 2000 fechas de vintage.

    UNRATE cambia de valor en cada vintage (3000 versiones de una fecha); PIB nunca cambia (una versión desde el primer vintage).
    Los intervalos devueltos se recortan a la ventana pedida, como hace la API.
    """

    def __init__(self, n=3000):
        self.vint = [(pd.Timestamp("2000-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]
        self.pedidos = []

    def get(self, url, params=None, timeout=None):
        assert url.startswith("https://api.stlouisfed.org/fred/"), url
        self.pedidos.append((url.rsplit("/", 1)[1], params.get("realtime_start"), params.get("realtime_end")))
        rs, re = params.get("realtime_start", "1776-07-04"), params.get("realtime_end", "9999-12-31")
        en_rango = [v for v in self.vint if rs <= v <= re]
        if url.endswith("series/observations") and len(en_rango) > 2000:
            msg = f"There are {len(en_rango)} vintage dates in the specified real-time period {rs} to {re}, which exceeds the maximum (2000)."
            return SimpleNamespace(status_code=400, json=lambda: {"error_message": msg}, raise_for_status=lambda: None)
        ok = lambda: None  # noqa: E731
        if url.endswith("series/vintagedates"):
            body = {"count": len(self.vint), "vintage_dates": self.vint[params["offset"]:][: params["limit"]]}
        else:
            dia_antes = lambda d: (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # noqa: E731
            fines = [dia_antes(v) for v in self.vint[1:]] + ["9999-12-31"]
            versiones = [("2020-01-01", str(i), v, fin) for i, (v, fin) in enumerate(zip(self.vint, fines))]
            versiones += [("2019-12-01", "5.5", self.vint[0], "9999-12-31")]
            obs = [
                {"date": f, "value": val, "realtime_start": max(a, rs), "realtime_end": min(b, re)}
                for f, val, a, b in versiones
                if a <= re and b >= rs
            ]
            body = {"count": len(obs), "observations": obs[params["offset"]:][: params["limit"]]}
        return SimpleNamespace(status_code=200, raise_for_status=ok, json=lambda: body)


def test_rango_con_mas_de_2000_vintages_se_parte_en_ventanas_y_no_duplica():
    ses = SesionConTopeDeVintages()
    df = FredClient(session=ses).observaciones("DGS10", "1776-07-04", "9999-12-31")
    ventanas = [p for p in ses.pedidos if p[0] == "observations"][1:]  # el primer pedido es el rango entero, rechazado
    assert len(ventanas) == 3
    cambia = df[df["fecha"] == "2020-01-01"]
    assert len(cambia) == 3000 and cambia["valor"].tolist() == [float(i) for i in range(3000)]
    # cada versión empieza el día de su vintage y termina el día antes del siguiente (último: 9999-12-31)
    assert (cambia["realtime_start"].iloc[1:].values - pd.Timedelta(days=1) == cambia["realtime_end"].iloc[:-1].values).all()
    assert cambia["realtime_end"].iloc[-1] == pd.Timestamp("9999-12-31")
    # la versión única que atraviesa las 3 ventanas vuelve entera, no en tres pedazos
    constante = df[df["fecha"] == "2019-12-01"]
    assert len(constante) == 1
    assert constante["realtime_start"].iloc[0] == pd.Timestamp("2000-01-01") and constante["realtime_end"].iloc[0] == pd.Timestamp("9999-12-31")


def test_rango_chico_no_pide_vintagedates():
    ses = SesionConTopeDeVintages(n=50)
    df = FredClient(session=ses).observaciones("UNRATE", "1776-07-04", "9999-12-31")
    assert [p[0] for p in ses.pedidos] == ["observations"] and len(df) == 51
