import os

import pandas as pd
import requests

BASE_URL = "https://api.stlouisfed.org/fred/"
MAX_VINTAGES = 1000  # la API rechaza pedidos con más de 2000 fechas de vintage en el rango realtime


class _DemasiadasVersiones(RuntimeError):
    pass


class FredClient:
    def __init__(self, session=None):
        self._key = os.environ.get("FRED_API_KEY")
        if not self._key:
            raise RuntimeError("falta FRED_API_KEY")
        self._session = session or requests.Session()

    def _get(self, endpoint, params):
        params = {**params, "api_key": self._key, "file_type": "json"}
        try:
            resp = self._session.get(BASE_URL + endpoint, params=params, timeout=30)
            if getattr(resp, "status_code", None) == 400 and "vintage dates" in str(resp.json().get("error_message")):
                raise _DemasiadasVersiones("rango realtime con demasiadas fechas de vintage")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(str(exc).replace(self._key, "***")) from None

    def _paginar(self, endpoint, clave, params):
        filas = []
        while True:
            data = self._get(endpoint, {**params, "offset": len(filas)})
            filas += data[clave]
            if not data[clave] or len(filas) >= data["count"]:
                return filas

    def _por_ventanas(self, params):
        """Parte el rango realtime en ventanas de MAX_VINTAGES fechas de vintage (el rango entero excede el tope de la API)."""
        vint = self._paginar("series/vintagedates", "vintage_dates", {"series_id": params["series_id"], "limit": 10000})
        filas = []
        for i in range(0, len(vint), MAX_VINTAGES):
            fin = "9999-12-31"
            if i + MAX_VINTAGES < len(vint):
                fin = (pd.Timestamp(vint[i + MAX_VINTAGES]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            filas += self._paginar("series/observations", "observations", {**params, "realtime_start": vint[i], "realtime_end": fin})
        return filas

    def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
        params = {"series_id": serie, "observation_start": str(desde),
                  "observation_end": str(hasta), "limit": 100000}
        if todas_las_versiones:
            params.update(realtime_start="1776-07-04", realtime_end="9999-12-31")
        ventanas = False
        try:
            filas = self._paginar("series/observations", "observations", params)
        except _DemasiadasVersiones:
            ventanas, filas = True, self._por_ventanas(params)
        df = pd.DataFrame(filas, columns=["date", "value", "realtime_start", "realtime_end"])
        df["valor"] = pd.to_numeric(df.pop("value"), errors="coerce")
        for c in ("date", "realtime_start", "realtime_end"):
            df[c] = pd.to_datetime(df[c])
        df = df.rename(columns={"date": "fecha"})[["fecha", "valor", "realtime_start", "realtime_end"]]
        return _unir_fragmentos(df) if ventanas else df


def _unir_fragmentos(df):
    """Une las versiones que el borde de una ventana cortó en dos (mismo valor, una empieza el día después de que la otra termina)."""
    df = df.sort_values(["fecha", "realtime_start"], ignore_index=True)
    previa = df.groupby("fecha").shift()
    continua = (df["realtime_start"] - pd.Timedelta(days=1) == previa["realtime_end"]) & (
        (df["valor"] == previa["valor"]) | (df["valor"].isna() & previa["valor"].isna())
    )
    return df.groupby((~continua).cumsum(), as_index=False).agg(
        {"fecha": "first", "valor": "first", "realtime_start": "first", "realtime_end": "last"}
    )
