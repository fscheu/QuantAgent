import os

import pandas as pd
import requests

BASE_URL = "https://api.stlouisfed.org/fred/"


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
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(str(exc).replace(self._key, "***")) from None

    def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
        params = {"series_id": serie, "observation_start": str(desde),
                  "observation_end": str(hasta), "limit": 100000}
        if todas_las_versiones:
            params.update(realtime_start="1776-07-04", realtime_end="9999-12-31")
        filas = []
        while True:
            data = self._get("series/observations", {**params, "offset": len(filas)})
            filas += data["observations"]
            if not data["observations"] or len(filas) >= data["count"]:
                break
        df = pd.DataFrame(filas, columns=["date", "value", "realtime_start", "realtime_end"])
        df["valor"] = pd.to_numeric(df.pop("value"), errors="coerce")
        for c in ("date", "realtime_start", "realtime_end"):
            df[c] = pd.to_datetime(df[c])
        return df.rename(columns={"date": "fecha"})[
            ["fecha", "valor", "realtime_start", "realtime_end"]]
