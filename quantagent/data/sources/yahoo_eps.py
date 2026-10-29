from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


class YahooEpsClient:
    """Cliente para obtener revisiones y tendencia de EPS desde Yahoo Finance vía yfinance."""

    def __init__(self, session: Any = None) -> None:
        self._session = session

    def estimaciones_analistas(self, simbolo: str) -> list[dict[str, Any]]:
        """Devuelve estimaciones y revisiones de EPS para un símbolo dado."""
        simbolo_yahoo = simbolo.replace(".", "-")
        try:
            ticker = yf.Ticker(simbolo_yahoo, session=self._session)
            tr = ticker.eps_trend
            rev = ticker.eps_revisions
        except Exception:
            return []

        tr_ok = tr is not None and isinstance(tr, pd.DataFrame) and not tr.empty
        rev_ok = rev is not None and isinstance(rev, pd.DataFrame) and not rev.empty

        if not tr_ok and not rev_ok:
            return []

        if tr_ok:
            tr = tr.rename(
                columns={
                    "7daysAgo": "d7",
                    "30daysAgo": "d30",
                    "60daysAgo": "d60",
                    "90daysAgo": "d90",
                }
            )

        if rev_ok:
            rev = rev.rename(
                columns={
                    "upLast7days": "up_7d",
                    "upLast30days": "up_30d",
                    "downLast7Days": "down_7d",
                    "downLast30days": "down_30d",
                }
            )

        if tr_ok and rev_ok:
            if "currency" in rev.columns and "currency" in tr.columns:
                rev = rev.drop(columns=["currency"])
            df = tr.join(rev, how="outer", sort=False)
        elif tr_ok:
            df = tr.copy()
        else:
            df = rev.copy()

        df.index.name = "period"
        df = df.reset_index()
        df["symbol"] = simbolo

        filas = df.to_dict(orient="records")
        for fila in filas:
            for k, v in list(fila.items()):
                if pd.isna(v):
                    fila[k] = None
        return filas
