import os
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://financialmodelingprep.com/stable/"


class FmpClient:
    """Cliente para la API de Financial Modeling Prep (FMP)."""

    def __init__(self, session: Optional[requests.Session] = None):
        self._key = os.environ.get("FMP_API_KEY")
        if not self._key:
            raise RuntimeError("falta FMP_API_KEY")
        self._session = session or requests.Session()

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Any:
        params = {**params, "apikey": self._key}
        try:
            resp = self._session.get(BASE_URL + endpoint, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(str(exc).replace(self._key, "***")) from None

    def estimaciones_analistas(self, symbol: str, period: str = "annual") -> List[Dict[str, Any]]:
        """Devuelve estimaciones de analistas para un símbolo."""
        data = self._get("analyst-estimates", {"symbol": symbol, "period": period})
        return data if isinstance(data, list) else []
