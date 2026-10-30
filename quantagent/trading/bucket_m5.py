"""Perfil operativo de la cuenta paper M5."""

import os
from pathlib import Path

from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager

RIESGO_POR_TRADE_PCT = 0.0025
MAX_POSITION_PCT = 0.10
MAX_HEAT_PCT = 0.02
CAPITAL_INICIAL = 3000.0


def crear_risk_manager(portfolio_manager, **kwargs):
    return RiskManager(portfolio_manager, max_position_pct=MAX_POSITION_PCT,
                       max_heat_pct=MAX_HEAT_PCT, **kwargs)


def crear_position_sizer():
    return PositionSizer(regla="riesgo", riesgo_por_trade_pct=RIESGO_POR_TRADE_PCT,
                         max_position_pct=MAX_POSITION_PCT)


def cargar_credenciales(ruta: str | Path = "~/secrets/alpaca-cartera-m5.env") -> None:
    """Exporta ambos convenios Alpaca desde un env file sin revelar sus valores."""
    values = {}
    for line in Path(ruta).expanduser().read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    pairs = (("APCA_API_KEY_ID", "ALPACA_API_KEY"),
             ("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"))
    for apca, alpaca in pairs:
        value = values.get(apca) or values.get(alpaca)
        if value:
            os.environ[apca] = value
            os.environ[alpaca] = value

