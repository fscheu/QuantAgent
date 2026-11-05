"""Market ETF rows derived from stored daily bars."""

from sqlalchemy import select

from quantagent.data.market_bars import MARKET_BAR_SYMBOLS
from quantagent.models import MarketBar


ASSET_CLASSES = {
    "us_equity": {"SPY", "QQQ", "IWM"},
    "sector": {"XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"},
    "international": {"EFA", "EEM", "EWJ", "EWZ", "FXI"},
    "bonds": {"SHY", "IEF", "TLT", "TIP", "AGG", "LQD"},
    "commodities": {"GLD", "SLV", "DBC", "USO"},
    "currency": {"FXE"},
    "real_estate": {"VNQ"},
}
ETF_NAMES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "XLB": "Materiales", "XLE": "Energía", "XLF": "Finanzas", "XLI": "Industriales",
    "XLK": "Tecnología", "XLP": "Consumo básico", "XLU": "Servicios públicos",
    "XLV": "Salud", "XLY": "Consumo discrecional", "EFA": "Mercados desarrollados",
    "EEM": "Mercados emergentes", "EWJ": "Japón", "EWZ": "Brasil", "FXI": "China",
    "SHY": "Bonos del Tesoro corto plazo", "IEF": "Bonos del Tesoro intermedio",
    "TLT": "Bonos del Tesoro largo plazo", "TIP": "Bonos protegidos contra inflación",
    "AGG": "Bonos agregados", "LQD": "Bonos corporativos grado de inversión",
    "GLD": "Oro", "SLV": "Plata", "DBC": "Materias primas", "USO": "Petróleo",
    "FXE": "Euro", "VNQ": "Inmobiliario",
}
ETF_SYMBOLS = tuple(symbol for symbol in MARKET_BAR_SYMBOLS if symbol in ETF_NAMES)
MIN_SESSIONS = {"d1": 2, "w1": 6, "m1": 21, "m3": 63, "ytd": 0}


def held_in_por_simbolo(cuentas):
    """Map each held symbol to the account IDs that hold it."""
    result = {}
    for account_id, symbols in cuentas:
        for symbol in symbols:
            result.setdefault(symbol, []).append(account_id)
    return result


def retornos_simbolo(session, symbol):
    """Return percentage returns and the number of stored sessions for a symbol."""
    bars = session.scalars(select(MarketBar).where(MarketBar.symbol == symbol).order_by(MarketBar.session_date)).all()
    closes = [float(bar.close) for bar in bars]
    dates = [bar.session_date for bar in bars]
    returns = {}
    offsets = {"d1": 1, "w1": 5, "m1": 20, "m3": 62}
    for key, offset in offsets.items():
        returns[key] = ((closes[-1] / closes[-1 - offset] - 1) * 100 if len(closes) > offset and len(closes) >= MIN_SESSIONS[key] else None)
    prior_year_close = next((i for i in range(len(dates) - 1, -1, -1) if dates[i].year == dates[-1].year - 1), None) if dates else None
    returns["ytd"] = ((closes[-1] / closes[prior_year_close] - 1) * 100 if prior_year_close is not None else None)
    return returns, len(closes)


def filas_etf(session, positions_by_symbol=None):
    """Build the 29 ETF rows from daily bars, keeping symbols without data."""
    positions_by_symbol = positions_by_symbol or {}
    rows = []
    for symbol in ETF_SYMBOLS:
        returns, history_sessions = retornos_simbolo(session, symbol)
        closes = [float(bar.close) for bar in session.scalars(select(MarketBar).where(MarketBar.symbol == symbol).order_by(MarketBar.session_date)).all()]
        rows.append({
            "symbol": symbol, "name": ETF_NAMES[symbol],
            "asset_class": next(key for key, members in ASSET_CLASSES.items() if symbol in members),
            "close": closes[-1] if closes else None,
            "returns": returns, "history_sessions": history_sessions,
            "spark_30d": closes[-30:] if len(closes) >= 2 else None,
            "held_in": positions_by_symbol.get(symbol, []), "in_universe": True,
        })
    return rows
