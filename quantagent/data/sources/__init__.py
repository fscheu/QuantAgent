"""Fuentes de velas (proveedores externos) detrás de la caché."""

from quantagent import settings
from quantagent.data.sources.alpaca import AlpacaSource
from quantagent.data.sources.base import CandleSource
from quantagent.data.sources.yahoo import YahooSource


def build_source(name: str | None = None) -> CandleSource:
    """Build the configured paper candle source; reject unknown names at startup."""
    name = settings.QUANTAGENT_DATA_SOURCE if name is None else name
    sources = {"yahoo": YahooSource, "alpaca": AlpacaSource}
    if name not in sources:
        raise ValueError(f"QUANTAGENT_DATA_SOURCE={name!r}; valores válidos: {', '.join(sources)}")
    return sources[name]()
