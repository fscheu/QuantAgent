"""Asset type classifications for trading hours determination."""

from enum import Enum


class AssetType(Enum):
    """Classification of assets by trading schedule."""

    CRYPTO = "crypto"
    US_EQUITY = "us_equity"
    US_FUTURES = "us_futures"
    EUROPEAN = "european"
    UNKNOWN = "unknown"


ASSET_TYPE_MAPPING: dict[str, AssetType] = {
    # Crypto
    "BTC": AssetType.CRYPTO,
    "ETH": AssetType.CRYPTO,
    "BTC-USD": AssetType.CRYPTO,
    "ETH-USD": AssetType.CRYPTO,
    # US Equity Indices
    "SPX": AssetType.US_EQUITY,
    "^GSPC": AssetType.US_EQUITY,
    "QQQ": AssetType.US_EQUITY,
    "VIX": AssetType.US_EQUITY,
    "^VIX": AssetType.US_EQUITY,
    # US Futures
    "ES": AssetType.US_FUTURES,
    "ES=F": AssetType.US_FUTURES,
    "NQ": AssetType.US_FUTURES,
    "NQ=F": AssetType.US_FUTURES,
    "CL": AssetType.US_FUTURES,
    "CL=F": AssetType.US_FUTURES,
    "GC": AssetType.US_FUTURES,
    "GC=F": AssetType.US_FUTURES,
    "DXY": AssetType.US_FUTURES,
    "DX-Y.NYB": AssetType.US_FUTURES,
    # European
    "DAX": AssetType.EUROPEAN,
    "^GDAXI": AssetType.EUROPEAN,
}


def get_asset_type(symbol: str) -> AssetType:
    """
    Get asset type for a symbol.

    Args:
        symbol: Trading symbol (e.g., "BTC", "SPX")

    Returns:
        AssetType classification
    """
    if symbol in ASSET_TYPE_MAPPING:
        return ASSET_TYPE_MAPPING[symbol]

    symbol_upper = symbol.upper()

    # Crypto patterns
    if symbol_upper.endswith("-USD") or symbol_upper in (
        "BTC",
        "ETH",
        "SOL",
        "XRP",
    ):
        return AssetType.CRYPTO

    # Futures patterns
    if symbol_upper.endswith("=F") or symbol_upper in (
        "ES",
        "NQ",
        "CL",
        "GC",
        "SI",
        "HG",
    ):
        return AssetType.US_FUTURES

    # Index patterns
    if symbol_upper.startswith("^"):
        return AssetType.US_EQUITY

    return AssetType.UNKNOWN
