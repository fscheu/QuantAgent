"""Tests for asset type classification."""


from quantagent.data.asset_types import AssetType, get_asset_type


class TestAssetTypeClassification:
    """Test asset type classification logic."""

    def test_crypto_classification(self):
        """Test crypto assets are classified correctly."""
        assert get_asset_type("BTC") == AssetType.CRYPTO
        assert get_asset_type("ETH") == AssetType.CRYPTO
        assert get_asset_type("BTC-USD") == AssetType.CRYPTO
        assert get_asset_type("SOL-USD") == AssetType.CRYPTO

    def test_equity_classification(self):
        """Test US equity assets are classified correctly."""
        assert get_asset_type("SPX") == AssetType.US_EQUITY
        assert get_asset_type("QQQ") == AssetType.US_EQUITY
        assert get_asset_type("^GSPC") == AssetType.US_EQUITY
        assert get_asset_type("^VIX") == AssetType.US_EQUITY

    def test_futures_classification(self):
        """Test US futures assets are classified correctly."""
        assert get_asset_type("ES") == AssetType.US_FUTURES
        assert get_asset_type("ES=F") == AssetType.US_FUTURES
        assert get_asset_type("CL") == AssetType.US_FUTURES
        assert get_asset_type("GC=F") == AssetType.US_FUTURES

    def test_unknown_classification(self):
        """Test unknown assets default to UNKNOWN."""
        assert get_asset_type("CUSTOM") == AssetType.UNKNOWN
        assert get_asset_type("RANDOM123") == AssetType.UNKNOWN

    def test_case_insensitive_patterns(self):
        """Test pattern matching is case-insensitive."""
        assert get_asset_type("btc") == AssetType.CRYPTO
        assert get_asset_type("es") == AssetType.US_FUTURES
        assert get_asset_type("sol-usd") == AssetType.CRYPTO
