"""
Position Sizer: Calculates order size based on portfolio value and signal confidence.

Formula:
    position_value = portfolio_value × base_position_pct × signal_confidence
    qty = position_value / current_price

Example:
    - Portfolio value: $100,000
    - Base position %: 5%
    - Signal confidence: 50% (low confidence)
    - Current price: $42,000 (BTC)

    position_value = $100,000 × 0.05 × 0.5 = $2,500
    qty = $2,500 / $42,000 = 0.0595 BTC (2.5% position size)

    - If confidence was 100%:
    position_value = $100,000 × 0.05 × 1.0 = $5,000
    qty = $5,000 / $42,000 = 0.119 BTC (5% position size)
"""


class PositionSizer:
    """Calculates position size based on capital and signal confidence."""

    REGLAS = ("confianza", "fijo")

    def __init__(self, base_position_pct: float = 0.05, regla: str = "confianza"):
        """
        Initialize Position Sizer.

        Args:
            base_position_pct: Base percentage of portfolio per trade (default 5%, max 100%: no leverage)
            regla: "confianza" (portfolio × base % × confidence) or "fijo" (portfolio × base %)
        """
        if not 0 < base_position_pct <= 1.0:
            raise ValueError("base_position_pct must be between 0% and 100%")
        if regla not in self.REGLAS:
            raise ValueError(f"regla must be one of {self.REGLAS}, got {regla!r}")
        self.base_position_pct = base_position_pct
        self.regla = regla

    def calculate_size(
        self,
        symbol: str,
        signal_confidence: float,
        current_price: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate position size for a trade.

        Args:
            symbol: Trading symbol (e.g., "BTC", "SPX")
            signal_confidence: Confidence level (0.0 to 1.0)
                - 0.5: Low confidence → 50% of base position size
                - 1.0: High confidence → 100% of base position size
            current_price: Current market price of the asset
            portfolio_value: Total portfolio value (cash + positions)

        Returns:
            Quantity to buy/sell

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not 0 <= signal_confidence <= 1.0:
            raise ValueError("signal_confidence must be between 0 and 1.0")
        if current_price <= 0:
            raise ValueError("current_price must be positive")
        if portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")

        # Calculate position value: portfolio × base % (× confidence unless regla is "fijo")
        position_value = portfolio_value * self.base_position_pct
        if self.regla == "confianza":
            position_value *= signal_confidence

        # Calculate quantity: position_value / current_price
        qty = position_value / current_price

        return qty
