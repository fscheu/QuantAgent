# Position Management Strategies

**Date:** 2026-01-02
**Status:** Design Proposal

## Current Flow Analysis

### Backtest Execution Flow

```
1. Backtest._analyze_and_trade() [per asset, per period]
   └─> TradingGraph.invoke() → generates Signal (LONG/SHORT/NEUTRAL)
       └─> if signal != NEUTRAL:
           └─> OrderManager.execute_decision()
               ├─> PositionSizer.calculate_size() → qty
               ├─> RiskManager.validate_trade(symbol, qty, price) → (valid, reason)
               │   ├─ Check: Sufficient capital?
               │   ├─ Check: Position size within limits?
               │   └─ Check: Daily loss limit not exceeded?
               │
               │   ❌ NOT CHECKED: Does position already exist?
               │   ❌ NOT CHECKED: What's the current position direction?
               │
               ├─> Create Order object
               ├─> Broker.place_order() → filled_order
               ├─> PortfolioManager.execute_trade() → Trade
               └─> RiskManager.on_trade_executed() → update daily P&L
```

### Current Gap

**The agent analyzes every period and generates signals without considering:**
- Whether a position is already open
- Whether the signal matches the current position direction
- Whether to add to existing positions (pyramiding)
- Whether to hold existing positions when signal repeats

## Position Management Strategies

Different trading strategies require different position management approaches:

### 1. **One Position Only** (Conservative)
- **Rule**: Only one position per asset at a time
- **Behavior**:
  - LONG signal + no position → Open LONG
  - LONG signal + LONG position → **HOLD** (do nothing)
  - LONG signal + SHORT position → **Close SHORT** → Open LONG (reversal)
  - SHORT signal + LONG position → **Close LONG** → Open SHORT (reversal)

### 2. **Pyramiding** (Aggressive)
- **Rule**: Add to winning positions when signal continues
- **Behavior**:
  - LONG signal + no position → Open LONG
  - LONG signal + LONG position + profit > X% → **Add to LONG**
  - LONG signal + LONG position + profit < 0% → **HOLD**
  - SHORT signal + LONG position → Close LONG → Open SHORT

### 3. **Scale In** (Gradual Entry)
- **Rule**: Enter positions in multiple tranches
- **Behavior**:
  - LONG signal + no position → Open 33% of target size
  - LONG signal + partial LONG → Add 33% if signal strengthens
  - Stop adding after 3 tranches or 100% allocation

### 4. **Reversal Only** (Swing Trading)
- **Rule**: Only trade when signal reverses direction
- **Behavior**:
  - LONG signal + no position → Open LONG
  - LONG signal + LONG position → **HOLD**
  - SHORT signal + LONG position → Close LONG → Open SHORT
  - SHORT signal + SHORT position → **HOLD**

## Proposed Architecture

### Option A: Extend RiskManager (Recommended)

**Pros:**
- All risk/position logic in one place
- Simple to implement
- Consistent with current architecture

**Cons:**
- RiskManager becomes more complex
- Mixing pre-trade risk with position strategy

**Implementation:**

```python
class RiskManager:
    def __init__(
        self,
        # ... existing params ...
        position_strategy: str = "one_position_only",  # NEW
        allow_pyramiding: bool = False,  # NEW
        max_pyramid_layers: int = 3,  # NEW
    ):
        self.position_strategy = position_strategy
        self.allow_pyramiding = allow_pyramiding
        self.max_pyramid_layers = max_pyramid_layers

    def validate_trade(
        self,
        symbol: str,
        side: OrderSide,  # NEW: need to know direction
        qty: float,
        price: float
    ) -> Tuple[bool, str]:
        """Validate trade considering existing positions."""

        # NEW: Check position management strategy FIRST
        existing_position = self.portfolio.get_position(symbol)

        if existing_position and existing_position["qty"] != 0:
            validation = self._validate_position_strategy(
                symbol, side, existing_position
            )
            if not validation[0]:
                return validation

        # Existing validations (capital, size, daily loss)
        # ...

        return True, "Trade approved"

    def _validate_position_strategy(
        self,
        symbol: str,
        side: OrderSide,
        existing_position: Dict
    ) -> Tuple[bool, str]:
        """Check if new trade is allowed given existing position."""

        current_qty = existing_position["qty"]
        is_long_position = current_qty > 0
        is_short_position = current_qty < 0
        is_adding_to_long = is_long_position and side == OrderSide.BUY
        is_adding_to_short = is_short_position and side == OrderSide.SELL
        is_reversing = (is_long_position and side == OrderSide.SELL) or \
                      (is_short_position and side == OrderSide.BUY)

        # Strategy: One Position Only
        if self.position_strategy == "one_position_only":
            if is_adding_to_long or is_adding_to_short:
                return False, f"Position already exists for {symbol}, strategy does not allow adding"
            # Reversals are allowed (will close existing, open new)

        # Strategy: Pyramiding
        elif self.position_strategy == "pyramiding":
            if is_adding_to_long or is_adding_to_short:
                # Check if position is profitable
                if existing_position["pnl"] <= 0:
                    return False, f"Position not profitable, pyramiding not allowed"
                # Check max layers (would need tracking in Position)
                # ...

        # Strategy: Reversal Only
        elif self.position_strategy == "reversal_only":
            if is_adding_to_long or is_adding_to_short:
                return False, f"Position exists, waiting for reversal signal"

        return True, "Position strategy check passed"
```

**Usage:**

```python
# In backtest or strategy setup
risk_manager = RiskManager(
    initial_capital=100000,
    portfolio=portfolio,
    position_strategy="one_position_only",  # or "pyramiding", "reversal_only"
)

# In OrderManager.execute_decision()
is_valid, reason = self.risk_manager.validate_trade(
    symbol=symbol,
    side=side,  # Pass the side
    qty=qty,
    price=current_price
)
```

### Option B: Separate PositionPolicy Component

**Pros:**
- Clean separation of concerns
- Easier to test different strategies
- Can be swapped without touching RiskManager

**Cons:**
- More files, more complexity
- Need to wire into OrderManager

**Implementation:**

```python
# quantagent/trading/position_policy.py

class PositionPolicy:
    """Determines whether to execute a trade given existing positions."""

    def __init__(
        self,
        portfolio: PortfolioManager,
        strategy: str = "one_position_only"
    ):
        self.portfolio = portfolio
        self.strategy = strategy

    def should_execute(
        self,
        symbol: str,
        side: OrderSide
    ) -> Tuple[bool, str]:
        """Check if trade should be executed."""
        # Similar logic as Option A
        pass

# In OrderManager
class OrderManager:
    def __init__(
        self,
        position_sizer: PositionSizer,
        risk_manager: RiskManager,
        position_policy: PositionPolicy,  # NEW
        broker,
        portfolio_manager,
        db: Session,
    ):
        self.position_policy = position_policy
        # ...

    def execute_decision(self, ...):
        # Step 2.5: Check position policy BEFORE sizing
        should_execute, reason = self.position_policy.should_execute(symbol, side)
        if not should_execute:
            logger.info(f"{symbol}: {reason}")
            return None

        # Continue with sizing, validation, etc.
```

### Option C: Signal Filtering Layer

**Pros:**
- Filters signals before they reach OrderManager
- More efficient (doesn't calculate size for filtered signals)
- Matches mental model: "filter signals based on context"

**Cons:**
- Another layer of abstraction
- Signal and execution logic are separated

## Recommendation

**Use Option A (Extend RiskManager)** for initial implementation because:

1. **Simple**: Minimal code changes, everything in RiskManager
2. **Logical**: Risk management includes "risk of adding to positions"
3. **Flexible**: Easy to add new strategies via config
4. **Testable**: All position logic in one testable unit

**Future Evolution**: If strategies become very complex, extract to PositionPolicy (Option B).

## Implementation Plan

### Phase 1: Add to RiskManager

1. Add parameters:
   - `position_strategy: str` (default: "one_position_only")
   - `allow_pyramiding: bool`
   - `pyramid_profit_threshold_pct: float` (default: 2.0)

2. Modify `validate_trade()`:
   - Add `side: OrderSide` parameter
   - Check existing position BEFORE other validations
   - Return early if strategy doesn't allow trade

3. Update callers:
   - `OrderManager.execute_decision()` → pass `side` to `validate_trade()`

### Phase 2: Configuration

Add to backtest config:
```python
config = {
    "position_strategy": "one_position_only",  # or "pyramiding", "reversal_only"
    "allow_pyramiding": False,
    "pyramid_profit_threshold_pct": 2.0,
    # ... existing config ...
}
```

### Phase 3: Documentation & Testing

- Document each strategy behavior
- Add unit tests for each strategy
- Add integration test with backtest

## Examples

### Example 1: One Position Only (Conservative)

```python
# Config
config = {"position_strategy": "one_position_only"}

# Backtest behavior:
# Period 1: Signal=LONG  → Open LONG position
# Period 2: Signal=LONG  → SKIP (position exists)
# Period 3: Signal=LONG  → SKIP (position exists)
# Period 4: Signal=SHORT → Close LONG, Open SHORT
# Period 5: Signal=SHORT → SKIP (position exists)
```

### Example 2: Pyramiding (Aggressive)

```python
# Config
config = {
    "position_strategy": "pyramiding",
    "pyramid_profit_threshold_pct": 2.0,
    "max_pyramid_layers": 3
}

# Backtest behavior:
# Period 1: Signal=LONG, P&L=0%      → Open LONG (layer 1)
# Period 2: Signal=LONG, P&L=-1%     → SKIP (not profitable)
# Period 3: Signal=LONG, P&L=+3%     → Add to LONG (layer 2)
# Period 4: Signal=LONG, P&L=+5%     → Add to LONG (layer 3)
# Period 5: Signal=LONG, P&L=+7%     → SKIP (max layers reached)
# Period 6: Signal=SHORT             → Close all LONG, Open SHORT
```

## Migration Path

### Backward Compatibility

Default behavior is **unrestricted** (current behavior):
```python
position_strategy = "unrestricted"  # Allow all trades (existing behavior)
```

Users can opt-in to new strategies:
```python
position_strategy = "one_position_only"  # Conservative approach
```

### Rollout

1. **Week 1**: Implement in RiskManager with default="unrestricted"
2. **Week 2**: Test with "one_position_only" in backtests
3. **Week 3**: Document strategies, add to config examples
4. **Week 4**: Make "one_position_only" the recommended default

## Related Files

- **Implementation**: `quantagent/risk/manager.py` (or new `quantagent/trading/position_policy.py`)
- **Caller**: `quantagent/trading/order_manager.py`
- **Models**: `quantagent/models.py` (Position model already has needed fields)
- **Config**: Backtest config, StrategyAssembler
- **Tests**: `tests/test_position_management_strategies.py` (new)

## Questions for Discussion

1. **Default strategy**: Should we change default to "one_position_only" or keep "unrestricted"?
2. **CLOSE signal**: Should we add a fourth signal type "CLOSE" for explicit position closing?
3. **Partial closes**: Should we support "reduce position by 50%" signals?
4. **Time-based rules**: Should we add "hold position for min X periods" rules?
5. **Stop loss integration**: How does this interact with stop-loss orders?

## Next Steps

1. Get user confirmation on Option A vs B vs C
2. Implement chosen option
3. Add configuration support
4. Test with historical backtests
5. Document strategy behaviors
6. Add to user guide
