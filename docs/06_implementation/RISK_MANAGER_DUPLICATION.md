# RiskManager Duplication Analysis

**Date:** 2026-01-02
**Issue:** Two RiskManager classes exist in the codebase
**Status:** Implemented

## Problem Statement

There are TWO separate `RiskManager` classes in the codebase:

1. **`quantagent/risk/manager.py`** (228 lines)
2. **`quantagent/trading/risk_manager.py`** (201 lines)

This creates confusion, maintenance burden, and risk of inconsistent behavior.

## Usage Analysis

### quantagent/risk/manager.py
**Used by:**
- `tests/test_risk_manager.py` (tests only)

**NOT used in production code**

### quantagent/trading/risk_manager.py (OFFICIAL)
**Used by:**
- `quantagent/backtesting/backtest.py` ✅
- `quantagent/strategy/assembler.py` ✅ (builds all trading components)
- `tests/test_trading_components.py` ✅

**This is the actively used RiskManager in production**

## Key Differences

### 1. Constructor Parameters

**risk/manager.py:**
```python
def __init__(
    self,
    initial_capital: float,
    portfolio: PortfolioManager,
    max_position_size_pct: float = 2.0,      # 2% default
    max_daily_loss_pct: float = 5.0,         # 5% default
    environment: Environment = Environment.PAPER,
    db: Optional[Session] = None,
)
```

**trading/risk_manager.py:**
```python
def __init__(
    self,
    portfolio_manager,  # PortfolioManager instance
    max_daily_loss_pct: float = 0.05,        # 0.05 (5%) default
    max_position_pct: float = 0.10,          # 0.10 (10%) default
    db: Optional[Session] = None,
)
```

**Differences:**
- `risk/manager.py` takes `initial_capital` separately
- `risk/manager.py` has `environment` parameter
- `risk/manager.py` uses percentage values (2.0 = 2%)
- `trading/risk_manager.py` uses decimal values (0.05 = 5%)
- Parameter names differ: `max_position_size_pct` vs `max_position_pct`

### 2. Daily Loss Tracking

**risk/manager.py:**
- Queries database for trades with `Trade.closed_at >= today`
- Uses `Trade.environment` filter
- Method: `_get_daily_loss()`

**trading/risk_manager.py:**
- Has in-memory daily tracker: `self.daily_pnl_tracker: Dict[date, float]`
- Queries database OR uses in-memory tracker
- Method: `get_daily_pnl()`
- Has explicit `reset_daily_tracker()` method

### 3. Circuit Breaker

**risk/manager.py:**
- Boolean flag: `self.circuit_breaker_active`
- Method: `check_circuit_breaker()` - checks and activates
- Method: `reset_circuit_breaker()` - resets flag

**trading/risk_manager.py:**
- Boolean flag: `self.circuit_breaker_triggered`
- Method: `check_circuit_breaker()` - only checks status
- Circuit breaker activated in `on_trade_executed()` when daily loss exceeded
- Reset via `reset_daily_tracker()`

### 4. Validation Logic (AFTER our changes)

Both now have the same Check 5 logic (position management), but:

**risk/manager.py:**
- Check order: Capital → Position size → Daily loss → **Position management**
- Returns `Tuple[bool, str]`

**trading/risk_manager.py:**
- Check order: Capital → Position size → Daily loss → Circuit breaker → **Position management**
- Returns `Tuple[bool, Optional[str]]` (slightly different type hint)

### 5. Helper Methods

**risk/manager.py has:**
- `load_profile()` - load risk profile configuration
- `get_max_position_size()` - calculate max position value
- `get_max_daily_loss()` - calculate max daily loss value
- `get_daily_loss()` - wrapper for `_get_daily_loss()`

**trading/risk_manager.py has:**
- `reset_daily_tracker()` - reset daily P&L tracking
- Simpler API, fewer convenience methods

## Recommendation

**Use `quantagent/trading/risk_manager.py` as the OFFICIAL RiskManager**

**Reasons:**
1. ✅ Used in production code (backtest, assembler)
2. ✅ Simpler API with fewer moving parts
3. ✅ In-memory daily tracking is more efficient for backtesting
4. ✅ Follows cleaner separation: trading components in `trading/`
5. ✅ Already integrated with StrategyAssembler

**Deprecate `quantagent/risk/manager.py`**

**Reasons:**
1. ❌ Only used in one test file
2. ❌ More complex with `environment` and `initial_capital` tracking
3. ❌ In wrong directory (`risk/` instead of `trading/`)
4. ❌ Creates confusion and maintenance burden

## Migration Plan

### Phase 1: Verify Functionality
1. Review `tests/test_risk_manager.py` to understand what it tests
2. Port any missing test cases to `tests/test_trading_components.py`
3. Ensure `trading/risk_manager.py` has feature parity

### Phase 2: Update References
1. Update `tests/test_risk_manager.py`:
   ```python
   # OLD
   from quantagent.risk.manager import RiskManager

   # NEW
   from quantagent.trading.risk_manager import RiskManager
   ```

### Phase 3: Delete Deprecated File
1. Delete `quantagent/risk/manager.py`
2. Remove empty `quantagent/risk/` directory if no other files
3. Update documentation references

### Phase 4: Consolidate Best Features
If `risk/manager.py` has useful features not in `trading/risk_manager.py`:
- `load_profile()` - Consider adding to trading version
- `get_max_position_size()` / `get_max_daily_loss()` - Useful helpers, consider porting

## Short-term Workaround (Current State)

Since we just modified BOTH files to add Check 5, the behavior is now **inconsistent**:
- Both have the same position management logic ✅
- But they're called by different parts of the system
- Tests might pass with one but fail with another

**Immediate action needed:**
1. Decide which RiskManager to keep
2. Update all imports to use single source of truth
3. Delete the deprecated one

## Questions to Answer

1. **Are there features in `risk/manager.py` that we need?**
   - `load_profile()` seems useful for different risk profiles
   - `environment` tracking may be needed for multi-env scenarios

2. **Should we merge best features before deprecating?**
   - Port useful methods from `risk/` to `trading/`
   - Ensure no functionality loss

3. **Are there other duplicated components?**
   - Should audit entire codebase for similar issues

## Proposed Resolution

**Option A: Quick Fix (Recommended for MVP)**
1. Keep `trading/risk_manager.py` as-is
2. Delete `quantagent/risk/manager.py`
3. Update test file to import from `trading/`
4. Move on with single source of truth

**Option B: Feature Merge (Better long-term)**
1. Audit both classes for unique features
2. Port useful methods to `trading/risk_manager.py`
3. Update all imports
4. Delete `risk/manager.py`
5. Comprehensive testing

**Option C: Proper Refactor (Post-MVP)**
1. Create unified `RiskManager` with best of both
2. Support both percentage styles (2.0 vs 0.05) via config
3. Clean architecture with clear responsibilities
4. Complete test coverage

## Timeline

**Immediate (Today):**
- Document the issue ✅ (this file)
- Decide on approach

**Short-term (This week):**
- Implement chosen option (A or B)
- Ensure single RiskManager across codebase

**Long-term (Post-MVP):**
- Consider Option C if needed
- Add to architecture review backlog

## Related Files

- **Production code**: Uses `quantagent/trading/risk_manager.py`
- **Test file**: `tests/test_risk_manager.py` (uses old one)
- **Integration**: `quantagent/strategy/assembler.py` (builds trading version)
- **This analysis**: `docs/03_technical/RISK_MANAGER_DUPLICATION.md`

## Impact of Current Duplication

**Low risk but needs fixing:**
- Both implementations now have same Check 5 logic (we updated both)
- Production code uses the right one (`trading/risk_manager.py`)
- Only risk is test coverage gaps or future divergence

**But maintaining two is problematic:**
- Future changes must be applied twice
- Easy to forget to update both
- Confusing for new developers
- Wastes time and increases bug risk
