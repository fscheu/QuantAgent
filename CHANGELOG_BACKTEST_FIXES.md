# Backtesting Integration Fixes

## Issues Fixed

### 1. Order Provenance and Environment in OrderManager

**Problem**: `Backtest` was attempting to pass `environment` and `trigger_signal_id` to `OrderManager.execute_decision()`, but the method signature didn't accept these parameters.

**Solution**: Extended `OrderManager.execute_decision()` signature to accept optional `environment` and `trigger_signal_id` parameters.

**Changes**:
- File: `quantagent/trading/order_manager.py`
- Line 54-62: Added parameters to method signature
  - `environment=None` (Environment enum: BACKTEST, PAPER, PROD)
  - `trigger_signal_id: Optional[int] = None` (Signal ID that triggered the order)
- Line 118-126: Set these fields when creating Order object

**Impact**:
- ✅ Backtest can now properly tag orders with environment
- ✅ Full provenance tracking: Order → Signal linkage
- ✅ Backward compatible (parameters are optional, default to None)
- ✅ Existing tests continue to work

**Usage**:
```python
# Backtest usage
order = order_manager.execute_decision(
    symbol='BTC',
    decision='LONG',
    confidence=0.8,
    current_price=42000.0,
    environment=Environment.BACKTEST,  # NEW
    trigger_signal_id=signal.id  # NEW
)

# Paper/Live usage (backward compatible)
order = order_manager.execute_decision(
    symbol='BTC',
    decision='LONG',
    confidence=0.8,
    current_price=42000.0
    # environment and trigger_signal_id default to None
)
```

### 2. RiskManager Daily Reset Method Name

**Problem**: `Backtest` was calling `risk_manager.reset_daily_pnl()`, but the actual method name is `reset_daily_tracker()`.

**Solution**: Updated `Backtest` to call the correct method name.

**Changes**:
- File: `quantagent/backtesting/backtest.py`
- Line 160: Changed `self.risk_manager.reset_daily_pnl()` to `self.risk_manager.reset_daily_tracker()`

**Impact**:
- ✅ Daily P&L tracking now works correctly in backtests
- ✅ Circuit breaker resets properly at start of each day
- ✅ Method name aligns with implementation in `risk_manager.py:164`

## Verification

### Database Schema Alignment

Both fields already exist in the Order model:

```python
# quantagent/models.py:74-75
environment = Column(Enum(Environment), nullable=False, default=Environment.PAPER)
trigger_signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)
```

No migration required - schema already supports these fields.

### Test Compatibility

All existing tests remain compatible:
- `tests/test_trading_components.py` - All execute_decision() calls work (optional params)
- `tests/test_backtest.py` - Now calls correct reset method
- `tests/test_backtest_integration.py` - Full provenance tracking validated

## Benefits

1. **Full Provenance Tracking**
   - Every order knows which signal triggered it
   - Can trace: Analysis → Signal → Order → Trade

2. **Environment Separation**
   - Backtest orders tagged with `Environment.BACKTEST`
   - Paper orders tagged with `Environment.PAPER`
   - Prod orders tagged with `Environment.PROD`
   - Clean data separation in queries

3. **Reproducibility**
   - Given a backtest run, can retrieve all orders and their triggering signals
   - Full audit trail for compliance and debugging

4. **Backward Compatibility**
   - Existing code continues to work
   - New parameters are optional
   - No breaking changes

## Related Documentation

- **Technical Spec**: `docs/03_technical/backtesting_engine.md`
- **Requirements**: `docs/01_requirements/trading_system_requirements.md` (Section B: Analysis Provenance)
- **Models**: `quantagent/models.py` (Order, Signal schemas)

---

**Date**: 2025-11-27
**Version**: Phase 1, Week 7-8 (Backtesting Engine)
