# QuantAgent-3o4 - Planning: TradingScheduler Implementation

**Issue**: QuantAgent-3o4  
**Related**: 
- `docs/01_requirements/QuantAgent-3o4-RQ-trading-scheduler.md`
- `docs/03_design/QuantAgent-3o4-DS-trading-scheduler.md`
- `docs/05_acceptance_tests/QuantAgent-3o4-AC-trading-scheduler.md`

---

## Summary

Implement TradingScheduler using APScheduler to enable automatic paper trading. This is a **TIER 2 ESSENTIAL** requirement (2.2) and an **MVP blocker**.

**Estimated Effort**: 1-2 days  
**Complexity**: Low (straightforward integration of existing components)

---

## Tasks

### Task 1: Add APScheduler Dependency
**Effort**: 15 minutes  
**Dependencies**: None

**Steps**:
1. Add `APScheduler>=3.10.0,<4.0.0` to `pyproject.toml` dependencies
2. Run `pip install -e .` to install
3. Verify import: `python -c "from apscheduler.schedulers.background import BackgroundScheduler"`

**Acceptance**:
- ✅ APScheduler installs without errors
- ✅ Import succeeds

---

### Task 2: Add Scheduler Configuration to Settings
**Effort**: 30 minutes  
**Dependencies**: Task 1

**Steps**:
1. Open `quantagent/settings.py`
2. Add `SchedulerSettings` dataclass:
   ```python
   @dataclass
   class SchedulerSettings:
       enabled: bool = False
       interval_hours: float = 1.0
       assets: List[str] = field(default_factory=lambda: ["BTC", "SPX"])
       environment: str = 'paper'
   ```
3. Add `scheduler: SchedulerSettings` field to main `Settings` class
4. Add validation in `__post_init__` (interval > 0, assets not empty)

**Acceptance**:
- ✅ Settings class has `scheduler` attribute
- ✅ Default values match specification
- ✅ Validation raises errors for invalid config

---

### Task 3: Implement TradingScheduler Class
**Effort**: 2-3 hours  
**Dependencies**: Task 1, Task 2

**Steps**:
1. Create `quantagent/trading/scheduler.py`
2. Implement `TradingScheduler` class per design spec:
   - `__init__()`: Accept dependencies (TradingGraph, OrderManager, DataProvider, config)
   - `_validate_config()`: Validate interval and assets list
   - `start()`: Initialize and start APScheduler
   - `stop()`: Gracefully shut down scheduler
   - `analyze_and_trade()`: Main scheduled job (iterates assets)
   - `_process_asset()`: Process single asset (fetch → analyze → execute)
3. Add structured logging (JSON-compatible format)
4. Implement error handling:
   - Configuration errors → raise exception (fatal)
   - Transient errors → log and continue (non-fatal)
   - Logic errors → log warning and skip

**Acceptance**:
- ✅ All methods implemented per design
- ✅ Error handling covers all scenarios
- ✅ Logs structured and parseable

**Reference**: `docs/03_design/QuantAgent-3o4-DS-trading-scheduler.md` (full class structure)

---

### Task 4: Write Unit Tests
**Effort**: 2-3 hours  
**Dependencies**: Task 3

**Steps**:
1. Create `tests/trading/test_scheduler.py`
2. Implement test cases:
   - `test_scheduler_start()` - verify APScheduler starts
   - `test_scheduler_stop()` - verify graceful shutdown
   - `test_analyze_and_trade_long_signal()` - mock LONG decision, verify execution
   - `test_analyze_and_trade_short_signal()` - mock SHORT decision, verify execution
   - `test_analyze_and_trade_hold_signal()` - mock HOLD, verify no execution
   - `test_error_handling_transient()` - mock API error, verify continues
   - `test_config_validation_invalid_interval()` - verify raises ValueError
   - `test_config_validation_empty_assets()` - verify raises ValueError
   - `test_environment_tagging()` - verify records tagged as 'paper'
3. Use mocks for TradingGraph, OrderManager, DataProvider
4. Achieve ≥ 70% code coverage

**Acceptance**:
- ✅ All tests pass
- ✅ Coverage ≥ 70%
- ✅ Tests cover happy path + error cases

**Reference**: `docs/03_design/QuantAgent-3o4-DS-trading-scheduler.md` (test strategy section)

---

### Task 5: Create Entry Point Script
**Effort**: 1 hour  
**Dependencies**: Task 3

**Steps**:
1. Create `apps/paper_trading.py` (or add to existing app)
2. Implement:
   - Initialize all dependencies (TradingGraph, OrderManager, DataProvider)
   - Create TradingScheduler instance
   - Start scheduler
   - Add signal handlers (SIGTERM, SIGINT) for graceful shutdown
   - Keep process alive (while loop)
3. Add command-line arguments (optional):
   - `--interval` - override interval
   - `--assets` - override asset list
   - `--config` - path to config file

**Acceptance**:
- ✅ Script runs without errors
- ✅ Scheduler starts and executes analysis
- ✅ Ctrl+C gracefully shuts down
- ✅ Logs all activities

---

### Task 6: Integration Testing
**Effort**: 2 hours  
**Dependencies**: Task 3, Task 4, Task 5

**Steps**:
1. Create `tests/integration/test_scheduler_integration.py`
2. Implement end-to-end test:
   - Use test database
   - Mock external APIs (yfinance)
   - Set interval to 1 minute (fast testing)
   - Run for 3 cycles
   - Verify:
     - 3 analysis runs completed
     - Orders created with `environment='paper'`
     - Database records correct
     - Logs present
3. Run test and fix any issues

**Acceptance**:
- ✅ Integration test passes
- ✅ All components work together
- ✅ Environment tagging verified in DB

---

### Task 7: Stability Testing (24h Test)
**Effort**: 1 hour (setup) + 24h (wait)  
**Dependencies**: Task 6

**Steps**:
1. Configure scheduler with 5-minute interval (faster than 1 hour for testing)
2. Use 2 assets: ["BTC", "SPX"]
3. Start scheduler in background
4. Monitor:
   - Memory usage (baseline, 12h, 24h)
   - CPU usage
   - Log files
   - Database growth
5. After 24h:
   - Verify no crashes
   - Check memory growth < 20%
   - Count analysis runs (expected: ~288 per asset = 2 assets × 12 runs/hour × 24 hours)
   - Verify success rate ≥ 95%

**Acceptance**:
- ✅ Uptime > 99%
- ✅ No crashes or unhandled exceptions
- ✅ Memory stable (< 20% growth)
- ✅ All runs logged

**Reference**: `docs/05_acceptance_tests/QuantAgent-3o4-AC-trading-scheduler.md` (AC-10)

---

### Task 8: Documentation
**Effort**: 1 hour  
**Dependencies**: Task 7

**Steps**:
1. Create or update `README.md` section: "Running Paper Trading Scheduler"
2. Document:
   - Installation: `pip install -e .`
   - Configuration: `config.yml` or environment variables
   - Starting: `python apps/paper_trading.py`
   - Stopping: Ctrl+C or `kill -TERM <pid>`
   - Monitoring: log file locations, database queries
3. Add troubleshooting section:
   - "Scheduler not starting" → check config validation
   - "No trades executing" → check logs for HOLD decisions or risk rejections
   - "High memory usage" → check log rotation settings

**Acceptance**:
- ✅ Documentation complete and clear
- ✅ Colleague can run scheduler from docs alone

---

## Task Dependencies

```
Task 1 (Dependency)
   └─→ Task 2 (Settings)
          └─→ Task 3 (Implementation)
                 ├─→ Task 4 (Unit Tests)
                 └─→ Task 5 (Entry Point)
                        └─→ Task 6 (Integration Test)
                               └─→ Task 7 (Stability Test)
                                      └─→ Task 8 (Documentation)
```

**Critical Path**: Task 1 → 2 → 3 → 6 → 7 → 8 (2 days + 24h test)

---

## Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| APScheduler not working as expected | High | Low | Extensive testing in Task 4; APScheduler is mature library |
| Memory leaks during 24h test | Medium | Medium | Monitor memory in Task 7; add log rotation if needed |
| OrderManager/TradingGraph failures | High | Low | Already tested components; errors caught and logged |
| Configuration errors in production | Medium | Medium | Strict validation in Task 2; clear error messages |

---

## Testing Strategy

### Unit Tests (Task 4)
- **Scope**: TradingScheduler class in isolation
- **Mocks**: All dependencies (TradingGraph, OrderManager, DataProvider)
- **Coverage**: ≥ 70%

### Integration Tests (Task 6)
- **Scope**: End-to-end flow (scheduler → analysis → execution → database)
- **Mocks**: External APIs only (yfinance)
- **Real**: Database, OrderManager, TradingGraph

### Stability Tests (Task 7)
- **Scope**: 24h continuous operation
- **Environment**: Test environment with real components
- **Metrics**: Uptime, memory, success rate

---

## Rollout Plan

### Phase 1: Development (Tasks 1-4)
- Implement and unit test TradingScheduler
- Verify all unit tests pass
- Code review

### Phase 2: Integration (Tasks 5-6)
- Create entry point
- Run integration tests
- Fix any issues

### Phase 3: Validation (Task 7)
- Run 24h stability test
- Monitor and measure
- Verify acceptance criteria met

### Phase 4: Documentation & Handoff (Task 8)
- Complete documentation
- Demo to team
- Mark issue as complete

---

## Definition of Done

- ✅ All tasks completed (1-8)
- ✅ All unit tests passing (≥ 70% coverage)
- ✅ Integration test passing
- ✅ 24h stability test passing (> 99% uptime, < 20% memory growth)
- ✅ Code reviewed and merged
- ✅ Documentation complete
- ✅ Issue QuantAgent-3o4 closed in Beads

**Verification Commands**:
```bash
# Unit tests
pytest tests/trading/test_scheduler.py -v --cov=quantagent/trading/scheduler

# Integration test
pytest tests/integration/test_scheduler_integration.py -v

# Start scheduler (manual verification)
python apps/paper_trading.py

# Check logs
tail -f logs/scheduler.log

# Verify database records
psql -d quantagent -c "SELECT COUNT(*) FROM signals WHERE environment='paper';"
```

---

## Open Questions

None - all requirements and design are clear.

---

## Next Steps After Completion

After QuantAgent-3o4 is complete and validated:

1. **Monitor production usage** (first week):
   - Review logs daily
   - Check for errors or anomalies
   - Measure success rate and latency

2. **Optimization opportunities** (future enhancements, NOT in scope):
   - Parallel asset processing (if > 5 assets)
   - Adaptive interval (reduce frequency during off-hours)
   - Advanced scheduling (market hours only)
   - Multiple strategies per asset

3. **Phase 2 integration**:
   - Human-in-the-loop approval before real broker trades
   - Real-time dashboard monitoring
   - Alert system for errors/circuit breakers
