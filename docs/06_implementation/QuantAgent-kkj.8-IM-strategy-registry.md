# QuantAgent-kkj.8 — Implementation Notes

- Added `quantagent.strategy.registry` as the central source of truth for the four supported strategies, including type, display metadata, minimum bar requirements, and configurable constructor params.
- Updated `TradingScheduler` to accept any `TradingStrategy` instance while preserving `LLMAgentStrategy` as the default when no strategy is injected.
- Fixed the scheduler/runtime mismatch where deterministic strategies failed because `_process_asset()` always passed `thread_id`; the scheduler now passes that kwarg only when the selected strategy supports it.
