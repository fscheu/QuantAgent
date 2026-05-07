# Tech Lead direct correction note

- Ticket: `QuantAgent-3hs`
- Mode: `correction`
- Executor routing was not used for live execution; the fix was a bounded test-fixture correction applied directly in an isolated worktree.
- Root cause: some tests reload `quantagent.trading_graph`, leaving previously imported `TradingGraph` aliases in test modules pointing to a stale class with the original `_create_llm` implementation. That stale class instantiated a real `ChatOpenAI`, which later failed at `indicator_agent.py:76` because the runtime object did not expose `bind_tools()`.
- Change applied: update `tests/conftest.py` so the autouse fixture both patches `quantagent.trading_graph.TradingGraph._create_llm` and rebinds test-module `TradingGraph` aliases to the current module class after reloads.
- Scope boundary respected: test-only change; no production files modified.
