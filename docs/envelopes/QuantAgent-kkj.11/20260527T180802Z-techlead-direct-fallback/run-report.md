# QuantAgent-kkj.11 — Tech Lead direct fallback

- **Run ID:** `20260527T180802Z-techlead-direct-fallback`
- **Estado:** `PARTIAL`
- **Worktree:** `/tmp/autodev-worktrees/preflight-20260527T174909Z/QuantAgent-kkj.11/implementer`
- **Executor intentado primero:** `auto -> codex`
- **Falla del executor:** `EXECUTOR_RUNTIME_ERROR / workspace_path_not_visible_to_executor`

## Qué pasó

El router ejecutó `auto` y terminó en Codex. El artifact canónico del executor reportó bloqueo porque el workspace declarado no era visible desde el executor, aunque el worktree existía en el host. O sea: hermoso éxito de transporte, fase realmente bloqueada. La clásica.

Artifacts del intento bloqueado:
- `docs/envelopes/QuantAgent-kkj.11/20260527T174919Z-QuantAgent-kkj.11-implementer/result.json`
- `docs/envelopes/QuantAgent-kkj.11/20260527T174919Z-QuantAgent-kkj.11-implementer/executor-stdout-codex.log`

## Fallback ejecutado

Se hizo implementación directa en el worktree ya preparado para no perder el turno:
- se creó `quantagent/llm/{registry,roles,routing}.py`
- `TradingGraph` ahora acepta `routing_policy` y resuelve roles `lite` / `deep_reasoning`
- `StrategyAssembler` preserva `routing_policy` desde snapshot y lo pasa a `TradingGraph`
- se agregó `tests/test_provider_routing.py` para registry, roles, policy, wiring de graph y assembler

## Verificación

Comandos corridos:
- `/home/azureuser/.hermes/hermes-agent/venv/bin/python -m ruff check --fix quantagent/llm quantagent/trading_graph.py quantagent/strategy/assembler.py tests/test_provider_routing.py`
- `/home/azureuser/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_provider_routing.py -v`
- `/home/azureuser/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_strategy_assembler.py -q`
- `/home/azureuser/.hermes/hermes-agent/venv/bin/python -m compileall -q quantagent tests`

Resultado:
- `test_provider_routing.py`: 8/8 PASS
- `test_strategy_assembler.py`: 2/2 PASS
- `compileall`: PASS

## Límite actual

El ticket no está listo para integración todavía. Quedan las fases posteriores del plan de `QuantAgent-kkj.11` (persistencia/snapshots completos, wiring adicional y validación más amplia).

## Next

Seguir sobre la misma branch con el próximo slice implementable del ticket, o dividir el remanente en un follow-up si conviene reducir scope antes de tester/integration.
