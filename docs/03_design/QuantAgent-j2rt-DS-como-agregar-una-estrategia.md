# Cómo agregar una estrategia (QuantAgent-j2rt)

Agregar una estrategia son dos cosas: un archivo en `quantagent/strategy/` y una entrada en `quantagent/strategy/registry.py`. El CLI (`backtest run|verify`) y `scripts/bt_adapter.py` leen los alias de ese registro (`cli_aliases()`); no hay otra lista que tocar. No se toca `quantagent/backtesting/` ni `quantagent/cli/`.

## Pasos
1. Crear `quantagent/strategy/<nombre>_strategy.py` con una subclase de `TradingStrategy`.
2. Importarla en `registry.py` y agregar una entrada en `STRATEGY_REGISTRY` con `"cli_alias": "<nombre-corto>"` (sin `cli_alias` la estrategia queda fuera del CLI, como `LLMAgentStrategy`, que necesita credenciales). Fuera del módulo, `register_strategy(Clase, min_bars=..., cli_alias=...)` arma la misma entrada en una línea.
3. Agregar `tests/test_<nombre>_strategy.py` (ver abajo).

## Contrato de `TradingStrategy` (`quantagent/strategy/base.py`)
| Miembro | Obligatorio | Qué tiene que cumplir |
|---|---|---|
| `generate_signal(kline_data, symbol, timeframe, current_price)` | sí | Devuelve `TradingSignal` (`decision` LONG/SHORT/HOLD, `confidence` en 0–1, `stop_loss`, `take_profit`) o `None` si no hay acción. `kline_data` es una lista de dicts con `open, high, low, close, volume, timestamp`, la última vela es la actual. Solo puede usar velas hasta esa. |
| `should_reevaluate(position, current_price)` | sí | `False` salvo política `REEVALUATE`. |
| `required_history_bars` (propiedad) | sí en la práctica | Cuántas velas necesita. Default 30: si necesita más y no lo declara, el motor le pasa 30 y da 0 trades. |
| `should_exit(position, current_price, ohlc_data)` | no | Default: stop loss, take profit y trailing stop porcentuales. Se sobreescribe si la salida la decide la regla. Devuelve `(salir, motivo)`. |
| `describe()` (classmethod) | no | `display_name`, `type` (`deterministic`/`llm`) y `description` que muestran el registro y la UI. |
| `__init__` | no | Los parámetros con default y validados (`ValueError` si no tienen sentido); el registro los declara en `params`. |
La entrada del registro pide `cls`, `type`, `display_name`, `description`, `min_bars` (igual a `required_history_bars`) y `params`.
## Tests mínimos
- La señal de entrada con una serie armada a mano.
- La señal de salida (o la salida por `should_exit`).
- Sin historia suficiente: devuelve `None` y no rompe.
- Un backtest corto de punta a punta que produce al menos un trade completo.
- Que el alias figura en `cli_aliases()` y `build_strategy(<nombre de clase>)` devuelve la clase.
Sin red ni datos reales. Ejemplos: `tests/test_sma_cross_strategy.py`, `tests/test_momentum_12m_strategy.py`. Lo que garantiza que registrar alcanza: `tests/test_strategy_registration.py`.
