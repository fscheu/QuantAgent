# StrategyAssembler: Arquitectura y Uso (MVP)

Este documento describe el objetivo, decisiones de diseño e interfaz de uso del StrategyAssembler propuesto para QuantAgent. Su función es estandarizar cómo resolvemos una configuración de estrategia en un conjunto coherente de componentes (PortfolioManager, PositionSizer, RiskManager, Broker, OrderManager y TradingGraph) con metadatos y enlaces de procedencia consistentes para backtesting y paper trading.

## Objetivo
- Unificar la construcción de la “pila de trading” a partir de perfiles de estrategia (portfolio, risk, modelo) y overrides.
- Proveer un snapshot inmutable de configuración para reproducibilidad (persistible en BacktestRun.config_snapshot).
- Asegurar consistencia de parámetros entre Backtest, Paper y Replay (incluido environment, universo de instrumentos y metadata de modelo).
- Encapsular la integración con checkpointing/threads del TradingGraph sin duplicar lógica en múltiples lugares.

## Alcance (MVP)
- Resolución de perfiles:
  - Portfolio (incluye `universe`, `base_position_pct`, `slippage_pct`, `initial_cash` opcional).
  - Risk (`max_daily_loss_pct`, `max_position_pct`).
  - Modelo (`model_provider`, `model_name`, `temperature`, `use_checkpointing`).
- Ensamblado de componentes:
  - PortfolioManager, PositionSizer, RiskManager, PaperBroker, OrderManager.
  - TradingGraph configurado con los flags de modelo y checkpointing.
- Reglas de universo:
  - Si el usuario no especifica `assets` al lanzar un backtest, se usan los símbolos de `portfolio.universe`.
- Procedencia y environment:
  - Environment propagado a órdenes y señales (BACKTEST/PAPER).
  - thread_id/checkpoint_id gestionados por run/asset/fecha cuando `use_checkpointing` está activo.
- Artefactos:
  - Imágenes en disco (DB almacena rutas), evitando blobs grandes en checkpoints.

## Diseño de Implementación

Sugerimos ubicarlo en: `quantagent/strategy/assembler.py`

Tipos principales:
- ResolvedConfig (dataclass): snapshot inmutable de la estrategia resuelta.
  - Campos clave (MVP):
    - environment: Environment
    - universe: list[str]
    - initial_cash: float
    - base_position_pct: float
    - max_daily_loss_pct: float
    - max_position_pct: float
    - slippage_pct: float
    - model_provider: str
    - model_name: str
    - temperature: float
    - use_checkpointing: bool
    - extras: dict (reserva para flags específicos del agente/graph)

- TradingComponents (dataclass): instancias conectadas listas para operar.
  - portfolio_manager: PortfolioManager
  - position_sizer: PositionSizer
  - risk_manager: RiskManager
  - broker: PaperBroker (MVP)
  - order_manager: OrderManager
  - graph: TradingGraph

Funciones principales:
- from_profiles(portfolio_profile: dict, risk_profile: dict, model_profile: dict, overrides: dict | None) -> ResolvedConfig
  - Merge con prioridad: overrides > model/portfolio/risk > defaults.
  - Validaciones básicas (rangos, listas no vacías, etc.).
- resolve_universe(resolved: ResolvedConfig, assets_override: list[str] | None) -> list[str]
  - Retorna `assets_override` si se pasa; si no, `resolved.universe`.
- build_components(resolved: ResolvedConfig, db_session: Session) -> TradingComponents
  - Crea PM/PS/RM/Broker/OM respetando parámetros y environment.
  - Construye TradingGraph con flags de modelo y `use_checkpointing`.
- config_snapshot(resolved: ResolvedConfig) -> dict
  - Dict serializable para BacktestRun.config_snapshot y reproducibilidad.
- make_thread_id(run_id: int, symbol: str, ts: datetime) -> str
  - Utilidad para estandarizar thread_id en backtests con checkpointing.

Decisiones clave:
- Fuente de verdad de `universe` en MVP: perfil de Portfolio.
- Sin exposición sectorial en esta fase; listas fijas de símbolos.
- Replay sweeps secuenciales (sin concurrencia).
- Checkpointing opcional; cuando está activo se debe pasar `config={"configurable": {"thread_id": ...}}` al invocar el graph.

## Ejemplo de Uso

Backtesting (refactor sugerido en `quantagent/backtesting/backtest.py`):

```python
from quantagent.strategy.assembler import StrategyAssembler

# 1) Resolver configuración
resolved = StrategyAssembler.from_profiles(
    portfolio_profile={"universe": ["BTC", "ETH"], "base_position_pct": 0.05, "slippage_pct": 0.01, "initial_cash": 100000.0},
    risk_profile={"max_daily_loss_pct": 0.05, "max_position_pct": 0.10},
    model_profile={"model_provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.1, "use_checkpointing": True},
    overrides=None,
)
assets = StrategyAssembler.resolve_universe(resolved, assets_override=None)

# 2) Construir componentes
components = StrategyAssembler.build_components(resolved, db_session=db)

# 3) Crear snapshot para BacktestRun
snapshot = StrategyAssembler.config_snapshot(resolved)

# 4) Invocar TradingGraph por activo/fecha
thread_id = StrategyAssembler.make_thread_id(run_id, symbol, current_date)
result = components.graph.graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
```

Paper trading (scheduler/job):
- Reusar `from_profiles` + `build_components` con `environment=Environment.PAPER` y `use_checkpointing` según necesidad.
- Generar thread_id por símbolo/ventana temporal para trazabilidad.

## Interacción con Modelos y DB
- Modelos existentes (resumen relevante):
  - StrategyConfig: almacena perfiles (kind: portfolio/risk/combined) en `json_config`.
  - BacktestRun: persiste `config_snapshot` inmutable y métricas del run.
  - Signal/Order: campos `environment`, `thread_id`, `checkpoint_id`, `trigger_signal_id`, `state_snapshot`, metadatos de modelo.
- StrategyAssembler no define tablas nuevas; solo produce snapshots listos para persistir.
- La UI (Streamlit) debe:
  - Guardar/leer perfiles a través de StrategyConfig.
  - Al crear un BacktestRun, persistir `config_snapshot(resolved)` y la lista final de `assets`.

## Integración con la UI (MVP)
- Pestaña Configuración: crea/edita perfiles y muestra un “Resolved Snapshot Preview”.
- Pestaña Backtesting: al programar un run, si el usuario no pasa `assets`, usar `resolve_universe()`.
- Pestaña Analyses/Orders: mostrar enlaces de procedencia (orden ↔ señal, thread/checkpoint) y rutas de imágenes.
- Pestaña Replay: reconstruir ejecuciones secuenciales reutilizando señales/analyses previas con perfiles alternativos (mismo universo o variante), siempre registrando el snapshot usado.

## Validaciones y Defaults
- base_position_pct: (0, 0.10]; max_position_pct: (0, 0.20]; max_daily_loss_pct: (0, 0.20].
- temperature: [0, 1]; slippage_pct: [0, 0.05].
- initial_cash: > 0; universe: lista no vacía si no hay `assets_override`.
- Fallbacks razonables si faltan campos, con logging de advertencia.

## Evolución (post‑MVP)
- Brokers adicionales (exchange simulados o APIs) + enrutado por environment.
- Soporte de exposición por sector y reglas de composición de universo.
- Paralelismo controlado para replay sweeps.
- Plantillas de StrategyConfig y versionado semántico de snapshots.
- Métricas extendidas en BacktestRun y artefactos agregados (rutas y hashes).

## Beneficios
- Menos duplicación de lógica de ensamblado y menos errores de desalineación.
- Reproducibilidad: snapshot consistente en cada ejecución.
- Portabilidad: misma interfaz para backtest/paper con cambios mínimos.
- Observabilidad: procedencia y environment pasan por una misma ruta de construcción.

---
Notas de implementación:
- No modifica migraciones; usa tablas existentes (ver `docs/MIGRATIONS.md`).
- Mantener parámetros compatibles con `quantagent/backtesting/backtest.py` y `quantagent/trading/*`.
- Mantener `universe` en el perfil de Portfolio durante el MVP.

