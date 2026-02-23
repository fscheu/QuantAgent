# MVP Manual Test Cases - Streamlit App & Backtesting

**Objetivo**: Validar el circuito completo del sistema antes de implementar el scheduler automático.

**Duración estimada**: 2-3 horas para los 3 casos

**Prerequisitos**:
- PostgreSQL corriendo (`docker-compose up -d db`)
- DATABASE_URL configurada en `.env`
- API keys configuradas (OPENAI_API_KEY o ANTHROPIC_API_KEY)
- Migraciones aplicadas (`python -m alembic upgrade head`)

---

## Caso de Test 1: Configuración Base y Backtest Simple (30-40 min)

### Objetivo
Validar que:
- La configuración de perfiles funciona correctamente
- El backtest engine ejecuta análisis y genera métricas
- Los datos se persisten en la base de datos
- La UI muestra resultados correctamente

### Pasos Detallados

#### 1.1 Iniciar la aplicación (5 min)

```bash
# Terminal 1: Verificar que PostgreSQL está corriendo
docker-compose ps

# Terminal 2: Iniciar Streamlit
streamlit run apps/streamlit/app.py
```

**Verificar**:
- [x] La app abre en http://localhost:8501
- [x] No hay errores de conexión a DB en la UI
- [x] Se muestran las 7 pestañas: Dashboard, Configuration, Analyses, Backtesting, Replay, Orders & Positions, Logs

**Troubleshooting**:
- Si error de DB: verificar `docker-compose ps` y DATABASE_URL en .env
- Si error de importación: `pip install -r requirements.txt`

---

#### 1.2 Crear perfil de portfolio (5 min)

**En la pestaña "Configuration"**:

1. Seleccionar `Profile kind`: **combined**
2. En `Profile name`: **test-mvp-profile**
3. En `Universe`: Seleccionar **BTC** y **SPX** (multi-select)
4. En `Profile JSON`, editar:
```json
{
  "universe": ["BTC", "SPX"],
  "base_position_pct": 0.05,
  "max_position_pct": 0.10,
  "max_daily_loss_pct": 0.05,
  "slippage_pct": 0.01
}
```
5. Click **Save profile**

**Verificar**:
- [x] Mensaje de éxito: "Saved combined profile 'test-mvp-profile' to database."
- [x] El perfil aparece en la tabla de **Profiles** con:
  - source: `db`
  - kind: `combined`
  - name: `test-mvp-profile`
  - version: `1`

**Troubleshooting**:
- Si error de JSON: validar formato en https://jsonlint.com
- Si no aparece en tabla: refrescar página (F5)

---

#### 1.3 Crear preset de modelo (3 min)

**En la pestaña "Configuration" (columna derecha)**:

1. En `Provider`: Seleccionar **openai**
2. En `Model name`: **gpt-4o-mini** (o tu modelo preferido)
3. En `Temperature`: **0.1**
4. En `Save as (name)`: **test-mvp-model**
5. Click **Save preset**

**Verificar**:
- [x] Mensaje de éxito: "Saved preset 'test-mvp-model'."
- [x] El preset aparece en **Presets preview** con los valores correctos

---

#### 1.4 Crear backtest run (5 min)

**En la pestaña "Backtesting"**:

1. **Assets**: Dejar vacío (usará Universe del perfil)
2. **Timeframe**: **4h**
3. **Start date**: 30 días atrás desde hoy
4. **End date**: Hoy
5. **Model preset**: **test-mvp-model**
6. **Portfolio profile**: **test-mvp-profile**
7. **Mode**: **Generate + Execute**
8. **Artifacts saving**: **path-only**
9. Click **Create run**

**Verificar**:
- [x] Mensaje: "Run X created. Backend execution wiring pending."
- [x] El run aparece en la tabla con:
  - status: `pending`
  - assets: `['BTC', 'SPX']` (heredado del Universe)
  - timeframe: `4h`
  - profile: `test-mvp-profile`

**Notas**:
- El run se crea pero **no se ejecuta automáticamente** (backend wiring pendiente)
- Esto es esperado en el MVP actual

---

#### 1.5 Ejecutar backtest manualmente via script (15-20 min)

**En Terminal 3**:

```bash
# Verificar que el script existe
ls examples/run_backtest.py

# Editar el script para usar nuestro perfil (opcional)
# O ejecutar directamente con config hardcodeada
python examples/run_backtest.py
```

**Qué esperar**:
```
Running backtest from 2025-XX-XX to 2025-XX-XX
Assets: ['BTC', 'SPX']
Timeframe: 4h
Initial capital: $100,000.00
------------------------------------------------------------
Fetching data for BTC...
Analyzing BTC at 2025-XX-XX XX:00...
  Decision: LONG, Confidence: 0.78
  Trade executed: BUY 0.05 BTC @ $45,000
...
[Múltiples análisis...]
...
============================================================
BACKTEST RESULTS
============================================================
Total Trades:      12
Winning Trades:    7
Losing Trades:     5
Win Rate:          58.33%
Profit Factor:     1.45
Sharpe Ratio:      0.85
Max Drawdown:      -8.23%
Total P&L:         $2,345.67
Total Return:      2.35%
...
```

**Verificar**:
- [ ] El script ejecuta sin errores
- [ ] Se muestran análisis para cada fecha/asset
- [ ] Se generan métricas al final
- [ ] Total Trades > 0
- [ ] Win Rate está entre 0-100%

**Troubleshooting**:
- Si error "No module named quantagent": `pip install -e .`
- Si error de API key: verificar `.env` o variables de entorno
- Si error "Rate limit": esperar 1 min y reintentar
- Si tarda mucho (>20 min): reducir rango de fechas a 7-15 días

---

#### 1.6 Verificar persistencia en base de datos (5 min)

**En Terminal 4 (verificación directa en DB)**:

```bash
docker-compose exec db psql -U postgres -d quantagent_dev
```

**Ejecutar queries**:

```sql
-- 1. Ver backtest_runs creados
SELECT id, name, created_at, total_trades, win_rate, total_pnl
FROM backtest_runs
ORDER BY created_at DESC
LIMIT 5;

-- 2. Ver signals generados
SELECT symbol, timeframe, signal, confidence, model_provider, environment
FROM signals
WHERE environment = 'backtest'
ORDER BY generated_at DESC
LIMIT 10;

-- 3. Ver orders ejecutadas
SELECT symbol, side, quantity, price, status, environment
FROM orders
WHERE environment = 'backtest'
ORDER BY created_at DESC
LIMIT 10;

-- 4. Ver trades cerrados
SELECT symbol, side, entry_price, exit_price, pnl, environment
FROM trades
WHERE environment = 'backtest' AND pnl IS NOT NULL
ORDER BY opened_at DESC
LIMIT 10;

-- Salir
\q
```

**Verificar**:
- [ ] `backtest_runs` tiene al menos 1 registro con métricas (`total_trades`, `win_rate`, etc.)
- [ ] `signals` tiene múltiples registros con `environment = 'backtest'`
- [ ] `orders` tiene registros con `status = 'FILLED'` y `environment = 'backtest'`
- [ ] `trades` tiene registros con `pnl` calculado

**Troubleshooting**:
- Si tablas vacías: verificar que el script `run_backtest.py` terminó sin errores
- Si no hay trades: puede ser que todos los signals fueron HOLD (verificar en signals)

---

#### 1.7 Verificar resultados en UI (5 min)

**Refrescar la app Streamlit (F5)**

**En la pestaña "Backtesting"**:
- [ ] La tabla de runs muestra el backtest ejecutado con:
  - status: `completed`
  - progress: `100`
  - métricas: `win_rate`, `profit_factor`, `sharpe_ratio`, etc.

**En la pestaña "Analyses"**:
1. En `Environment`: seleccionar **backtest**
2. Click en cualquier filtro (o dejar por defecto)
- [ ] Se muestran los signals generados
- [ ] Al expandir un detalle, se ven:
  - Indicators: RSI, MACD, Stochastic, ROC, Williams %R
  - Pattern y Trend
  - Model provider y model name

**En la pestaña "Orders & Positions"**:
1. En `Environment`: seleccionar **backtest**
- [ ] Se muestran las órdenes ejecutadas
- [ ] Columnas: symbol, side (BUY/SELL), quantity, price, status, created_at

**En la pestaña "Dashboard"**:
1. En `Environment`: seleccionar **backtest**
- [ ] KPIs muestran valores del backtest:
  - Portfolio Value (si hay posiciones abiertas)
  - Daily P&L (si hay trades del día actual)
  - Win Rate (% de trades ganadores)
  - Open Positions, Open Orders
- [ ] Tabla de **Recent Trades** muestra trades cerrados con PnL

---

### ✅ Criterios de Éxito del Caso 1

- [ ] Perfil creado y visible en DB
- [ ] Backtest ejecutado vía script sin errores
- [ ] Métricas generadas (total_trades > 0, win_rate calculado)
- [ ] Datos persistidos en 4 tablas: backtest_runs, signals, orders, trades
- [ ] UI muestra resultados correctamente en todas las pestañas relevantes
- [ ] No hay errores en logs de Streamlit o terminal

**Si todo pasa**: ✅ El circuito base funciona correctamente

---

## Caso de Test 2: Replay con Diferentes Perfiles (30 min)

### Objetivo
Validar que:
- Se pueden crear múltiples perfiles de riesgo
- El sistema de replay usa un backtest existente
- Se comparan métricas entre diferentes configuraciones
- La UI de Replay funciona correctamente

### Pasos Detallados

#### 2.1 Crear segundo perfil de riesgo (5 min)

**En la pestaña "Configuration"**:

1. `Profile kind`: **combined**
2. `Profile name`: **aggressive-profile**
3. `Universe`: **BTC, SPX** (mismo que antes)
4. `Profile JSON`:
```json
{
  "universe": ["BTC", "SPX"],
  "base_position_pct": 0.10,
  "max_position_pct": 0.20,
  "max_daily_loss_pct": 0.10,
  "slippage_pct": 0.01
}
```
5. Click **Save profile**

**Verificar**:
- [ ] Perfil guardado con versión 1
- [ ] Aparece en tabla de Profiles

**Diferencias vs. test-mvp-profile**:
- `base_position_pct`: 5% → 10% (posiciones más grandes)
- `max_position_pct`: 10% → 20% (permite posiciones más agresivas)
- `max_daily_loss_pct`: 5% → 10% (mayor tolerancia a pérdidas diarias)

---

#### 2.2 Identificar backtest_run para replay (3 min)

**En la pestaña "Backtesting"**:

1. Mirar la tabla de **Runs**
2. Identificar el `run_id` del backtest completado en Caso 1
3. Anotar:
   - run_id: **___**
   - assets: **['BTC', 'SPX']**
   - timeframe: **4h**
   - range_start: **____**
   - range_end: **____**

**Verificar**:
- [ ] El run tiene `status = completed`
- [ ] Tiene métricas: `win_rate`, `profit_factor`, etc.

---

#### 2.3 Ejecutar replay con perfil agresivo (20 min)

**Nota**: El replay UI está implementado pero la ejecución backend requiere un script.

**Crear script temporal: `run_replay.py`**:

```python
"""
Script temporal para ejecutar replay con perfil diferente.
"""
from datetime import datetime
from quantagent.backtesting.backtest import Backtest
from quantagent.database import SessionLocal
from quantagent.models import BacktestRun

def main():
    # Obtener el backtest_run original
    db = SessionLocal()
    original_run = db.query(BacktestRun).filter_by(id=YOUR_RUN_ID).one()  # Reemplazar YOUR_RUN_ID

    print(f"Replaying backtest run: {original_run.id}")
    print(f"Original config: {original_run.config_snapshot}")

    # Configuración del replay (perfil agresivo)
    replay_config = {
        'base_position_pct': 0.10,  # Más agresivo
        'max_daily_loss_pct': 0.10,
        'max_position_pct': 0.20,
        'slippage_pct': 0.01,
        'agent_llm_provider': 'openai',
        'agent_llm_model': 'gpt-4o-mini',
        'agent_llm_temperature': 0.1
    }

    # Crear nuevo backtest con análisis reutilizados
    replay = Backtest(
        start_date=original_run.start_date,
        end_date=original_run.end_date,
        assets=original_run.assets,
        timeframe=original_run.timeframe,
        initial_capital=100000.0,
        config=replay_config,
        use_checkpointing=True
    )

    print("-" * 60)
    metrics = replay.run(name=f"Replay {original_run.id} - Aggressive")

    # Mostrar comparación
    print("\n" + "=" * 60)
    print("REPLAY RESULTS - Aggressive Profile")
    print("=" * 60)
    print(f"Total Trades:      {metrics.total_trades}")
    print(f"Win Rate:          {metrics.win_rate:.2%}")
    print(f"Profit Factor:     {metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown:      {metrics.max_drawdown:.2%}")
    print(f"Total P&L:         ${metrics.total_pnl:,.2f}")
    print(f"Total Return:      {metrics.total_return_pct:.2f}%")
    print("=" * 60)

    print("\nOriginal vs Replay Comparison:")
    if original_run.total_trades:
        print(f"Trades:        {original_run.total_trades} → {metrics.total_trades}")
        print(f"Win Rate:      {original_run.win_rate:.2%} → {metrics.win_rate:.2%}")
        print(f"Sharpe:        {original_run.sharpe_ratio:.2f} → {metrics.sharpe_ratio:.2f}")
        print(f"Total P&L:     ${original_run.total_pnl:,.2f} → ${metrics.total_pnl:,.2f}")

    db.close()

if __name__ == "__main__":
    main()
```

**Ejecutar**:
```bash
python run_replay.py
```

**Verificar**:
- [ ] Script ejecuta sin errores
- [ ] Se muestran métricas del replay
- [ ] Se muestra comparación entre original y replay
- [ ] Las métricas son **diferentes** (por el perfil más agresivo)

**Qué esperar**:
- **Total Trades**: Debería ser similar o ligeramente mayor (mismos signals)
- **Total P&L**: Probablemente mayor (posiciones más grandes)
- **Max Drawdown**: Probablemente mayor (más riesgo)
- **Sharpe Ratio**: Puede ser mayor o menor (depende de la volatilidad)

---

#### 2.4 Verificar resultados en UI (5 min)

**Refrescar Streamlit (F5)**

**En la pestaña "Backtesting"**:
- [ ] Ahora hay **2 runs** en la tabla:
  - Run original (test-mvp-profile)
  - Run de replay (aggressive-profile)
- [ ] Ambos tienen `status = completed`
- [ ] Métricas son diferentes entre ambos

**En la pestaña "Replay"** (si está implementada):
- [ ] Se puede seleccionar el backtest_run original
- [ ] Se puede seleccionar el perfil aggressive-profile
- [ ] (Nota: la ejecución UI puede estar pendiente, pero datos están en DB)

---

### ✅ Criterios de Éxito del Caso 2

- [ ] Segundo perfil creado con parámetros más agresivos
- [ ] Replay ejecutado reutilizando el backtest original
- [ ] Métricas del replay son diferentes (esperado por config distinta)
- [ ] Ambos runs visibles en UI con métricas correctas
- [ ] No hay errores durante la ejecución

**Si todo pasa**: ✅ El sistema de replay funciona correctamente

---

## Caso de Test 3: Validación de Risk Management (20-30 min)

### Objetivo
Validar que:
- Los límites de riesgo se aplican correctamente
- Las órdenes rechazadas se registran
- El circuit breaker funciona cuando se alcanza pérdida máxima diaria
- La UI refleja correctamente los rechazos

### Pasos Detallados

#### 3.1 Crear perfil de riesgo muy restrictivo (5 min)

**En la pestaña "Configuration"**:

1. `Profile kind`: **combined**
2. `Profile name`: **ultra-conservative**
3. `Profile JSON`:
```json
{
  "universe": ["BTC"],
  "base_position_pct": 0.01,
  "max_position_pct": 0.02,
  "max_daily_loss_pct": 0.01,
  "slippage_pct": 0.01
}
```
4. Click **Save profile**

**Características**:
- Posiciones muy pequeñas (1% base, 2% max)
- Circuit breaker muy estricto (1% pérdida diaria)
- Solo BTC (un activo)

---

#### 3.2 Crear script de test de riesgo (5 min)

**Crear `test_risk_limits.py`**:

```python
"""
Test para validar que el RiskManager rechaza trades correctamente.
"""
from datetime import datetime, timedelta
from quantagent.backtesting.backtest import Backtest

def main():
    config = {
        'base_position_pct': 0.01,
        'max_daily_loss_pct': 0.01,  # MUY restrictivo
        'max_position_pct': 0.02,
        'slippage_pct': 0.01,
        'agent_llm_provider': 'openai',
        'agent_llm_model': 'gpt-4o-mini',
        'agent_llm_temperature': 0.1
    }

    # Backtest corto (7 días) para generar rápido algunos trades
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    backtest = Backtest(
        start_date=start_date,
        end_date=end_date,
        assets=['BTC'],
        timeframe='4h',
        initial_capital=100000.0,
        config=config,
        use_checkpointing=True
    )

    print("Testing ultra-conservative risk limits...")
    print(f"Max daily loss: 1% = ${1000.00}")
    print(f"Base position: 1% = ${1000.00}")
    print("-" * 60)

    metrics = backtest.run(name="Risk Limits Test - Ultra Conservative")

    print("\n" + "=" * 60)
    print("RISK MANAGEMENT TEST RESULTS")
    print("=" * 60)
    print(f"Total Signals Generated: [check signals table]")
    print(f"Total Trades Executed:   {metrics.total_trades}")
    print(f"Max Drawdown:            {metrics.max_drawdown:.2%}")
    print(f"Total P&L:               ${metrics.total_pnl:,.2f}")
    print("=" * 60)

    # Validación
    if metrics.max_drawdown <= 0.015:  # 1.5% tolerance
        print("\n✅ Risk limits working: Max drawdown stayed within limit")
    else:
        print(f"\n⚠️ Max drawdown exceeded limit: {metrics.max_drawdown:.2%} > 1.5%")

    print("\nCheck logs for rejected trades due to:")
    print("  - Insufficient capital")
    print("  - Position size limit")
    print("  - Daily loss limit")
    print("  - Circuit breaker triggered")

if __name__ == "__main__":
    main()
```

**Ejecutar**:
```bash
python test_risk_limits.py
```

---

#### 3.3 Analizar resultados (10 min)

**En Terminal, verificar logs durante ejecución**:

Buscar líneas como:
```
WARNING: Trade rejected: Daily loss limit exceeded (-$1,023.45 > -$1,000.00)
WARNING: Trade rejected: Position size would exceed 2% limit
INFO: Circuit breaker triggered at 2025-XX-XX XX:XX - stopping all trades
```

**Verificar en DB**:

```bash
docker-compose exec db psql -U postgres -d quantagent_dev
```

```sql
-- 1. Contar signals vs trades ejecutados
SELECT
    (SELECT COUNT(*) FROM signals WHERE symbol = 'BTC' AND environment = 'backtest') as total_signals,
    (SELECT COUNT(*) FROM orders WHERE symbol = 'BTC' AND environment = 'backtest' AND status = 'FILLED') as filled_orders;

-- 2. Ver si hay órdenes rechazadas (esto requeriría un campo "rejection_reason" en Order model)
-- Por ahora, comparamos: si signals >> filled_orders, hubo rechazos

-- 3. Verificar que no hay trades con pérdidas excesivas
SELECT symbol, pnl
FROM trades
WHERE environment = 'backtest' AND symbol = 'BTC'
ORDER BY pnl ASC
LIMIT 10;

\q
```

**Verificar**:
- [ ] Hay más signals generados que trades ejecutados (esperado por rechazos)
- [ ] Max drawdown <= 1.5% (circuit breaker funcionó)
- [ ] No hay trades individuales con pérdidas > $1,000 aprox
- [ ] En logs: mensajes de "Trade rejected"

---

#### 3.4 Comparar con perfil normal (5 min)

**Crear tabla comparativa manual**:

| Métrica | test-mvp-profile (Caso 1) | ultra-conservative (Caso 3) |
|---------|---------------------------|------------------------------|
| Total Signals | ___ | ___ |
| Total Trades | ___ | ___ |
| Rejection Rate | ___ | ___ |
| Max Drawdown | ___% | ___% |
| Total P&L | $____ | $____ |

**Calcular**:
```
Rejection Rate = (Total Signals - Total Trades) / Total Signals * 100
```

**Qué esperar**:
- **ultra-conservative** debería tener:
  - Rejection Rate más alto (>50%)
  - Total Trades menor
  - Max Drawdown menor (<2%)
  - Total P&L probablemente menor (menos trades, más pequeños)

---

#### 3.5 Verificar en UI (5 min)

**En la pestaña "Dashboard" (environment: backtest)**:
- [ ] Win Rate y métricas reflejan el perfil conservador
- [ ] Daily P&L no excede límites

**En la pestaña "Analyses"**:
- [ ] Filtrar por `symbol = BTC`
- [ ] Verificar que hay signals generados
- [ ] Algunos no tienen `order_id` (rechazados, no ejecutados)

**En la pestaña "Orders & Positions"**:
- [ ] Cantidad de órdenes es menor que signals totales
- [ ] Todas las órdenes tienen `quantity` pequeño (1% posiciones)

---

### ✅ Criterios de Éxito del Caso 3

- [ ] Perfil ultra-conservative creado
- [ ] Backtest ejecutado con límites restrictivos
- [ ] **Rejection Rate > 30%** (muchas órdenes rechazadas)
- [ ] **Max Drawdown ≤ 1.5%** (circuit breaker funcionó)
- [ ] Logs muestran mensajes de rechazo con razones específicas
- [ ] UI refleja correctamente trades ejecutados vs signals generados
- [ ] No hay trades que violen límites de riesgo

**Si todo pasa**: ✅ El sistema de risk management funciona correctamente

---

## Resumen de Validación del MVP

### Checklist Final

**Circuito Completo**:
- [ ] Configuración de perfiles (portfolio + risk)
- [ ] Creación de backtest runs
- [ ] Ejecución de análisis (TradingGraph invocado)
- [ ] Generación de signals (Indicator, Pattern, Trend, Decision)
- [ ] Sizing de posiciones (PositionSizer)
- [ ] Validación de riesgo (RiskManager)
- [ ] Ejecución de órdenes (PaperBroker con slippage)
- [ ] Actualización de portfolio (PortfolioManager)
- [ ] Cálculo de métricas (BacktestMetrics)
- [ ] Persistencia en DB (6 tablas: backtest_runs, signals, orders, fills, trades, positions)
- [ ] Visualización en UI (Dashboard, Analyses, Backtesting, Orders)

**Funcionalidades Avanzadas**:
- [ ] Replay con perfiles diferentes (reutiliza análisis)
- [ ] Risk management con rechazos (circuit breaker)
- [ ] Comparación de métricas entre runs
- [ ] Filtrado y búsqueda en Analyses

**Quality Gates**:
- [ ] No hay errores en logs de Streamlit
- [ ] No hay errores en terminal de ejecución
- [ ] Todas las tablas de DB se populan correctamente
- [ ] Métricas son consistentes (win_rate entre 0-100%, profit_factor > 0, etc.)
- [ ] La UI responde sin freezes (<3s por vista)

---

## Próximos Pasos Tras Validación

**Si los 3 casos pasan exitosamente**:

1. **Implementar APScheduler** (Week 9-10):
   - Scheduler para ejecutar backtests automáticos
   - Scheduler para paper trading en vivo (cada 1h)

2. **Completar Documentación**:
   - Setup guide actualizado
   - Configuration guide con ejemplos de perfiles
   - Troubleshooting guide

3. **Tests de Integración Automatizados**:
   - Convertir estos casos manuales en `pytest` end-to-end
   - CI/CD para ejecutar en cada commit

4. **Validación de Estrategia**:
   - Backtest en 3-4 meses de datos
   - Validar métricas: Win Rate ≥40%, Sharpe ≥1.0, Max DD ≤15%
   - Decidir si proceder a Phase 2 (real broker) o iterar

**Si algún caso falla**:

- Identificar el componente que falla (ver sección de Troubleshooting)
- Crear issue en GitHub con logs y pasos para reproducir
- Corregir y re-ejecutar todos los casos desde el principio

---

## Troubleshooting Common Issues

### Database Connection Errors

**Síntoma**: `Could not connect to database`

**Soluciones**:
```bash
# 1. Verificar PostgreSQL está corriendo
docker-compose ps

# 2. Reiniciar PostgreSQL
docker-compose restart db

# 3. Verificar DATABASE_URL
cat .env | grep DATABASE_URL

# 4. Test de conexión
python -c "from quantagent.database import SessionLocal; db = SessionLocal(); print('Connected!'); db.close()"
```

---

### API Key Errors

**Síntoma**: `AuthenticationError` o `Invalid API key`

**Soluciones**:
```bash
# 1. Verificar variables de entorno
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# 2. Verificar .env
cat .env | grep API_KEY

# 3. Exportar temporalmente
export OPENAI_API_KEY="sk-..."
```

---

### Backtest No Generates Trades

**Síntoma**: `Total Trades: 0` pero signals existen

**Causas posibles**:
1. Todos los signals fueron HOLD
2. RiskManager rechazó todos los trades
3. Errores en OrderManager

**Debugging**:
```sql
-- Ver distribución de signals
SELECT signal, COUNT(*)
FROM signals
WHERE environment = 'backtest'
GROUP BY signal;

-- Si todos son HOLD: ajustar fechas o assets para períodos con más volatilidad
```

---

### Streamlit Crashes or Freezes

**Síntoma**: La app se congela o cierra inesperadamente

**Soluciones**:
```bash
# 1. Reiniciar Streamlit
Ctrl+C
streamlit run apps/streamlit/app.py

# 2. Limpiar cache de Streamlit
rm -rf ~/.streamlit/cache

# 3. Verificar logs
# Terminal de Streamlit mostrará errores Python
```

---

## Anexo: Queries Útiles de Validación

```sql
-- Resumen completo de un backtest_run
SELECT
    br.id, br.name, br.created_at,
    br.total_trades, br.win_rate, br.sharpe_ratio, br.max_drawdown, br.total_pnl,
    COUNT(DISTINCT s.id) as signals_generated,
    COUNT(DISTINCT o.id) as orders_placed,
    COUNT(DISTINCT t.id) as trades_closed
FROM backtest_runs br
LEFT JOIN signals s ON s.environment = 'backtest'
LEFT JOIN orders o ON o.environment = 'backtest'
LEFT JOIN trades t ON t.environment = 'backtest'
WHERE br.id = YOUR_RUN_ID
GROUP BY br.id;

-- Top 5 trades más rentables
SELECT symbol, side, entry_price, exit_price, pnl,
       (pnl / (entry_price * quantity) * 100) as return_pct
FROM trades
WHERE environment = 'backtest' AND pnl IS NOT NULL
ORDER BY pnl DESC
LIMIT 5;

-- Top 5 trades más perdedores
SELECT symbol, side, entry_price, exit_price, pnl,
       (pnl / (entry_price * quantity) * 100) as return_pct
FROM trades
WHERE environment = 'backtest' AND pnl IS NOT NULL
ORDER BY pnl ASC
LIMIT 5;

-- Análisis de signals por confidence
SELECT
    CASE
        WHEN confidence >= 0.8 THEN 'High (≥0.8)'
        WHEN confidence >= 0.6 THEN 'Medium (0.6-0.8)'
        ELSE 'Low (<0.6)'
    END as confidence_bucket,
    COUNT(*) as count
FROM signals
WHERE environment = 'backtest'
GROUP BY confidence_bucket
ORDER BY count DESC;
```

---

**Documento creado**: 2025-12-08
**Autor**: Planning para validación MVP
**Estado**: Ready for execution
