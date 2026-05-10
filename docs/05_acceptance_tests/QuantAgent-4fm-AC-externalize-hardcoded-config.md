# Acceptance Criteria: Externalize Hardcoded Trading Configuration

**Issue ID:** QuantAgent-4fm  
**Title:** Externalize hardcoded trading configuration to env or database  
**Created:** 2026-05-10

---

## AC-1: Variables de entorno en settings.py

**Given** el módulo `quantagent/settings.py`  
**When** se importa el módulo  
**Then** las siguientes variables están definidas y cargadas desde `.env`:

- `TRADING_INITIAL_CASH` (default: 100000.0)
- `TRADING_BASE_POSITION_PCT` (default: 0.05)
- `TRADING_MAX_DAILY_LOSS_PCT` (default: 0.05)
- `TRADING_MAX_POSITION_PCT` (default: 0.10)
- `TRADING_SLIPPAGE_PCT` (default: 0.01)
- `TRADING_USE_CHECKPOINTING` (default: False)
- `TRADING_UNIVERSE` (default: "")

**And** si las variables no existen en `.env`, se usan los defaults sin fallar

---

## AC-2: StrategyAssembler usa settings.py

**Given** el archivo `quantagent/strategy/assembler.py`  
**When** se llama a `StrategyAssembler.from_profiles(...)` sin argumentos  
**Then** los valores retornados en `ResolvedConfig` coinciden con los definidos en `settings.py`

**And** el diccionario `DEFAULTS` ya no existe o está deprecated

---

## AC-3: Backtest elimina fallbacks hardcoded

**Given** el archivo `quantagent/backtesting/backtest.py`  
**When** se revisan los métodos `__init__`, `_build_config_snapshot`, `_create_signal_from_strategy`, `_create_signal`  
**Then** no hay valores hardcoded inline del tipo:

```python
self.config.get("base_position_pct", 0.05)  # ❌ Hardcoded fallback
self.config.get("model_name", "gpt-4o-mini")  # ❌ Hardcoded fallback
```

**Instead** debe usar:

```python
self.config.get("base_position_pct", settings.TRADING_BASE_POSITION_PCT)  # ✅
```

---

## AC-4: Compatibilidad con overrides

**Given** un script que instancia `Backtest` con `config={"base_position_pct": 0.10}`  
**When** se ejecuta el backtest  
**Then** el valor usado es `0.10` (override explícito gana sobre env var)

**And** si no se pasa override, se usa el valor de `settings.TRADING_BASE_POSITION_PCT`

---

## AC-5: Tests existentes siguen pasando

**Given** la suite de tests del repo  
**When** se ejecuta `pytest`  
**Then** todos los tests que pasaban antes siguen pasando

**Específicamente:**
- Tests de `StrategyAssembler`
- Tests de `Backtest`
- Tests de integración con Streamlit (si existen)

---

## AC-6: Archivo .env.example actualizado

**Given** el archivo `.env.example` en la raíz  
**When** se revisa el contenido  
**Then** contiene las nuevas variables documentadas:

```bash
# Trading Strategy Defaults
TRADING_INITIAL_CASH=100000.0
TRADING_BASE_POSITION_PCT=0.05
TRADING_MAX_DAILY_LOSS_PCT=0.05
TRADING_MAX_POSITION_PCT=0.10
TRADING_SLIPPAGE_PCT=0.01
TRADING_USE_CHECKPOINTING=false
TRADING_UNIVERSE=""
```

**And** cada variable tiene un comentario explicativo

---

## AC-7: Backwards compatibility total

**Given** código existente que usa `StrategyAssembler` o `Backtest` sin pasar config  
**When** se ejecuta ese código  
**Then** el comportamiento es idéntico al anterior (mismos defaults numéricos)

**Específicamente:**
- Un backtest sin config usa initial_cash=100000.0
- Un assembler sin profiles usa base_position_pct=0.05
- No hay errores ni warnings nuevos

---

## Oráculos de Validación

### 1. Grep de valores hardcoded

**Command:**
```bash
cd ~/repos/projects/QuantAgent
grep -n '0.05\|100000\|"gpt-4o-mini"\|"openai"' quantagent/strategy/assembler.py quantagent/backtesting/backtest.py
```

**Expected:** Solo aparecen en docstrings o imports, no en código ejecutable de defaults

### 2. Import test

**Command:**
```bash
python3 -c "from quantagent import settings; print(settings.TRADING_INITIAL_CASH)"
```

**Expected:** Imprime `100000.0` (o el valor definido en .env)

### 3. Assembler test

**Command:**
```bash
python3 -c "
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.models import Environment
cfg = StrategyAssembler.from_profiles(environment=Environment.BACKTEST)
assert cfg.initial_cash == 100000.0, f'Expected 100000.0, got {cfg.initial_cash}'
print('✓ StrategyAssembler usa settings correctamente')
"
```

**Expected:** Imprime mensaje de éxito sin assertion errors

### 4. Backtest initialization test

**Command:**
```bash
python3 -c "
from quantagent.backtesting.backtest import Backtest
from datetime import datetime
bt = Backtest(
    start_date=datetime(2024,1,1),
    end_date=datetime(2024,1,2),
    assets=['BTC']
)
# Si no falla al inicializar, la config por default funciona
print('✓ Backtest inicializa con defaults de settings.py')
"
```

**Expected:** No falla, imprime mensaje de éxito

---

## Métricas de Éxito

- **0 hardcoded defaults** en assembler.py y backtest.py (fuera de deprecation warnings)
- **100% tests passing** (antes y después del cambio)
- **0 breaking changes** reportados en QA manual con Streamlit
