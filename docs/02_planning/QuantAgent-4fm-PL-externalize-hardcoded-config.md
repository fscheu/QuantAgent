# Planning: Externalize Hardcoded Trading Configuration

**Issue ID:** QuantAgent-4fm  
**Title:** Externalize hardcoded trading configuration to env or database  
**Created:** 2026-05-10

---

## Resumen Ejecutivo

Refactor conservador de configuración hardcoded: mover 10 constantes de trading desde diccionarios Python hacia `settings.py` + `.env`, manteniendo 100% de compatibilidad con código existente.

**Esfuerzo estimado:** 2-4 horas  
**Riesgo:** Bajo (no cambia lógica de negocio)  
**Valor:** Alto (habilita operabilidad y experimentación)

---

## Tareas (bite-sized, ~30-60 min cada una)

### Task 1: Agregar variables de entorno a settings.py

**Objetivo:** Definir las nuevas variables en el módulo de configuración

**Pasos:**

1. Abrir `quantagent/settings.py`
2. Después de las variables de LLM (línea ~68), agregar sección de Trading Defaults:

```python
# Trading Strategy Defaults (QuantAgent-4fm)
TRADING_INITIAL_CASH: float = float(os.getenv("TRADING_INITIAL_CASH", "100000.0"))
TRADING_BASE_POSITION_PCT: float = float(os.getenv("TRADING_BASE_POSITION_PCT", "0.05"))
TRADING_MAX_DAILY_LOSS_PCT: float = float(os.getenv("TRADING_MAX_DAILY_LOSS_PCT", "0.05"))
TRADING_MAX_POSITION_PCT: float = float(os.getenv("TRADING_MAX_POSITION_PCT", "0.10"))
TRADING_SLIPPAGE_PCT: float = float(os.getenv("TRADING_SLIPPAGE_PCT", "0.01"))
TRADING_USE_CHECKPOINTING: bool = os.getenv("TRADING_USE_CHECKPOINTING", "false").lower() in {"true", "1", "yes"}
TRADING_UNIVERSE: str = os.getenv("TRADING_UNIVERSE", "")
```

3. Agregar helper para parsear universe (lista de símbolos separados por comas):

```python
def get_trading_universe() -> list[str]:
    """Parse TRADING_UNIVERSE env var into list of symbols."""
    if not TRADING_UNIVERSE:
        return []
    return [s.strip().upper() for s in TRADING_UNIVERSE.split(",") if s.strip()]
```

**Verificación:**
```bash
python3 -c "from quantagent import settings; print(settings.TRADING_INITIAL_CASH)"
```

**Dependencias:** Ninguna  
**Salida:** Variables cargadas en settings.py

---

### Task 2: Refactorizar StrategyAssembler.DEFAULTS

**Objetivo:** Reemplazar diccionario hardcoded por referencias a settings.py

**Pasos:**

1. Abrir `quantagent/strategy/assembler.py`
2. Reemplazar el diccionario `DEFAULTS` (líneas 54-65) por:

```python
@property
@staticmethod
def DEFAULTS() -> dict:
    """Default strategy configuration (loaded from settings.py)."""
    from quantagent import settings
    return {
        "initial_cash": settings.TRADING_INITIAL_CASH,
        "base_position_pct": settings.TRADING_BASE_POSITION_PCT,
        "max_daily_loss_pct": settings.TRADING_MAX_DAILY_LOSS_PCT,
        "max_position_pct": settings.TRADING_MAX_POSITION_PCT,
        "slippage_pct": settings.TRADING_SLIPPAGE_PCT,
        "model_provider": settings.AGENT_LLM_PROVIDER,
        "model_name": settings.AGENT_LLM_MODEL,
        "temperature": settings.AGENT_LLM_TEMPERATURE,
        "use_checkpointing": settings.TRADING_USE_CHECKPOINTING,
        "universe": settings.get_trading_universe(),
    }
```

**Nota:** Se mantiene como dict-returning function para no romper código que accede `StrategyAssembler.DEFAULTS[...]`

**Alternativa más limpia (opcional):** Convertir a función estática:

```python
@staticmethod
def get_defaults() -> dict:
    ...
```

Y buscar/reemplazar todas las referencias `StrategyAssembler.DEFAULTS` → `StrategyAssembler.get_defaults()`

**Verificación:**
```bash
python3 -c "
from quantagent.strategy.assembler import StrategyAssembler
defaults = StrategyAssembler.DEFAULTS
assert defaults['initial_cash'] == 100000.0
print('✓ DEFAULTS usa settings')
"
```

**Dependencias:** Task 1  
**Salida:** StrategyAssembler usa settings.py

---

### Task 3: Eliminar fallbacks hardcoded en backtest.py

**Objetivo:** Reemplazar valores inline por referencias a settings

**Pasos:**

1. Abrir `quantagent/backtesting/backtest.py`
2. En el método `__init__` (líneas 131-152), reemplazar fallbacks:

**Antes:**
```python
resolved = StrategyAssembler.from_snapshot(
    {
        "initial_cash": initial_capital,
        "base_position_pct": self.config.get("base_position_pct", 0.05),
        ...
        "model_name": self.config.get("agent_llm_model", self.config.get("model_name", "gpt-4o-mini")),
        ...
    },
    ...
)
```

**Después:**
```python
from quantagent import settings

resolved = StrategyAssembler.from_snapshot(
    {
        "initial_cash": initial_capital,
        "base_position_pct": self.config.get("base_position_pct", settings.TRADING_BASE_POSITION_PCT),
        "max_daily_loss_pct": self.config.get("max_daily_loss_pct", settings.TRADING_MAX_DAILY_LOSS_PCT),
        "max_position_pct": self.config.get("max_position_pct", settings.TRADING_MAX_POSITION_PCT),
        "slippage_pct": self.config.get("slippage_pct", settings.TRADING_SLIPPAGE_PCT),
        "model_provider": self.config.get("agent_llm_provider", self.config.get("model_provider", settings.AGENT_LLM_PROVIDER)),
        "model_name": self.config.get("agent_llm_model", self.config.get("model_name", settings.AGENT_LLM_MODEL)),
        "temperature": self.config.get("agent_llm_temperature", self.config.get("temperature", settings.AGENT_LLM_TEMPERATURE)),
        "use_checkpointing": use_checkpointing,
        "universe": self.config.get("universe", settings.get_trading_universe()),
    },
    environment=Environment.BACKTEST,
)
```

3. Repetir lo mismo en `_build_config_snapshot` (líneas 319-340)

4. En `_create_signal_from_strategy` (líneas 605-607), reemplazar:

**Antes:**
```python
model_provider=self.config.get("agent_llm_provider", "openai"),
model_name=self.config.get("agent_llm_model", "gpt-4o-mini"),
temperature=self.config.get("agent_llm_temperature", 0.1),
```

**Después:**
```python
model_provider=self.config.get("agent_llm_provider", settings.AGENT_LLM_PROVIDER),
model_name=self.config.get("agent_llm_model", settings.AGENT_LLM_MODEL),
temperature=self.config.get("agent_llm_temperature", settings.AGENT_LLM_TEMPERATURE),
```

5. Repetir lo mismo en `_create_signal` (líneas 690-692)

**Verificación:** Buscar hardcoded values:
```bash
grep -n '"openai"\|"gpt-4o-mini"\|0.05\|0.10\|100000' quantagent/backtesting/backtest.py
```

Debe retornar solo líneas en comments/docstrings, no en código ejecutable.

**Dependencias:** Task 1  
**Salida:** backtest.py usa settings.py para todos los defaults

---

### Task 4: Actualizar .env.example

**Objetivo:** Documentar las nuevas variables de entorno

**Pasos:**

1. Abrir `.env.example` en la raíz del repo
2. Agregar sección después de las variables de LLM:

```bash
# ============================================================================
# Trading Strategy Defaults (QuantAgent-4fm)
# ============================================================================
# Capital inicial para backtests y paper trading
TRADING_INITIAL_CASH=100000.0

# Tamaño base de posición como % del capital (0.05 = 5%)
TRADING_BASE_POSITION_PCT=0.05

# Pérdida máxima diaria como % del capital (0.05 = 5%)
TRADING_MAX_DAILY_LOSS_PCT=0.05

# Tamaño máximo de posición como % del capital (0.10 = 10%)
TRADING_MAX_POSITION_PCT=0.10

# Slippage simulado como % del precio (0.01 = 1%)
TRADING_SLIPPAGE_PCT=0.01

# Habilitar checkpointing de LangGraph (PostgreSQL requerido)
TRADING_USE_CHECKPOINTING=false

# Universo de activos por default (comma-separated, ej: "BTC,ETH,SPX")
# Dejar vacío para no tener default
TRADING_UNIVERSE=""
```

3. Si existe `.env` (no debe estar en git), asegurarse de que no tiene estas variables seteadas (para que use defaults)

**Verificación:**
```bash
cat .env.example | grep TRADING_
```

**Dependencias:** Ninguna (puede hacerse en paralelo)  
**Salida:** .env.example actualizado

---

### Task 5: Testing y validación

**Objetivo:** Verificar que no se rompió nada

**Pasos:**

1. **Unit tests:**
```bash
cd ~/repos/projects/QuantAgent
pytest tests/unit/test_assembler.py -v
pytest tests/unit/test_backtest.py -v
```

2. **Import test:**
```bash
python3 -c "
from quantagent import settings
from quantagent.strategy.assembler import StrategyAssembler
from quantagent.backtesting.backtest import Backtest
from datetime import datetime

# Test settings
assert hasattr(settings, 'TRADING_INITIAL_CASH')
assert settings.TRADING_INITIAL_CASH == 100000.0
print('✓ Settings OK')

# Test assembler
cfg = StrategyAssembler.from_profiles()
assert cfg.initial_cash == 100000.0
assert cfg.base_position_pct == 0.05
print('✓ Assembler OK')

# Test backtest init
bt = Backtest(
    start_date=datetime(2024,1,1),
    end_date=datetime(2024,1,2),
    assets=['BTC']
)
assert bt.initial_capital == 100000.0
print('✓ Backtest OK')

print('\\n✅ All validations passed')
"
```

3. **Manual QA con Streamlit:**
```bash
cd ~/repos/projects/QuantAgent
python -m streamlit run apps/streamlit/app.py
```

- Navegar a Configuration → verificar que los valores se muestran correctamente
- Navegar a Backtesting → lanzar un backtest simple
- Verificar logs: no debe haber warnings sobre missing config

4. **Grep final de hardcoded values:**
```bash
grep -rn '0.05\|100000\|"gpt-4o-mini"' quantagent/strategy/assembler.py quantagent/backtesting/backtest.py | grep -v '#'
```

Resultado esperado: solo comentarios y docstrings, no código ejecutable

**Dependencias:** Tasks 1-4  
**Salida:** Todos los tests pasan, QA manual OK

---

## Rollout Strategy

1. **Feature branch:** `feature/QuantAgent-4fm-externalize-hardcoded-config`
2. **Commits:**
   - "feat(settings): add trading strategy defaults from env"
   - "refactor(assembler): use settings.py for DEFAULTS"
   - "refactor(backtest): replace hardcoded fallbacks with settings"
   - "docs: update .env.example with trading defaults"
   - "test: validate config externalization"

3. **PR checklist:**
   - [ ] Todos los tests pasan
   - [ ] `.env.example` actualizado
   - [ ] QA manual con Streamlit OK
   - [ ] No hay valores hardcoded en grep
   - [ ] Beads comment agregado

4. **Merge:** Squash merge a main después de review humano

---

## Riesgos y Mitigaciones

### Riesgo 1: Breaking change accidental

**Probabilidad:** Baja  
**Impacto:** Alto  
**Mitigación:** Tests exhaustivos + QA manual antes de merge

### Riesgo 2: Orden de precedencia cambia

**Probabilidad:** Media  
**Impacto:** Medio  
**Mitigación:** Documentar explícitamente en código que override > profiles > env > (removed hardcoded)

### Riesgo 3: Performance de import

**Probabilidad:** Muy baja  
**Impacto:** Bajo  
**Mitigación:** `settings.py` ya se importa en todos lados, agregar 7 variables más es negligible

---

## Dependencies

- Ninguna dependency externa
- No requiere cambios en DB schema
- No requiere upgrade de packages

---

## Post-Implementation

### Follow-ups (fuera de scope)

- [ ] UI en Streamlit para editar estos valores (QuantAgent-XXX)
- [ ] Persistir overrides en DB en lugar de solo .env (QuantAgent-XXX)
- [ ] Validaciones de rango (ej: `base_position_pct` debe estar entre 0.01 y 0.50)
- [ ] Deprecar `StrategyAssembler.DEFAULTS` como dict, forzar uso de function

### Monitoreo

- Logs en backtests deben mostrar de dónde vienen los defaults (env vs override)
- Streamlit config page debe mostrar source de cada valor (hardcoded/env/override)
