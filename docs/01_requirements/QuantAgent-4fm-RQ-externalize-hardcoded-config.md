# Requirements: Externalize Hardcoded Trading Configuration

**Issue ID:** QuantAgent-4fm  
**Title:** Externalize hardcoded trading configuration to env or database  
**Created:** 2026-05-10  
**Level of Detail:** STANDARD

---

## Objetivo

Externalizar todas las configuraciones de trading hardcodeadas en el código fuente hacia variables de entorno (vía `settings.py`) o configuración en base de datos, permitiendo que sean modificables sin cambios en el código de producción.

---

## Scope (Qué entra)

1. **StrategyAssembler.DEFAULTS** (`quantagent/strategy/assembler.py`, líneas 54-65)
   - `initial_cash`: 100000.0
   - `base_position_pct`: 0.05
   - `max_daily_loss_pct`: 0.05
   - `max_position_pct`: 0.10
   - `slippage_pct`: 0.01
   - `model_provider`: "openai"
   - `model_name`: "gpt-4o-mini"
   - `temperature`: 0.1
   - `use_checkpointing`: False
   - `universe`: []

2. **Backtest fallback defaults** (`quantagent/backtesting/backtest.py`)
   - Fallbacks en `__init__` (líneas 134-149)
   - Fallbacks en `_build_config_snapshot` (líneas 322-336)
   - Fallbacks en `_create_signal_from_strategy` (líneas 605-607)
   - Fallbacks en `_create_signal` (líneas 690-692)

3. **Actualización de settings.py**
   - Agregar nuevas variables de entorno para parámetros de trading
   - Mantener compatibilidad con configuración existente de LLMs

---

## Non-Scope (Qué NO entra)

- No se modifica la lógica de negocio de trading
- No se cambia la estructura de la base de datos (usar solo env vars por ahora)
- No se refactoriza el sistema de perfiles (portfolio/risk/model)
- No se toca TradingGraph (ya usa settings.py correctamente)
- No se implementa UI para cambiar estos valores (queda para futuro)

---

## Contexto Relevante

### Estado actual

El sistema ya soporta configuración por environment variables:
- Archivo `.env` en la raíz del proyecto
- Módulo `quantagent/settings.py` carga las variables con `dotenv`
- TradingGraph ya consume toda su config desde `settings.py`
- El patrón de `settings.update_env_file()` permite persistir cambios en runtime

### Problema

Los defaults de StrategyAssembler están hardcoded en el código, lo que significa:
- Cambiar defaults requiere modificar código Python
- No hay un lugar centralizado para ver/modificar estos valores
- Los backtests usan fallbacks inline que duplican estos valores
- Inconsistencia: TradingGraph usa settings.py pero StrategyAssembler no

### Por qué importa

- **Operabilidad**: Cambiar defaults de riesgo/portfolio requiere deploy
- **Experimentación**: Backtests con diferentes defaults requieren cambiar código
- **Visibilidad**: No está claro cuáles son los defaults actuales sin leer el código
- **Consistencia**: Mixing de hardcoded values y env vars es error-prone

---

## Criterios de "Done"

1. ✅ Todas las constantes de `StrategyAssembler.DEFAULTS` están definidas en `settings.py`
2. ✅ `StrategyAssembler` lee defaults desde `settings.py` en lugar del diccionario hardcoded
3. ✅ `backtest.py` elimina todos los fallbacks inline y usa `settings.py`
4. ✅ Los tests existentes siguen pasando
5. ✅ La aplicación Streamlit sigue funcionando sin cambios
6. ✅ El archivo `.env.example` documenta las nuevas variables
7. ✅ Los valores default preservan el comportamiento actual (no hay breaking changes)

---

## Casos Edge

### Compatibilidad con overrides existentes

Si un usuario o test pasa `config={"base_position_pct": 0.10}`, debe seguir funcionando. Los overrides tienen prioridad sobre env vars.

**Orden de precedencia (debe mantenerse):**
1. Overrides explícitos (argumento `config` o `overrides`)
2. Profiles (portfolio_profile, risk_profile, model_profile)
3. **Environment variables (nuevo)** ← se inserta acá
4. ~~Hardcoded DEFAULTS~~ ← se elimina

### Valores faltantes en .env

Si una variable no está definida en `.env`, usar el valor actual hardcoded como fallback. No debe fallar la aplicación.

### Backwards compatibility

Código existente que no pasa ningún config debe seguir funcionando con los mismos defaults que antes.

---

## Decisiones Requeridas

Ninguna. El approach está claro: mover defaults a settings.py manteniendo compatibilidad.

---

## Restricciones

- **No cambiar valores default**: Los valores numéricos actuales deben preservarse
- **No modificar código de producción**: Solo cambios en archivos de configuración y settings
- **No breaking changes**: Código existente debe seguir funcionando sin modificaciones
