# Requirements: Complete Universe Management in Configuration UI

**Issue ID**: QuantAgent-ia2  
**Type**: Feature Enhancement  
**Level**: MVP  
**Labels**: configuration, mvp, streamlit

---

## Objetivo

Completar la funcionalidad de gestión de Universe (lista de instrumentos) en perfiles Portfolio dentro del tab Configuration de la UI Streamlit, permitiendo a los usuarios seleccionar fácilmente los símbolos que desean incluir en sus backtests.

---

## Alcance

### Incluye
- Widget multi-select para símbolos en editor de perfiles Portfolio
- Soporte para símbolos definidos en `DataProvider.SYMBOL_MAPPING`:
  - BTC, SPX, CL, DAX, ES, NQ, QQQ, GC, VIX, DXY
- Persistencia de Universe en `json_config` del `StrategyConfig`
- Preview de Universe resuelto antes de guardar perfil
- Integración con backtest: usar Universe del perfil cuando `assets` no se especifica
- Validación de símbolos seleccionados (todos deben estar en SYMBOL_MAPPING)

### No Incluye
- Reglas de exposición por sector (out of scope para MVP)
- Adición dinámica de nuevos símbolos (lista fija por ahora)
- Gestión de múltiples universes por perfil
- Universes compartidos entre perfiles
- Validación de disponibilidad de datos históricos al seleccionar

---

## Contexto Actual

### Estado del Sistema
- Modelo `StrategyConfig` soporta `json_config` con campo `universe`
- UI permite crear perfiles pero Universe management es limitado
- `DataProvider` define lista fija de símbolos soportados en `SYMBOL_MAPPING`
- Backtests actualmente requieren especificar `assets` explícitamente

### Referencias
- **UI Requirements**: `docs/01_requirements/ui_streamlit_mvp_requirements.md` Tab 2
- **Data Provider**: `quantagent/data/provider.py` (SYMBOL_MAPPING)
- **Strategy Config**: `quantagent/models/strategy_config.py`

---

## Constraints

- **Símbolos fijos**: Solo símbolos en `DataProvider.SYMBOL_MAPPING` son válidos
- **Validación**: Universe debe validarse antes de persistir
- **Compatibilidad**: No romper backtests existentes que especifican `assets`
- **Precedencia**: `assets` explícito en backtest > Universe del perfil > default vacío
- **UI Framework**: Streamlit multiselect widget (no custom components por MVP)

---

## Flujo de Usuario

### Crear/Editar Perfil Portfolio
1. Usuario navega a Configuration tab
2. Selecciona "Create Portfolio Profile" o edita perfil existente
3. Ve widget multiselect con símbolos disponibles (BTC, SPX, CL, etc.)
4. Selecciona uno o más símbolos para el Universe
5. Configura otros parámetros (sizing, risk limits, etc.)
6. Ve preview del perfil resuelto (incluye Universe)
7. Guarda perfil → Universe se persiste en `json_config.universe`

### Ejecutar Backtest con Universe del Perfil
1. Usuario navega a Backtesting tab
2. Selecciona perfil Portfolio (con Universe configurado)
3. Deja campo `assets` vacío (opcional)
4. Sistema usa `profile.json_config.universe` como lista de assets
5. Backtest ejecuta con los símbolos del Universe

---

## Acceptance Criteria

### AC1: Widget multiselect disponible
```
Given un usuario está creando/editando un perfil Portfolio
When abre el formulario de configuración
Then ve un widget multiselect con label "Universe"
  And la lista contiene todos los símbolos de DataProvider.SYMBOL_MAPPING
  And los símbolos están ordenados alfabéticamente
```

### AC2: Selección de Universe
```
Given el widget multiselect está visible
When el usuario selecciona 3 símbolos (ej: BTC, SPX, GC)
  And hace clic en "Save Profile"
Then el perfil se guarda con json_config.universe = ["BTC", "SPX", "GC"]
  And se muestra mensaje de confirmación
```

### AC3: Preview de Universe
```
Given un usuario está configurando un perfil con Universe
When selecciona símbolos en el multiselect
Then ve una sección "Preview" del perfil resuelto
  And el preview incluye el campo universe con los símbolos seleccionados
```

### AC4: Backtest usa Universe del perfil
```
Given un perfil Portfolio con universe = ["BTC", "SPX"]
  And un backtest configurado con ese perfil
  And el campo assets está vacío
When el backtest se ejecuta
Then procesa datos para BTC y SPX
  And no procesa otros símbolos
```

### AC5: Assets explícito override
```
Given un perfil Portfolio con universe = ["BTC", "SPX"]
  And un backtest configurado con assets = ["CL", "GC"]
When el backtest se ejecuta
Then procesa CL y GC (ignora Universe del perfil)
```

### AC6: Validación de símbolos
```
Given un usuario intenta guardar un perfil
  And el Universe contiene símbolos no válidos
When hace clic en "Save Profile"
Then se muestra error de validación
  And el perfil no se guarda
```

---

## Notas Técnicas

### Estructura de json_config
```python
{
  "base_position_pct": 0.1,
  "max_position_pct": 0.25,
  "max_daily_loss_pct": 0.05,
  "slippage_pct": 0.001,
  "universe": ["BTC", "SPX", "GC", "VIX"]  # Nueva clave
}
```

### Validación
- Todos los elementos en `universe` deben estar en `DataProvider.SYMBOL_MAPPING.keys()`
- Universe puede estar vacío (lista vacía válida)
- Universe con duplicados debe normalizarse automáticamente

### Precedencia de Assets
1. **Backtest.assets** (explícito) → máxima prioridad
2. **Profile.json_config.universe** → si backtest.assets vacío
3. **Lista vacía** → error o comportamiento default del sistema

---

## Out of Scope (Futuro)

- Universe por timeframe (ej: diferentes símbolos para 1h vs 1d)
- Universe dinámico basado en filtros (ej: "todos los cryptos")
- Validación de suficiencia de datos históricos
- Sugerencias de Universe basadas en correlación
- Import/Export de Universe templates
- Reglas de exposición por sector o clase de activo
