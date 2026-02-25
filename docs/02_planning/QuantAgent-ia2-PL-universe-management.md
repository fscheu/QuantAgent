# Planning: Complete Universe Management in Configuration UI

**Issue ID**: QuantAgent-ia2  
**Type**: Feature Enhancement  
**Estimated Effort**: 3–4 hours

---

## Tareas

### 1. Agregar validación de Universe en StrategyConfig
**Estimado**: 45 min  
**Dependencias**: Ninguna  
**Prioridad**: Alta

**Subtareas**:
- Agregar helper `validate_universe(universe: list[str]) -> None` en `quantagent/models/strategy_config.py`
- Importar `DataProvider.SYMBOL_MAPPING` para validación
- Llamar validación en `__init__` o setter si `json_config` contiene `universe`
- Agregar test unitario `test_validate_universe_valid()`
- Agregar test unitario `test_validate_universe_invalid()`
- Agregar test unitario `test_validate_universe_empty()`

**Archivos afectados**:
- `quantagent/models/strategy_config.py`
- `tests/test_strategy_config.py` (crear si no existe)

**Verificación**:
```python
# Test case
config = StrategyConfig(
    name="test",
    kind="portfolio",
    json_config={"universe": ["BTC", "INVALID"]}
)
# Debe lanzar ValueError
```

---

### 2. Implementar resolución de assets desde Universe en BacktestEngine
**Estimado**: 60 min  
**Dependencias**: Tarea 1  
**Prioridad**: Alta

**Subtareas**:
- Modificar `quantagent/backtest/engine.py::run()` method
- Agregar lógica de resolución:
  ```python
  resolved_assets = assets or (
      strategy_profile.json_config.get("universe", [])
      if strategy_profile else []
  )
  ```
- Agregar validación: lanzar `ValueError` si `resolved_assets` está vacío
- Agregar logging de resolved assets (`logger.info(f"Using assets: {resolved_assets}")`)
- Agregar test unitario `test_backtest_uses_universe_when_no_assets()`
- Agregar test unitario `test_backtest_assets_override_universe()`
- Agregar test unitario `test_backtest_error_on_empty_assets_and_universe()`

**Archivos afectados**:
- `quantagent/backtest/engine.py`
- `tests/test_backtest_engine.py`

**Verificación**:
```python
# Test case
profile = StrategyConfig(..., json_config={"universe": ["BTC", "SPX"]})
engine = BacktestEngine(...)
result = engine.run(strategy_profile=profile, assets=None)
# Debe procesar BTC y SPX
```

---

### 3. Agregar multiselect widget en Configuration tab
**Estimado**: 60 min  
**Dependencias**: Tarea 1, 2  
**Prioridad**: Alta

**Subtareas**:
- Localizar sección Portfolio profile editor en `apps/streamlit_ui.py`
- Importar `DataProvider.SYMBOL_MAPPING`
- Agregar helper para obtener símbolos disponibles:
  ```python
  def get_available_symbols() -> list[str]:
      from quantagent.data.provider import DataProvider
      return sorted(DataProvider.SYMBOL_MAPPING.keys())
  ```
- Agregar widget `st.multiselect`:
  - Label: "Universe"
  - Options: `get_available_symbols()`
  - Default: `current_profile.json_config.get("universe", [])` si editando
  - Help text explicativo
- Actualizar `json_config["universe"]` con selección
- Normalizar duplicados: `json_config["universe"] = list(set(selected_universe))`

**Archivos afectados**:
- `apps/streamlit_ui.py`

**Verificación**:
- Iniciar UI: `streamlit run apps/streamlit_ui.py`
- Navegar a Configuration tab
- Verificar widget visible con 10 símbolos
- Seleccionar 3 símbolos y guardar
- Verificar perfil guardado con `universe` en DB

---

### 4. Agregar preview de Universe en perfil
**Estimado**: 30 min  
**Dependencias**: Tarea 3  
**Prioridad**: Media

**Subtareas**:
- Actualizar sección Profile Preview en Configuration tab
- Incluir `universe` en JSON preview:
  ```python
  st.json({
      "name": profile_name,
      "kind": profile_kind,
      "json_config": json_config  # Ya incluye universe
  })
  ```
- Opcional: Agregar sección separada "Selected Universe" con badges:
  ```python
  st.write("**Selected Universe:**")
  for symbol in json_config.get("universe", []):
      st.badge(symbol)
  ```

**Archivos afectados**:
- `apps/streamlit_ui.py`

**Verificación**:
- Seleccionar símbolos en multiselect
- Verificar preview se actualiza en tiempo real
- Confirmar JSON preview incluye clave `universe`

---

### 5. Agregar warning en Backtesting tab si assets y Universe ambos configurados
**Estimado**: 20 min  
**Dependencias**: Tarea 2, 3  
**Prioridad**: Baja (nice-to-have)

**Subtareas**:
- En Backtesting tab, detectar si:
  - User selecciona perfil con Universe
  - Y también especifica assets manualmente
- Mostrar warning:
  ```python
  if assets_input and selected_profile and selected_profile.json_config.get("universe"):
      st.warning(
          "⚠️ Both `assets` and profile `universe` are configured. "
          "The explicit `assets` list will be used (Universe ignored)."
      )
  ```

**Archivos afectados**:
- `apps/streamlit_ui.py`

**Verificación**:
- Configurar perfil con Universe
- En Backtesting tab, seleccionar ese perfil
- Especificar assets manualmente
- Verificar warning aparece

---

### 6. Tests de integración end-to-end
**Estimado**: 45 min  
**Dependencias**: Todas las anteriores  
**Prioridad**: Alta

**Subtareas**:
- Crear `tests/integration/test_backtest_with_universe.py`
- Test case 1: Crear perfil con Universe → ejecutar backtest sin assets
- Test case 2: Crear perfil con Universe → ejecutar backtest con assets override
- Test case 3: Perfil sin Universe → backtest falla si no assets
- Verificar resultados esperados (trades, equity, etc.)

**Archivos afectados**:
- `tests/integration/test_backtest_with_universe.py` (nuevo)

**Verificación**:
```bash
pytest tests/integration/test_backtest_with_universe.py -v
# Todos los tests pasan
```

---

### 7. Actualizar documentación
**Estimado**: 20 min  
**Dependencias**: Todas las anteriores  
**Prioridad**: Media

**Subtareas**:
- Actualizar `docs/01_requirements/ui_streamlit_mvp_requirements.md`:
  - Marcar Universe management como completado en Tab 2
- Opcional: Agregar ejemplo de uso en README.md
- Opcional: Agregar screenshot de UI con multiselect

**Archivos afectados**:
- `docs/01_requirements/ui_streamlit_mvp_requirements.md`
- (opcional) `README.md`

**Verificación**:
- Docs consistentes con implementación
- No referencias obsoletas

---

## Orden de Ejecución Recomendado

1. **Tarea 1** (validación models) → base sólida
2. **Tarea 2** (backtest engine) → lógica core
3. **Tarea 3** (UI multiselect) → feature visible
4. **Tarea 4** (preview) → UX improvement
5. **Tarea 6** (tests integración) → validación
6. **Tarea 5** (warning) → polish (opcional)
7. **Tarea 7** (docs) → cierre

---

## Riesgos y Mitigaciones

### Riesgo 1: Perfiles existentes sin `universe`
**Impacto**: Medio  
**Mitigación**: Usar `.get("universe", [])` en todas partes (backward compatible)

### Riesgo 2: Validación rompe perfiles legacy
**Impacto**: Alto  
**Mitigación**: Validar solo si `universe` está presente; no validar en load, solo en save

### Riesgo 3: DataProvider.SYMBOL_MAPPING cambia
**Impacto**: Bajo  
**Mitigación**: Single source of truth; cambios en SYMBOL_MAPPING se propagan automáticamente

### Riesgo 4: UI performance con muchos símbolos
**Impacto**: Bajo (solo 10 símbolos en MVP)  
**Mitigación**: Si crece a 100+, usar searchable dropdown (future)

---

## Métricas de Éxito

- ✅ Todos los tests unitarios pasan
- ✅ Test de integración end-to-end pasa
- ✅ UI permite crear perfil con Universe en <30 segundos
- ✅ Backtest con Universe ejecuta sin errores
- ✅ Assets explícito override funciona correctamente
- ✅ No regresiones en backtests existentes

---

## Rollback Plan

Si issues críticos después de deploy:

1. **Rollback UI**: Comentar widget multiselect → users usan assets explícito
2. **Rollback Engine**: Remover assets resolution → forzar assets explícito
3. **Keep Validation**: Mantener validación en models (no afecta backtests sin Universe)

**Condiciones para rollback**:
- Backtests fallan con Universe configurado
- Perfiles legacy no se pueden cargar
- Performance degradation significativa

---

## Post-Implementation

### Opcional (si tiempo permite)
- Agregar tooltip en UI explicando cada símbolo (ej: "BTC → Bitcoin")
- Agregar contador de símbolos seleccionados (ej: "3 of 10 selected")
- Agregar botón "Select All" / "Clear All"

### Future Enhancements (separar en nuevos issues)
- Universe templates (QuantAgent-xyz)
- Validación de data availability (QuantAgent-xyz)
- Universe por timeframe (QuantAgent-xyz)
