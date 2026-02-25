# Design: Complete Universe Management in Configuration UI

**Issue ID**: QuantAgent-ia2  
**Type**: Feature Enhancement  
**Level**: MVP

---

## Componentes Afectados

- `apps/streamlit_ui.py` — Tab Configuration: agregar widget multiselect Universe
- `quantagent/models/strategy_config.py` — Validar `json_config.universe`
- `quantagent/backtest/engine.py` — Resolver assets desde Universe del perfil
- `quantagent/data/provider.py` — Exportar SYMBOL_MAPPING keys para UI

---

## Decisiones Técnicas

### 1. Storage de Universe
**Decisión**: Almacenar como lista de strings en `json_config.universe`  
**Razón**: 
- Consistente con estructura flexible de `json_config` (JSONB en DB)
- No requiere cambios en esquema de base de datos
- Fácil de serializar/deserializar
- Permite extensiones futuras (ej: metadata por símbolo)

**Alternativas consideradas**:
- Campo dedicado `universe` en tabla → rechazado (requires migration, less flexible)
- String CSV → rechazado (parsing manual, error-prone)

### 2. Validación de Universe
**Decisión**: Validar en dos capas:
1. **UI (Streamlit)**: Restrictiva (multiselect solo símbolos válidos)
2. **Model (StrategyConfig)**: Defensiva (valida contra SYMBOL_MAPPING)

**Razón**: Defense in depth; UI puede bypassearse (API, scripts)

### 3. Precedencia Assets vs Universe
**Decisión**: `backtest.assets` (explícito) > `profile.universe` (default)  
**Razón**: 
- Flexibilidad: permite override temporal sin modificar perfil
- Compatibilidad: backtests existentes con `assets` siguen funcionando
- Explícito > implícito (clarity)

### 4. Symbol List Source
**Decisión**: UI consulta `DataProvider.SYMBOL_MAPPING.keys()` directamente  
**Razón**: Single source of truth; evita duplicación

**Alternativa considerada**: Hardcoded list en UI → rechazado (sync issues)

### 5. Preview Format
**Decisión**: JSON pretty-printed con syntax highlight (st.json())  
**Razón**: 
- Nativo en Streamlit
- Fácil de leer
- Consistente con current profile preview

---

## Contratos

### 1. StrategyConfig.json_config Schema (Extended)

```python
{
  "base_position_pct": float,          # Existing
  "max_position_pct": float,           # Existing
  "max_daily_loss_pct": float,         # Existing
  "slippage_pct": float,               # Existing
  "universe": list[str],               # NEW - optional, default []
}
```

**Validación**:
```python
def validate_universe(universe: list[str]) -> None:
    """Valida que todos los símbolos estén en SYMBOL_MAPPING."""
    from quantagent.data.provider import DataProvider
    
    valid_symbols = set(DataProvider.SYMBOL_MAPPING.keys())
    invalid = set(universe) - valid_symbols
    
    if invalid:
        raise ValueError(
            f"Invalid symbols in universe: {invalid}. "
            f"Valid symbols: {sorted(valid_symbols)}"
        )
```

### 2. BacktestEngine.run() Signature (Extended)

```python
def run(
    self,
    assets: Optional[list[str]] = None,  # Existing - high priority
    strategy_profile: Optional[StrategyConfig] = None,  # Existing
    # ...
) -> BacktestResult:
    """
    Run backtest.
    
    Assets resolution order:
    1. Explicit `assets` parameter (highest priority)
    2. strategy_profile.json_config.get("universe", [])
    3. Empty list (error or system default)
    """
    resolved_assets = assets or (
        strategy_profile.json_config.get("universe", [])
        if strategy_profile else []
    )
    
    if not resolved_assets:
        raise ValueError("No assets specified. Provide `assets` or configure Universe in profile.")
    
    # ... existing backtest logic
```

### 3. Streamlit UI Widget

```python
# apps/streamlit_ui.py - Tab 2: Configuration

from quantagent.data.provider import DataProvider

# Get available symbols
available_symbols = sorted(DataProvider.SYMBOL_MAPPING.keys())

# Multiselect widget
selected_universe = st.multiselect(
    label="Universe",
    options=available_symbols,
    default=current_profile.json_config.get("universe", []),
    help="Select instruments to include in this portfolio profile. "
         "Backtests will use these symbols if no explicit assets are provided."
)

# Update json_config
json_config["universe"] = selected_universe

# Preview
st.subheader("Profile Preview")
st.json({
    "name": profile_name,
    "kind": "portfolio",
    "json_config": json_config
})
```

---

## Flujo de Datos

### Crear/Editar Perfil
```
UI (streamlit_ui.py)
  ↓ user selects symbols in multiselect
  ↓ on "Save Profile" button
  ↓ construct json_config with universe key
  ↓ 
StrategyConfig.create()
  ↓ validate_universe(json_config["universe"])
  ↓ save to database (json_config as JSONB)
  ↓
UI displays success message
```

### Ejecutar Backtest con Universe
```
UI (streamlit_ui.py)
  ↓ user selects profile (with universe)
  ↓ leaves assets field empty
  ↓ on "Run Backtest" button
  ↓
BacktestEngine.run(strategy_profile=profile, assets=None)
  ↓ resolved_assets = profile.json_config.get("universe", [])
  ↓ validate resolved_assets not empty
  ↓
DataProvider.get_ohlc() for each symbol in resolved_assets
  ↓
Agent processing per symbol
  ↓
Backtest results persisted
```

---

## Casos Edge

### 1. Universe vacío en perfil
**Comportamiento**: Backtest.run() lanza `ValueError` si assets=None y universe=[]  
**Razón**: Evitar backtests silenciosamente sin assets

### 2. Símbolos duplicados en Universe
**Comportamiento**: Normalizar automáticamente (dedup) al guardar  
**Implementación**: `json_config["universe"] = list(set(selected_universe))`

### 3. Perfil sin clave `universe`
**Comportamiento**: Tratar como lista vacía (`json_config.get("universe", [])`)  
**Razón**: Backward compatibility con perfiles existentes

### 4. Assets explícito + Universe
**Comportamiento**: Assets gana (ignorar Universe)  
**UI Feedback**: Mostrar warning en UI si ambos están configurados

### 5. Símbolo en Universe sin datos históricos
**Comportamiento**: Error en runtime (DataProvider.get_ohlc fails)  
**Futuro**: Pre-validar disponibilidad de datos (out of scope MVP)

---

## Cambios en Archivos

### 1. `quantagent/data/provider.py`
- **Cambio**: Ninguno (SYMBOL_MAPPING ya existe y es público)
- **Nota**: UI importa directamente `DataProvider.SYMBOL_MAPPING`

### 2. `quantagent/models/strategy_config.py`
- **Agregar**: Método `validate_universe()` (classmethod o helper)
- **Agregar**: Validación en `__init__` o property setter (si existe)

### 3. `quantagent/backtest/engine.py`
- **Modificar**: `run()` method para resolver assets desde Universe
- **Agregar**: Logging de resolved assets
- **Agregar**: Validación de empty assets

### 4. `apps/streamlit_ui.py`
- **Modificar**: Tab 2 (Configuration) section
- **Agregar**: Widget `st.multiselect` para Universe
- **Agregar**: Preview actualizado con Universe
- **Agregar**: Helper para cargar símbolos disponibles

---

## Testing Strategy

### Unit Tests
- `test_strategy_config.py`: Validación de Universe (válidos, inválidos, vacíos)
- `test_backtest_engine.py`: Assets resolution logic (precedencia)

### Integration Tests
- `test_backtest_with_universe.py`: Backtest end-to-end usando Universe del perfil

### UI Tests (Manual)
- Crear perfil con Universe → verificar persistencia
- Backtest con Universe → verificar ejecución
- Backtest con assets explícito → verificar override

---

## Performance Considerations

- **Multiselect rendering**: 10 symbols → negligible overhead
- **Validation**: O(n) where n=len(universe) → acceptable (n≤10 for MVP)
- **Assets resolution**: O(1) dict lookup → no impact

---

## Security Considerations

- **SQL Injection**: JSONB field con validation → safe (ORM handles escaping)
- **Path Traversal**: Symbols son alfanuméricos → no filesystem interaction
- **DoS**: Limited symbol set (10 items) → no concern

---

## Rollout Plan

1. **Phase 1**: Agregar validation logic en models (no UI changes)
2. **Phase 2**: Implementar assets resolution en BacktestEngine
3. **Phase 3**: Agregar UI multiselect widget en Configuration tab
4. **Phase 4**: Testing end-to-end + docs update

**Rollback**: Si issues críticos, remover widget de UI (backend sigue funcionando con assets explícito)

---

## Future Enhancements (Out of Scope)

- Universe templates (ej: "Crypto", "Commodities", "Indices")
- Universe por timeframe
- Validación de data availability al seleccionar
- Import/Export Universe configs
- Universe versioning (track changes over time)
