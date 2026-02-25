# Acceptance Criteria: Complete Universe Management in Configuration UI

**Issue ID**: QuantAgent-ia2  
**Type**: Feature Enhancement

---

## Criterios de Aceptación

### AC1: Widget multiselect disponible en Configuration tab
```
Given un usuario navega al Configuration tab
  And selecciona "Create Portfolio Profile" o "Edit Profile"
When el formulario de configuración se renderiza
Then ve un widget multiselect con label "Universe"
  And el widget contiene todos los símbolos de DataProvider.SYMBOL_MAPPING
  And los símbolos están ordenados alfabéticamente: [BTC, CL, DAX, DXY, ES, GC, NQ, QQQ, SPX, VIX]
  And el widget tiene help text explicativo
```

### AC2: Selección y persistencia de Universe
```
Given un usuario está creando un perfil Portfolio
When selecciona 3 símbolos: BTC, SPX, GC
  And completa otros campos requeridos (name, sizing params)
  And hace clic en "Save Profile"
Then el perfil se guarda exitosamente
  And json_config.universe = ["BTC", "SPX", "GC"]
  And se muestra mensaje de confirmación
  And el perfil aparece en la lista de perfiles disponibles
```

### AC3: Edición de Universe existente
```
Given existe un perfil con universe = ["BTC", "SPX"]
When el usuario edita ese perfil
Then el widget multiselect muestra BTC y SPX preseleccionados
  And puede agregar/remover símbolos
  And al guardar, el universe se actualiza correctamente
```

### AC4: Preview de Universe
```
Given un usuario está configurando un perfil
When selecciona símbolos en el multiselect: ["BTC", "GC", "VIX"]
Then ve una sección "Profile Preview"
  And el preview muestra json_config.universe con los símbolos seleccionados
  And el preview se actualiza en tiempo real al cambiar selección
```

### AC5: Backtest usa Universe cuando assets no especificado
```
Given un perfil Portfolio con universe = ["BTC", "SPX"]
  And el usuario navega a Backtesting tab
  And selecciona ese perfil
  And deja el campo "assets" vacío
  And configura timeframe = "1h", date range válido
When hace clic en "Run Backtest"
Then el backtest se ejecuta
  And procesa datos solo para BTC y SPX
  And genera analyses para ambos símbolos
  And produce métricas de resultado
  And el backtest completa sin errores
```

### AC6: Assets explícito override Universe
```
Given un perfil Portfolio con universe = ["BTC", "SPX"]
  And el usuario en Backtesting tab selecciona ese perfil
  And especifica assets = ["CL", "GC"] manualmente
When ejecuta el backtest
Then el backtest usa CL y GC (ignora Universe del perfil)
  And no procesa BTC ni SPX
  And el backtest completa exitosamente
```

### AC7: Validación de símbolos inválidos
```
Given un usuario intenta crear perfil via API/script (bypass UI)
  And el json_config contiene universe = ["BTC", "INVALID_SYMBOL"]
When el sistema intenta guardar el perfil
Then se lanza ValueError
  And el mensaje indica símbolos inválidos: "INVALID_SYMBOL"
  And el mensaje lista símbolos válidos disponibles
  And el perfil NO se guarda en base de datos
```

### AC8: Universe vacío es válido
```
Given un usuario crea un perfil Portfolio
  And no selecciona ningún símbolo (universe vacío)
When guarda el perfil
Then el perfil se guarda con json_config.universe = []
  And no se lanza error de validación
```

### AC9: Error cuando backtest sin assets ni Universe
```
Given un perfil Portfolio con universe = [] (vacío)
  And el usuario intenta ejecutar backtest con ese perfil
  And el campo assets está vacío
When hace clic en "Run Backtest"
Then se muestra error antes de ejecutar
  And el mensaje indica "No assets specified. Provide assets or configure Universe."
  And el backtest NO se ejecuta
```

### AC10: Normalización de duplicados
```
Given un usuario selecciona símbolos con duplicados (edge case via script)
When el sistema guarda el perfil
Then universe se normaliza automáticamente (dedup)
  And solo se almacenan valores únicos
Example: Input ["BTC", "SPX", "BTC"] → Stored ["BTC", "SPX"]
```

### AC11: Backward compatibility con perfiles existentes
```
Given perfiles creados antes de esta feature (sin clave "universe")
When el sistema carga esos perfiles para backtest
Then no se lanza error
  And se trata universe como lista vacía
  And requiere assets explícito para ejecutar backtest
```

### AC12: UI warning cuando ambos configurados
```
Given un perfil con universe = ["BTC", "SPX"]
  And el usuario en Backtesting tab selecciona ese perfil
  And también especifica assets = ["CL"]
When ambos están configurados
Then se muestra warning amarillo
  And el mensaje indica que assets explícito tendrá prioridad
  And el usuario puede proceder con claridad
```

---

## Criterios de Regresión

### REG1: Backtests existentes con assets explícito siguen funcionando
```
Given backtests históricos que especifican assets directamente
When se ejecutan después de implementar Universe
Then completan sin cambios en comportamiento
  And producen resultados idénticos
  And no se lanza ningún error
```

### REG2: Perfiles sin Universe no rompen UI
```
Given perfiles Portfolio legacy sin clave "universe"
When se muestran en Configuration tab
Then aparecen correctamente en lista
  And pueden editarse sin error
  And el widget multiselect está vacío (sin selección previa)
```

### REG3: Validación no afecta load de perfiles
```
Given un perfil con universe válido
When se carga desde base de datos
Then no se ejecuta validación (solo en save)
  And la carga es rápida
  And no hay overhead de validación
```

---

## Criterios de Performance

### PERF1: Multiselect rendering
```
Given 10 símbolos disponibles
When el widget multiselect se renderiza
Then aparece en < 100ms
  And la interacción es fluida (sin lag)
```

### PERF2: Validación overhead
```
Given un perfil con universe de 10 símbolos
When se valida antes de guardar
Then la validación completa en < 10ms
  And no impacta UX
```

### PERF3: Assets resolution
```
Given un backtest con Universe de 5 símbolos
When se resuelven assets desde Universe
Then la resolución es O(1)
  And no agrega latencia mensurable al inicio del backtest
```

---

## Criterios de Usability

### UX1: Claridad de widget
```
Given un usuario nuevo ve el multiselect
Then entiende su propósito sin documentación externa
  And el help text es suficiente
  And los labels son claros
```

### UX2: Preview inmediato
```
Given un usuario cambia selección de símbolos
Then el preview se actualiza inmediatamente
  And no requiere guardar para ver cambios
```

### UX3: Feedback de guardado
```
Given un usuario guarda perfil con Universe
Then recibe confirmación visual clara
  And puede verificar que Universe se guardó correctamente
```

---

## Test Cases Manuales

### Manual Test 1: Flujo completo end-to-end
1. Iniciar Streamlit UI
2. Navegar a Configuration tab
3. Crear nuevo Portfolio profile
4. Seleccionar 3 símbolos: BTC, SPX, GC
5. Configurar sizing params
6. Guardar perfil
7. Verificar perfil en lista
8. Navegar a Backtesting tab
9. Seleccionar perfil creado
10. Dejar assets vacío
11. Ejecutar backtest
12. Verificar completa exitosamente

**Resultado esperado**: Backtest procesa solo BTC, SPX, GC

### Manual Test 2: Override con assets explícito
1. Usar perfil con Universe = ["BTC", "SPX"]
2. En Backtesting tab, especificar assets = ["CL"]
3. Verificar warning aparece
4. Ejecutar backtest
5. Verificar procesa solo CL

**Resultado esperado**: Warning visible, backtest usa solo CL

### Manual Test 3: Edición de Universe
1. Editar perfil existente con Universe = ["BTC"]
2. Agregar "SPX" y "GC"
3. Remover "BTC"
4. Guardar
5. Recargar perfil
6. Verificar Universe = ["SPX", "GC"]

**Resultado esperado**: Cambios persisten correctamente

---

## Definition of Done

- ✅ Todos los Acceptance Criteria pasan
- ✅ Tests unitarios cubren validación y resolution logic
- ✅ Test de integración end-to-end pasa
- ✅ Manual tests completados exitosamente
- ✅ No regresiones detectadas
- ✅ Documentación actualizada
- ✅ Code review aprobado
- ✅ UI funciona en Chrome/Firefox
- ✅ Logs apropiados para debugging
- ✅ Error messages son claros y accionables

---

## Notas de Testing

### Datos de Test
```python
# Valid symbols from DataProvider.SYMBOL_MAPPING
VALID_SYMBOLS = ["BTC", "SPX", "CL", "DAX", "ES", "NQ", "QQQ", "GC", "VIX", "DXY"]

# Test profiles
TEST_PROFILE_1 = {
    "name": "crypto_only",
    "kind": "portfolio",
    "json_config": {
        "universe": ["BTC"],
        "base_position_pct": 0.1
    }
}

TEST_PROFILE_2 = {
    "name": "diversified",
    "kind": "portfolio", 
    "json_config": {
        "universe": ["BTC", "SPX", "GC", "CL"],
        "base_position_pct": 0.05
    }
}

TEST_PROFILE_3 = {
    "name": "legacy_no_universe",
    "kind": "portfolio",
    "json_config": {
        "base_position_pct": 0.1
        # No universe key
    }
}
```

### Test Environment
- **Database**: Test DB con datos limpios
- **Market Data**: Mock data para símbolos test
- **Date Range**: 2024-01-01 to 2024-01-31 (suficiente para test)
- **Timeframe**: "1h" (rápido para test)

---

## Sign-off

- [ ] Developer testing complete
- [ ] Code review approved
- [ ] QA testing passed
- [ ] Product owner acceptance
- [ ] Documentation updated
- [ ] Ready for deployment
