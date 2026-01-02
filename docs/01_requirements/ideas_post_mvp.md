* Cada ejecución del agente se debería almacenar en la base de datos, junto con las imagenes generadas (estas posiblemente en disco). con el checkpointer se debería poder lograr, pero sino guardar los objetos de respuesta.

## Position Management Strategies (Post-MVP)

Implementar sistema completo de estrategias de manejo de posiciones configurables:

### Estrategias a Soportar

1. **one_position_only** (conservadora) - Solo una posición a la vez
2. **pyramiding** - Añadir a posiciones ganadoras cuando señal continúa
3. **scale_in** - Entrada gradual en múltiples tranches
4. **reversal_only** - Solo operar en cambios de dirección

### Implementación Técnica

- **Ubicación**: Extender `RiskManager` con parámetros de estrategia
- **Configuración**: Agregar `position_strategy` a config de backtest y estrategias
- **Validación**: En `validate_trade()`, verificar estrategia ANTES de otras validaciones
- **Parámetros**:
  - `position_strategy: str` (default: "one_position_only")
  - `allow_pyramiding: bool`
  - `pyramid_profit_threshold_pct: float`
  - `max_pyramid_layers: int`

### Beneficios

- Control fino sobre cuándo añadir a posiciones
- Soporte para diferentes estilos de trading
- Reducción de over-trading
- Mejor gestión de riesgo

### Documentación

Ver diseño completo en: `docs/03_technical/POSITION_MANAGEMENT_STRATEGIES.md`

**Fecha agregada**: 2026-01-02
**Prioridad**: Media (post-MVP)
**Estado**: Diseñado, pendiente implementación
