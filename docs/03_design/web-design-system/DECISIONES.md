# Decisiones abiertas

Cada fila tiene un valor provisorio para no bloquear el desarrollo. Cuando se decida, completar la última columna y actualizar el archivo indicado.

| # | Duda | Valor provisorio | Dónde vive | Se decide cuando… | Decisión |
|---|---|---|---|---|---|
| 1 | Nombres y función de las etapas 0–3 | "Etapa 0…3" | `config.ts`, `help-texts.es.json` | Se cierre el diseño del embudo | — |
| 2 | Motivos de descarte (salvo los 3 dados) | Inventados, de ejemplo | Backend | El motor del embudo los emita | — |
| 3 | Métrica que muestra cada hipótesis | Sharpe deflactado | Backend (`last_metric`) | Se definan las métricas por etapa | — |
| 4 | Umbrales del stop | cerca < 2 %, atención < 5 %, escala 15 % | `config.ts` | Después de ver stops reales | — |
| 5 | Distancia en % o en ATR | % | `config.ts` | Idem | — |
| 6 | Posiciones cortas | — | `stopDistance()` | — | **Cerrada 30/10/2026**: se muestran. Barra con eje de precio (corta crece desde la derecha), distancia (stop − precio)/precio, mismas zonas, badge Larga/Corta, `side` en `Position`. |
| 7 | Orden por defecto de posiciones | Distancia al stop ascendente | Pantalla Cuenta | 2 semanas de uso | — |
| 8 | Rechazadas siempre arriba | Sí, también con "Todas" | Pantalla Cuenta | 2 semanas de uso | — |
| 9 | Ciclo por defecto en Embudo | Todos | `config.ts` | Prueba con datos reales | — |
| 10 | Hipótesis en escritorio: tarjetas o tabla | Tarjetas | Pantalla Embudo | Más de 30 activas | — |
| 11 | Alarmas: catálogo y severidades | drawdown (crítica), sombra (advertencia), datos viejos (advertencia) | Backend + EVENTOS.md | Se implementen las alarmas | — |
| 12 | Umbral de "dato viejo" | 30 min con mercado abierto | `config.ts` | Se conozca la frecuencia real del backend | — |
| 13 | Frecuencia de refresco | 60 s abierto / 15 min cerrado | `config.ts` | Idem | — |
| 14 | Dividendos: retención por pago o total del mes | Ambos (total + lista) | Pantalla Cuenta | Se implemente | — |
| 15 | Ventana del feed | Última visita o 24 h | `config.ts` | No urgente | — |
| 16 | Publicaciones macro a seguir en FRED | Inflación, empleo | Backend (lista blanca) | Se implemente el calendario | — |
| 17 | Biblioteca de gráficos (no velas) | Recharts | `package.json` | Primer gráfico | — |
| 18 | Secciones "próximamente": datos de ejemplo o fantasma | Fantasma punteado en producción | Flag | Primera entrega | — |
| 19 | Resumen con cortas: qué es "invertido" | Largo y corto por separado + efectivo; sin "exposición neta" | Pantalla Cuenta | Se use con cortas reales | — |
| 20 | % de P&L en cortas: sobre precio de entrada | (entrada − actual) / entrada | Backend (`unrealized_pl_pct`) | Confirmar que el backend calcula igual | — |
| 21 | Barra de stop en corta: espejada (crece desde la derecha) vs siempre desde la izquierda | Espejada (eje de precio) | `StopDistanceBar` | 2 semanas de uso con cortas | — |
| 22 | Estado "stop superado" | Se muestra si el precio pasó el stop y la posición sigue abierta | `stopDistance()` | Ver cuánto pasa en la práctica | — |
| 23 | Motivo de cuenta sin datos | `no_data_reason` con 3 valores; si el backend no distingue, `unknown` | Backend | Al implementar el chequeo de cuentas | — |
| 24 | Frecuencia de reconsulta de cuenta sin datos | 15 min | `config.ts` | Idem | — |
| 25 | Vencidas: ¿también "Vencidas" en el feed de Inicio? | No (no requieren acción) | EVENTOS.md | Uso real | — |
| 26 | DataTable: tercer clic en el encabezado | Vuelve al orden por defecto | `DataTable` | Primer uso en Mercado | — |
| 27 | Pantallas de Live con datos (v0.1) | Quedan como referencia de "cuando la cuenta esté abierta" | Canvas | Al abrir la cuenta live | — |
| 28 | Umbrales de métricas de Rendimiento | — | Backend (parámetros) + espejo en `performanceConfig` | — | **01/11/2026**: escalones 21 (descriptivo) / 63 (riesgo con aviso) / 252 (anualizadas). Provisorios: parámetros del backend devueltos en `sample.thresholds`, no fijos en el diseño. |
| 29 | Aviso “pocos rebalanceos” | — | Backend (parámetro) | — | **01/11/2026**: < 12, razonable para estrategia mensual; provisorio y parametrizable. |
| 30 | Tasa libre de riesgo | Letra del Tesoro de EE. UU. a 3 meses, diaria | Backend (`conventions`) | Al implementar | — |
| 31 | Selector de Período: global o solo de la curva | Global (afecta métricas y todos los gráficos); `EquityChart` sin selector propio en esta pantalla | `SubjectHeader` | Uso real | — |
| 32 | Cuenta: pestañas “Hoy · Rendimiento” vs sección con enlace | Pestañas | Pantalla Cuenta | Uso real | — |
| 33 | Distribución: mensual o diaria | — | `ReturnsHistogram` | — | **01/11/2026**: mensual por defecto + conmutador “Diaria” secundario desde 63 días, con VaR 95 % y CVaR. |
| 34 | Ventanas de métricas móviles | 6M y 12M (126 y 252 días) | `performanceConfig` | Uso real | — |
| 35 | Comparar dos sujetos con períodos distintos | Período común (`align_with`) y aviso en el encabezado | Backend | Al implementar | — |
| 36 | Lista de backtests: ubicación | — | Navegación | — | **01/11/2026**: sección propia `/backtests` en la navegación principal (5 destinos). La ficha de hipótesis enlaza al backtest. |
| 37 | Cartera: retorno ponderado por tiempo | Cálculo por snapshots y flujos registrados por cuenta | Backend | Cerrada con PERF-05 el 01/11/2026; flujos sin conciliar devuelven `returns_not_computable`. |
| 38 | Cómo se asigna capital por estrategia (para aislar su rendimiento) | Una cuenta por estrategia, sin caja virtual: el rendimiento de `run` es el de su cuenta | Backend (`models.py`, `cartera_paper.py`) | Cerrada por Fede el 01/11/2026; documentada en [PERF-05/06](../../04_decisions/QuantAgent-pf05-DC-flujos-y-capital-por-estrategia.md) | — |
| 39 | Diseño completo de la sección Backtests | — | Canvas | — | **01/11/2026**: primero el corte de Rendimiento con la lista mínima (casillas, hasta dos). Búsqueda, filtros y estados, después, cuando haya backtests suficientes. |
| 40 | Pestaña Rendimiento en Cuenta | Pestañas “Hoy · Rendimiento” | `featureFlags.accountPerformanceTab` | **01/11/2026**: habilitada con PERF-05; flujos no conciliados se informan en Rendimiento. |

## Decisiones tomadas (registro)

| Fecha | Decisión |
|---|---|
| 29/10/2026 | Ganancia/pérdida en azul/naranja con signo y flecha; nunca solo color. |
| 29/10/2026 | Velas verde/rojo (convención de mercado) con respaldo hueco/lleno; solo en velas. |
| 29/10/2026 | Horas siempre con zona: órdenes y mercado en ET; actualización de datos en ART; eventos en ambas. |
| 29/10/2026 | Formato es-AR (USD 98.412,37), menos tipográfico U+2212. |
| 29/10/2026 | Sin total general en Inicio (no mezclar paper con live). |
| 29/10/2026 | Geist (texto) + Geist Mono tabular (cifras). |
| 29/10/2026 | El código es la fuente de verdad de los tokens; el canvas, de la estructura de pantallas. |
| 30/10/2026 | v0.2: se muestran posiciones cortas (decisión #6 cerrada). Lado neutro: badge Larga/Corta con ícono y texto. |
| 30/10/2026 | v0.2: estados de orden del backend `cancelled` (dos L) y `expired`; "Vencida" con reloj de arena y borde punteado, distinta de "Cancelada". |
| 30/10/2026 | v0.2: cuenta sin datos = `status: 'no_data'`, nunca ceros. Reemplaza toda la pantalla Cuenta; tarjeta punteada en Inicio; "Sin datos todavía" en el selector. |
| 30/10/2026 | v0.2: `DataTable` genérica; `PositionRow` y `OrderRow` pasan a ser configuraciones de ella. |
| 01/11/2026 | v0.3: pantalla Rendimiento, un layout para backtest, ejecución y cartera; máximo dos sujetos (A continuo, B en guiones, aunque B sea SPY). |
| 01/11/2026 | v0.3: métricas anualizadas con muestra insuficiente = “no calculable” + motivo, nunca un número; el estado lo decide el backend. |
| 01/11/2026 | v0.3: todo gráfico tiene frase-resumen visible y alternativa en tabla con los mismos datos. |
| 01/11/2026 | v0.3.1: Backtests como sección propia; distribución mensual + diaria; cartera con estado “retorno no calculable” hasta que el backend registre aportes. |
| 01/11/2026 | v0.3.2: primer corte = Rendimiento de backtests. Cartera y estrategia en ejecución detrás de flags (`accountPerformanceTab`, `runPerformance`). Estrategia en ejecución: estado “rendimiento no aislable”. |
| 01/11/2026 | PERF-06: cada cuenta corresponde a una estrategia; `run` muestra el informe de la cuenta y #38 queda cerrada. |
