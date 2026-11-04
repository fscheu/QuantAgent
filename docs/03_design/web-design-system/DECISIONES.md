# Decisiones abiertas

Cada fila tiene un valor provisorio para no bloquear el desarrollo. Cuando se decida, completar la última columna y actualizar el archivo indicado.

| # | Duda | Valor provisorio | Dónde vive | Se decide cuando… | Decisión |
|---|---|---|---|---|---|
| 1 | Nombres y función de las etapas 0–3 | — | `config.ts`, `help-texts.es.json` | — | **Cerrada 03/11/2026**: E0 Admisión, E1 Filtro, E2 Validación en el motor (barra a barra contra control independiente), E3 Holdout. Después: Sombra local → Paper → Candidata como “fase posterior”. La reserva 2023+ no es etapa (chip en la ficha). |
| 2 | Motivos de descarte (salvo los 3 dados) | Ejemplos del canvas agrupados por `cause_code`; texto corto + `Measure` (valor vs umbral) | Backend | El motor del embudo los emita | — |
| 3 | Métrica que muestra cada hipótesis | `last_metric` nulable (nula en la mayoría); se muestra solo si viene, con fecha | Backend (`last_metric`) | Se definan las métricas por etapa | — |
| 4 | Umbrales del stop | cerca < 2 %, atención < 5 %, escala 15 % | `config.ts` | Después de ver stops reales | — |
| 5 | Distancia en % o en ATR | % | `config.ts` | Idem | — |
| 6 | Posiciones cortas | — | `stopDistance()` | — | **Cerrada 30/10/2026**: se muestran. Barra con eje de precio (corta crece desde la derecha), distancia (stop − precio)/precio, mismas zonas, badge Larga/Corta, `side` en `Position`. |
| 7 | Orden por defecto de posiciones | Distancia al stop ascendente | Pantalla Cuenta | 2 semanas de uso | — |
| 8 | Rechazadas siempre arriba | Sí, también con "Todas" | Pantalla Cuenta | 2 semanas de uso | — |
| 9 | Ciclo por defecto en Embudo | Todos = todos los ciclos mensuales (sin el historial, que va en “Sin ciclo”) + período Todo | `config.ts` | Prueba con datos reales | — |
| 10 | Hipótesis en escritorio: tarjetas o tabla | Tarjetas hasta 30 (`funnelConfig.cardsUpTo`), `DataTable` con más; con el máximo de 5 por ciclo se llega a 30 en ~6 meses | Pantalla Embudo | Más de 30 en el filtro activo | — |
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
| 37 | Cartera: retorno ponderado por tiempo | — | Backend | — | **01/11/2026**: hoy NO se calcula (`EquitySnapshot` sin flujos ni id de cuenta). Prerrequisito de backend junto con QW-04. Mientras tanto, estado `returns_not_computable`. |
| 38 | Cómo se asigna capital por estrategia (para aislar su rendimiento) | Sin capital por estrategia: estado “rendimiento no aislable” y acceso oculto (`runPerformance` apagado). No se promete pantalla para estrategias sueltas | Backend (`models.py`, `cartera_paper.py`) | **Decide Fede**; pregunta anotada en el ticket de backend | — |
| 39 | Diseño completo de la sección Backtests | — | Canvas | — | **01/11/2026**: primero el corte de Rendimiento con la lista mínima (casillas, hasta dos). Búsqueda, filtros y estados, después, cuando haya backtests suficientes. |
| 40 | Pestaña Rendimiento en Cuenta | — | `featureFlags.accountPerformanceTab` | — | **01/11/2026**: oculta detrás de flag hasta que el backend calcule TWR. El estado “1 aporte sin registrar” queda diseñado para después. |
| 41 | ¿ERROR y ALARMA entran en el chip “Descartadas”? | Sí, con su badge propio y borde punteado/lleno; nunca suman a “cayeron” (van como “detenidas”) | Pantalla Embudo | Uso real | — |
| 42 | ¿“Todos” incluye el historial sin ciclo? | No: Todos = ciclos mensuales; el historial solo con “Sin ciclo” | `funnelConfig` | Ver si se extraña verlo junto | — |
| 43 | Qué filtra “Período” (30d / 90d) en Embudo | Hipótesis con actividad (`updated_at`) en la ventana; el recibo de un ciclo no cambia | Backend | Al implementar `/api/funnel` | **04/11/2026 (Fede): eventos de etapa en los últimos N días; una hipótesis aparece si tuvo alguno y su estado es el del último dentro de la ventana. Rótulo “con actividad en los últimos N días”.** |
| 44 | Historial importado: ¿sus 55 etapas se mapean a E0–E3? | Supuesto: sí (el canvas lo dibuja así) | Backend (importación) | **Confirmar con el registro real** | **04/11/2026 (Fede): las 55 son eventos de etapa E0–E3 (no hipótesis), sin ciclo e importados. Los 202 intentos tienen la etapa en texto libre: no se asumen E0–E3. “Sin ciclo” muestra tabla de eventos + intentos por etiqueta, sin embudo.** |
| 45 | Mínimo para dibujar el embudo | 2 hipótesis (con 1, recorrido + aviso) | `funnelConfig.minForChart` | Uso real | — |
| 46 | Nombre del comando de línea que se menciona como ayuda | Texto genérico “qa ciclo …” | Copy | Confirmar el comando real | **04/11/2026 (Fede): quantagent registro ciclo abrir 2026-11 H1 H2 · ciclo cerrar 2026-11 · ciclo informe 2026-11 · quantagent registro corte.** |
| 47 | Significado de cada estado del motor | Risk-on: compras nuevas permitidas · Neutral: compras nuevas solo en sectores habilitados · Risk-off: sin compras nuevas (dado) | Backend (`MotorState.meanings`) | Lo define el motor | **04/11/2026 (Fede): el backend manda state_description. Definiciones: risk-off = sin compras nuevas; neutral = estrés macro elevado sin que SPY esté en risk-off; risk-on = el resto.** |
| 48 | Umbrales del mapa de calor de sectores por plazo | 1d ±0,25/0,75/1,5 · 1s ±0,5/1,5/3 · 1m ±1/3/6 · 3m ±2/5/10 · Año ±3/8/15 (%) | `marketConfig.heatThresholds` | Ver con datos reales | — |
| 49 | Colores del régimen | Alcista azul (chart-1), bajista naranja (chart-2), rayado = volatilidad alta, ▲/▼ escrito; coincide a propósito con ganancia/pérdida | Tokens existentes | Uso real | — |
| 50 | Macro por defecto: 5A en teléfono, Todo en escritorio | Así (en 5A no hay recesiones; se aclara con una nota) | `marketConfig.defaultMacroRange` | Uso real | — |
| 51 | Detalle de ETF en escritorio: panel lateral o página | Panel lateral de 560 px (misma lógica que el recibo) | Pantalla Mercado | Uso real | — |
| 52 | “En cartera”: qué cuentas | Todas las cuentas paper con posición abierta hoy en ese símbolo; enlaza a la cuenta | Backend (`EtfRow.held_in`) | Al implementar | — |
| 53 | ¿Mercado también muestra “En cartera” en sectores? | No (el brief lo pide solo en los 29 ETF) | Pantalla Mercado | Si Paper M4 opera sectores, conviene sumarlo | **04/11/2026 (Fede): sí, “En cartera” también en la tabla de sectores, con enlace a la cuenta.** |
| 55 | ¿Qué sectores rankea el motor? | — | Backend (motor M5) | — | **04/11/2026 (Fede)**: los 11 SPDR (universo M5 de 14 = 11 + SPY, EEM, EWZ; `catalogo/m5-priors.yaml`). Los 29 son el universo de M4. La admisión acepta activos de ambos. “En cartera” sale de posiciones reales, no del ranking. Estados: habilitado con posición / sin posición todavía. |
| 56 | Evento del feed al llegar a 6 | — | EVENTOS.md | — | **04/11/2026 (Fede)**: `review_rule_reached`, informativo, una vez por cierre, enlaza a Embudo. |
| 57 | Regla de “dato viejo” en macro | — | `marketConfig.macroStale` | — | **04/11/2026 (Fede)**: fecha esperada del calendario FRED + 2 días hábiles; sin calendario, 1,5 × frecuencia. Notice stale + cifra atenuada. |
| 58 | Universo de 29 ETF | — | `marketConfig.universe` | — | **04/11/2026 (Fede)**: SPY QQQ IWM · 9 sectores · EFA EEM EWJ EWZ FXI · SHY IEF TLT TIP AGG LQD · GLD SLV DBC USO · FXE · VNQ. Sin efectivo. |
| 59 | Período y ciclo en Embudo | — | `CycleFilter` | — | **04/11/2026 (Fede)**: el período aplica solo al embudo y a la lista (rótulo “Período: aplica al embudo y a la lista”); la tarjeta del ciclo actual y la regla no dependen de él; con un ciclo concreto el período se deshabilita. |
| 60 | FXE | — | Mercado | — | **04/11/2026 (Fede)**: solo en la tabla de ETF, sin bloque propio. FXE es el euro, no el dólar amplio (`DTWEXBGS`, tarjeta “próximamente”); cuando se active, se enlaza desde la fila de FXE. |
| 54 | Inflación y demás indicadores macro | Diseñados como fantasma “próximamente”; solo T10Y2Y activa | `marketConfig.macroSeries` | Cuando se sumen las series al backend | — |

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
| 03/11/2026 | v0.4: Embudo rediseñado sobre el registro real: ciclos mensuales `YYYY-MM` (abierto/cerrado con recibo inmutable), etapas E0–E3 + fase posterior, veredictos PASA/CORTA/ERROR/ALARMA, promovida = pasó E3. Decisión #1 cerrada. |
| 03/11/2026 | v0.4: ERROR y ALARMA se muestran aparte (“detenidas”), nunca como caídas. |
| 03/11/2026 | v0.4: regla de revisión (6 ciclos cerrados seguidos sin promovida) siempre visible en Embudo; es de proceso y queda fuera del estado general de Inicio. |
| 03/11/2026 | v0.4: Mercado = contexto diario al cierre con snapshots propios; sin recomendaciones ni precios en vivo. Embudo y Mercado activos en la navegación. |
| 03/11/2026 | v0.4: sin tokens nuevos; solo clases utilitarias (`.qa-seg-error`, `.qa-seg-alarm`, `.qa-ghost`, `.qa-reg-*`) sobre variables existentes. |
| 04/11/2026 | v0.4.1: respuestas de Fede aplicadas (historial = eventos, período por actividad, `state_description`, comandos reales, “En cartera” en sectores, universo real de 29, evento de feed a los 6 ciclos, regla de dato macro viejo). |
| 04/11/2026 | v0.4.2: sectores con los 11 rankeados (motor de M5), “En cartera” por posiciones reales (M5 en efectivo), estados habilitado con/sin posición; período solo para embudo y lista; FXE sin bloque propio. |
