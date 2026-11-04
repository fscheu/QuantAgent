# Pantalla Rendimiento (v0.3)

Versión nativa de un tearsheet de QuantStats, de solo lectura: **todo número lo calcula el backend**; la web solo lo muestra y arma frases a partir de esos números.
Referencia visual: canvas, página **Rendimiento** (escritorio, teléfono, qué cambia por sujeto, estados y componentes nuevos).

---

## 1. Sujetos

| Tipo (`type`) | Qué es | Ejemplo |
|---|---|---|
| `backtest` | Una corrida histórica del motor de backtest | “Rotación sectorial por momentum 3M”, #bt-0412, 2015–2026 |
| `run` | Una estrategia en ejecución (paper o live), con su historia real | “Reversión a la media en ETFs grandes” en Paper M5 desde 14/07 |
| `portfolio` | Una cuenta con varias estrategias | Paper M4 |

Mismo layout para los tres. Lo que cambia está en la sección 7.

---

## 2. Rutas, entrada y salida

| Ruta | Sujeto |
|---|---|
| `/cuenta/:accountId/rendimiento` | `portfolio` (pestaña dentro de Cuenta) |
| `/rendimiento/run/:runId` | `run` |
| `/rendimiento/backtest/:backtestId` | `backtest` |

Parámetros de URL (compartibles y persistentes al recargar): `?range=ytd|1y|3y|5y|all` (por defecto `all`), `?vs=spy|none|backtest:<id>|run:<id>|portfolio:<id>` (por defecto `spy`), `?view=chart|table` por gráfico no se guarda en la URL (preferencia local).

**Cómo se llega**
1. **Cuenta** → pestañas “Hoy · Rendimiento”, **solo con el flag `featureFlags.accountPerformanceTab`** (apagado hasta que el backend registre aportes y calcule el retorno ponderado por tiempo; apagado = Cuenta sin pestañas). Pestañas (control segmentado con `role="tablist"`). Rendimiento de la cuenta = sujeto `portfolio`. No hay botón “volver”: se cambia de pestaña.
2. **Ficha de hipótesis** → botón “Rendimiento del backtest” (siempre). El botón “Rendimiento en paper/live” queda **oculto con el flag `featureFlags.runPerformance`** mientras no haya capital asignado por estrategia (#38): hoy el rendimiento de una estrategia no se puede aislar de su cuenta. No se promete pantalla para estrategias sueltas.
3. **Sección Backtests** (propia, en la navegación principal, ruta `/backtests`; los backtests existen sin Embudo y también se corren sueltos) → tocar una fila abre su Rendimiento. Desde la ficha de hipótesis se enlaza al backtest, pero la lista no vive adentro de Embudo. Con casillas (`DataTable`, máximo 2) aparece “Comparar 2 ›”, que abre `/rendimiento/backtest/<A>?vs=backtest:<B>`.
4. **Cartera** → la lista “Estrategias en la cartera” (con retorno por estrategia) también depende de `runPerformance`; con el flag apagado no se muestra.

**Cómo se vuelve**: enlace “‹ {origen}” arriba a la izquierda (44 px). Si se llegó desde la app, vuelve al origen real (historial). Si se entró por link directo, vuelve al padre lógico: `backtest` → lista de backtests; `run` → ficha de hipótesis; `portfolio` no tiene (es pestaña). En escritorio, además, una ruta de migas (`Embudo / Backtests / {nombre}`).

**Comparación (máximo dos sujetos)**
- El sujeto de la pantalla es **A**. “Comparar con” elige **B**: SPY (por defecto), nada, u otro sujeto (backtest, ejecución o cartera; recientes primero).
- A: línea continua `--qa-chart-1`. B: guiones `--qa-chart-2`, también cuando B es SPY. La sombra (solo `run` y `portfolio`) es la tercera serie, en puntos. Nunca más de 3 líneas.
- Métricas: columnas A y B. El grupo “Contra SPY” se calcula siempre contra SPY, para A y para B.
- Si los períodos no coinciden, todo se calcula sobre el **período común** y el encabezado lo dice (“Período común: ene 2018 – oct 2026”).
- Comparar más de dos está fuera de alcance.

---

## 3. Layout

**Escritorio (≥ 1024 px)**: barra lateral → migas → encabezado (izquierda: datos y frase; derecha: Período y Comparar con) → aviso de muestra corta (si aplica) → grilla `360px | 1fr`:
- Columna izquierda: 3 grupos de métricas → tabla “5 peores caídas” → tabla “Retorno por año contra SPY”.
- Columna derecha: Curva de valor → Retorno acumulado (lineal/log) → Caídas (underwater) → Mapa mensual → Retornos por año/mes → [Distribución | Métricas móviles] en dos columnas.
- Pie de convenciones a todo el ancho.

**Teléfono (< 1024 px)**: una columna: volver → encabezado (Período a todo el ancho, Comparar con a todo el ancho) → aviso → 3 grupos de métricas → gráficos apilados en el mismo orden → 5 peores caídas (tarjetas) → retorno por año (tabla de 4 columnas) → pie.
- Tooltip **fijo al tocar**: el valor queda en una franja arriba del gráfico (no tapa líneas) con botón “quitar” de 44 px.
- Todo control ≥ 44 px.

**Zoom 200 % y anchos chicos**: el layout responde al ancho disponible, no al dispositivo. Por debajo de 1024 px de ancho útil (por ejemplo escritorio con zoom 200 %) se usa la versión de una columna. El mapa mensual pasa a tarjetas por año con grilla 6 × 2 y, por debajo de 360 px útiles, 4 × 3. Las tablas pasan a tarjetas (`DataTable`). Los gráficos son SVG con `viewBox` y ancho 100 %, y las marcas de ejes se recalculan por ancho (menos marcas en angosto). Nunca hay scroll horizontal.

---

## 4. Bloques

### 4.1 Encabezado (`SubjectHeader`)
- Insignia de tipo (Backtest / En ejecución / Cartera) + entorno (Paper / Live; en backtest “Histórico · sin entorno”) + identificador (corrida, cuenta donde corre o “N estrategias”).
- Nombre (h1).
- Período: “ene 2015 – oct 2026 · 11 años y 10 meses” + “Último dato: 30/10/2026” (en `run`/`portfolio` con hora y zona) + cantidad de rebalanceos.
- **Frase-resumen** armada por el cliente con plantilla fija y números del backend:
  `En {período} subió|bajó {total_return} %; {B}, {b_total_return} %. La peor caída fue {max_dd} % (fondo en {mes}) y tardó {n} meses en recuperarse | todavía no se recuperó.` + en `run`/`portfolio`: `La sombra local va {gap} pp por encima|debajo.`
  Con muestra corta, la frase no menciona métricas anualizadas.

### 4.2 Aviso de muestra corta (`ShortSampleBanner`)
- Aparece si `sample.short` es verdadero. Tono **info** (es una limitación, no un problema), ícono reloj de arena.
- Dice: cuánto hay (“3,6 meses y 15 rebalanceos”), qué métricas **no se calculan** y cuándo se van a poder calcular, y cuáles se muestran pero **todavía no son confiables**.
- Variante “pocos rebalanceos” (ícono de ciclo) cuando hay suficientes días pero pocas decisiones.
- Al volver a una pantalla ya vista queda plegado en una línea (“Muestra corta · 7 métricas afectadas”); nunca desaparece mientras la muestra sea corta.

### 4.3 Métricas clave (3 × `MetricGroup`, 13 filas)

| Grupo | Métrica | Fórmula (backend) | Formato | Flecha | Mínimo para calcular | “Muestra corta” hasta |
|---|---|---|---|---|---|---|
| Rendimiento | Retorno acumulado | Π(1 + rᵢ) − 1, ponderado por tiempo | % · 1 dec | Sí | 21 días hábiles | — |
| | CAGR | (1 + R)^(252/n) − 1 | % · 1 dec | Sí | 252 días (12 meses) | — |
| | Meses positivos | meses completos con r > 0 / meses completos | % · 0 dec (o “2 de 3”) | No | 3 meses completos | 12 meses |
| | Mejor mes / Peor mes | máx./mín. de meses completos + mes | % · 1 dec + “abr 2020” | Sí | 2 meses completos | 12 meses |
| Riesgo | Volatilidad | σ(diaria) × √252 | % · 1 dec | No | 63 días | 252 días |
| | Sharpe | (μ − r_f) / σ × √252, diarios | 2 dec | No | 63 días | 252 días |
| | Sortino | (μ − r_f) / σ_bajista × √252 | 2 dec | No | 63 días | 252 días |
| | Drawdown máximo | mín(Vₜ / máx previo − 1) + inicio, fondo, recuperación y duración | % · 1 dec + “ene → jul 2020 · 6 meses” | Sí | 21 días | — |
| | Calmar | CAGR / \|drawdown máximo\| | 2 dec | No | 252 días | — |
| Contra SPY | Alfa | intercepto anualizado de la regresión de excesos diarios contra SPY | pp · 1 dec | Sí | 252 días | — |
| | Beta | cov(r, r_SPY) / var(r_SPY), diarios | 2 dec | No | 63 días | 252 días |
| | Retorno relativo | (1 + R) / (1 + R_SPY) − 1 | % · 1 dec | Sí | 21 días | — |

- **El backend decide el estado** de cada métrica (`ok`, `short_sample`, `not_computable` + `reason`); la web no recalcula nada.
- **Los umbrales son parámetros provisorios, no diseño fijo.** Tres escalones de confianza: 21 días = solo descriptivo (retorno, drawdown, relativo); 63 días = métricas de riesgo con aviso de muestra corta; 252 días = anualizadas. Aviso de pocos rebalanceos por debajo de 12 (razonable para una estrategia mensual). Viven como parámetros en el backend, que los devuelve en `sample.thresholds` para que los textos (“Necesita 12 meses; hay 3,6”) usen los valores vigentes; `performanceConfig` en `config.ts` es solo el espejo por defecto.
- `not_computable`: “—” + “no calculable” + motivo (“Necesita 12 meses; hay 3,6”). Nunca un número.
- `short_sample`: valor en texto secundario + marca “muestra corta”.
- Cada fila tiene su “?” (`HelpTip`, claves en `help-texts.es.json`).

### 4.4 Gráficos (todos dentro de `ChartCard`)
Cada tarjeta: título + “?” si aplica · controles · conmutador **Gráfico | Tabla** · **frase-resumen siempre visible** (también es el `aria-label` del SVG) · gráfico · leyenda con muestra de trazo. Estados: cargando (esqueleto), vacío (“Sin datos para 3A: empezó el 14/07/2026. Probá 1M o Todo”), error dentro de la tarjeta.

| Gráfico | Componente | Controles | Frase-resumen (plantilla) | Tabla alternativa |
|---|---|---|---|---|
| Curva de valor vs B y sombra | `EquityChart` (existente) | Usa el Período del encabezado (sin selector propio en esta pantalla) | “USD {inicio} de {fecha} terminan en USD {fin}; en {B}, USD {finB}.” (+ reserva en backtest) | Fecha · A · B · Sombra |
| Retorno acumulado | `CumulativeChart` + `scale` | Lineal / Log | “Acumulado: A {x} %, B {y} %.” | Fecha · A % · B % |
| Caídas (underwater) | `DrawdownChart` | — | “Peores caídas: 1) … 2) … 3) …” | Fecha · A % · B % |
| Mapa mensual | `MonthlyHeatmap` | Mapa / Tabla | “{k} de {n} meses positivos. Mejor: … peor: …” | Año · 12 meses · Año · B |
| Retornos por año o mes | `ReturnsBars` | Mes / Año | “{k} de {n} años completos positivos. Mejor … peor …” | Período · A · B · diferencia |
| Distribución | `ReturnsHistogram` | **Mensual** (por defecto) / Diaria. Diaria es secundaria y se habilita desde 63 días hábiles (deshabilitada con motivo antes); agrega VaR 95 %, CVaR y sombrea la cola izquierda | Mensual: “Promedio mensual {m} %. {k} de {n} meses entre −2 % y +4 %; el peor fue …” · Diaria: “1 de cada 20 días cae más de {VaR} % (VaR 95 %); en esos días, la caída promedio es {CVaR} %.” | Rango · cantidad A · cantidad B (+ VaR y CVaR en diaria) |
| Métricas móviles | `RollingChart` | Sharpe / Volatilidad · ventana 6M / 12M | “Sharpe a 12 meses hoy: {x} (B {y}). El mínimo fue {z} en {mes}.” | Fecha · A · B |

### 4.5 Tablas
- **Las 5 peores caídas**: # · inicio (último máximo) · fondo · recuperación (o “sin recuperar”, en advertencia) · caída % · duración. Las 3 primeras están numeradas en `DrawdownChart`.
- **Retorno por año contra SPY**: año (con * si es parcial) · A · SPY · diferencia en pp con flecha.

### 4.6 Pie (`CalcConventions`)
Frecuencia (diaria), método (ponderado por tiempo; en cartera, depósitos y retiros excluidos), anualización (252), tasa libre (fuente + promedio del período), benchmark (SPY con dividendos), fuente de datos (motor de backtest / precios del broker), costos (simulados en backtest; reales en ejecución) y enlace al manual.

---

## 5. Estados de la pantalla

| Estado | Cuándo | Qué se ve |
|---|---|---|
| Cargando | Sin respuesta todavía | Encabezado y etiquetas reales; valores y gráficos en esqueleto estático |
| Sin datos suficientes | `status: 'insufficient_data'` (menos de 21 días hábiles) | “Faltan N días hábiles para el primer informe”, barra de progreso (`role="progressbar"`), fecha estimada. Sin métricas ni gráficos |
| Error | La API falla | Mensaje + qué hacer + reintento automático (30 s) + “Reintentar ahora”. Si falla una sola sección, el error queda en su tarjeta |
| Desactualizado | `stale: true` (`run`/`portfolio`) | Aviso punteado en advertencia con antigüedad y hora; cifras en texto secundario sin color semántico (conservan signo y flecha) |
| Sin ejecuciones | `status: 'no_runs'` (`run` aprobado sin días operados) | “Todavía no empezó a operar”, cuenta, fecha de aprobación, próximo rebalanceo y enlace al backtest |
| Muestra corta | `sample.short: true` | Pantalla completa + `ShortSampleBanner` + métricas marcadas |
| Retorno no calculable: aporte sin registrar | `status: 'returns_not_computable'`, `not_computable_reason: 'unregistered_flows'` (cartera, con el flag encendido y algún movimiento en `flows.unreconciled`) | “Retorno no calculable: hay 1 aporte sin registrar” + qué movimiento (fecha y monto) + “cuando se registre, el informe se calcula solo”. Sin métricas ni gráficos de rendimiento. Solo “Valor de la cuenta” en USD, marcado “incluye aportes”, sin % ni color de ganancia/pérdida, con el salto marcado. Sin registro de flujos no se llega acá: la pestaña está oculta |
| Rendimiento no aislable | `status: 'returns_not_computable'`, `not_computable_reason: 'capital_not_isolated'` (estrategia en ejecución) | “Rendimiento no aislable”: comparte posiciones y caja con la cuenta; no hay capital por estrategia. Enlaces al backtest y a la cuenta. Sin números. El acceso está oculto (`runPerformance`); el estado solo cubre un link directo |

Cartera sin datos usa el estado “cuenta sin datos” de v0.2.

---

## 6. Accesibilidad
- Toda variación: signo, flecha y color. Nada distinguido solo por color: series con trazo propio, heatmap con número en la celda, barras con ▲/▼, bandas de caídas numeradas.
- Cada gráfico: `role="img"` + `aria-label` = frase-resumen; alternativa en tabla real (`DataTable`) con el mismo contenido.
- Controles reales (`button`, `role="radiogroup"`/`tablist`), foco visible, 44 px mínimo, `prefers-reduced-motion`.
- Contraste: solo tokens `--qa-*`.

---

## 7. Qué cambia por sujeto

| Elemento | Backtest | En ejecución | Cartera |
|---|---|---|---|
| Insignia | Backtest | En ejecución | Cartera + “N estrategias” |
| Entorno | “Histórico · sin entorno” | Paper/Live + cuenta | Paper/Live |
| Período | Inicio–fin de la corrida | Desde el primer día operado | Desde la apertura |
| Último dato | Fin de la corrida (no se desactualiza) | Fecha y hora (puede desactualizarse) | Ídem |
| Sombra | No | Sí (tercera serie) | Sí |
| Reserva fuera de muestra | Franja sombreada en curva y caídas | No | No |
| Aviso de muestra corta | Raro | Frecuente al inicio | Frecuente al inicio |
| Volver a | Sección Backtests / ficha (según origen) | Ficha de hipótesis | (pestaña de Cuenta) |
| Retorno ponderado por tiempo | No aplica | Hoy no aislable: estado “rendimiento no aislable”, acceso oculto hasta definir capital por estrategia | Pestaña oculta hasta que exista registro de aportes y TWR; después, estado “1 aporte sin registrar” si falta alguno |
| Bloque extra | Configuración y costos de la corrida | Enlace al backtest original | Lista de estrategias + nota de depósitos/retiros |
| Estado “sin ejecuciones” | No aplica | Sí | Usa “cuenta sin datos” |

---

## 8. Datos que necesita cada elemento (contrato para el backend)

Un endpoint por sujeto, todo precalculado: `GET /api/performance/{type}/{id}?range={range}&align_with={type}:{id}` → `PerformanceReport` (ver `api-types.ts`). Para comparar, la web pide los dos informes con `align_with` cruzado para que ambos usen el período común. SPY como B: el informe de A ya trae `benchmark_metrics` y las series de SPY.

| Elemento | Campos |
|---|---|
| Encabezado | `subject.{type,id,name,env,account_id,hypothesis_id,run_label,strategies}` · `period.{start,end,trading_days,complete_months,rebalances}` · `as_of` · `stale` · `headline.{total_return,benchmark_total_return,max_drawdown,max_dd_trough,max_dd_recovery_days,shadow_gap_pp}` |
| Período | `range` · `available_ranges` (para deshabilitar 3A/5A con motivo) |
| Aviso de muestra corta | `sample.{short,reasons,not_computable,unreliable,months_until_full}` |
| Métricas (A y SPY) | `metrics[key]` y `benchmark_metrics[key]` = `{value,status,reason,detail}`; `detail` de drawdown: `{peak_date,trough_date,recovery_date,days}`; de mejor/peor mes: `{month}` |
| Curva de valor | `series.equity[] = {date,value,benchmark,shadow}` (diaria; el backend puede bajar a semanal para > 3 años con `resolution`) · `subject.out_of_sample_start` |
| Retorno acumulado | Se deriva de `series.equity` (sin endpoint extra) |
| Caídas | `series.drawdown[] = {date,dd,benchmark_dd}` · `worst_drawdowns[0..2]` para las bandas |
| Mapa mensual | `monthly_returns[] = {year,month,ret,benchmark_ret,complete}` |
| Barras por año/mes | `annual_returns[] = {year,ret,benchmark_ret,partial}` + `monthly_returns` |
| Distribución | `distributions.monthly.{bin_width_pct,bins[{from,to,count,benchmark_count}],mean,benchmark_mean,n}` · `distributions.daily` (igual + `var_95`, `cvar_95`; `null` con menos de 63 días) · `distributions.daily_reason` |
| Umbrales vigentes | `sample.thresholds.{descriptive_days,risk_days,annualized_days,min_rebalances}` |
| Flujos (cartera) | `flows.{registered,unreconciled[{date,amount}]}` · `series.equity` en USD se manda igual en `returns_not_computable` |
| Métricas móviles | `rolling.{window_days,points[{date,sharpe,volatility,benchmark_sharpe,benchmark_volatility}]}` para 126 y 252 días |
| Peores caídas | `worst_drawdowns[] = {rank,peak_date,trough_date,recovery_date,depth,days_to_trough,days_to_recovery,total_days}` (5) |
| Retorno por año | `annual_returns` |
| Cartera | `subject.strategies[] = {run_id,name,total_return}` · `cash_flows[] = {date,amount}` |
| Estados | `status: 'ready'|'insufficient_data'|'no_runs'|'returns_not_computable'` · `insufficient.{days_have,days_needed,first_report_estimate}` · `no_runs.{approved_at,account_id,next_rebalance_at}` |
| Pie | `conventions.{frequency,method,annualization_days,risk_free_source,risk_free_avg,benchmark,data_source,costs_bps,manual_url}` |

Lista de backtests (sección propia y selector de comparación; en el primer corte, lista mínima con casillas para comparar hasta dos, sin búsqueda ni filtros): `GET /api/backtests?hypothesis_id=&q=` → `{id,label,hypothesis_name|null,start,end,total_return}[]` (`hypothesis_name` puede ser `null`: hay backtests sueltos).

---

## 9. Prerrequisitos de backend

0. **Flags**: `accountPerformanceTab` se enciende cuando esté el punto 1; `runPerformance`, cuando se cierre el punto 4.
1. **Registro de aportes y retiros por cuenta** (ticket previo, junto con QW-04, que ya cubre depósitos en los tests de contabilidad). Hoy `EquitySnapshot` guarda equity, cash e invested sin flujos y sin id de cuenta: un depósito se vería como ganancia. Hace falta: id de cuenta en los snapshots, tabla de flujos (fecha, monto, tipo) y cálculo de retorno ponderado por tiempo. Hasta que esté, la cartera muestra el estado `returns_not_computable`; los sujetos `backtest` y `run` no dependen de esto (en `run`, a confirmar según cómo se asigne capital).
2. **Umbrales como parámetros** del cálculo de rendimiento (21 / 63 / 252 días, 12 rebalanceos), devueltos en `sample.thresholds`.
3. **Distribución diaria con VaR y CVaR** históricos al 95 %, disponible desde 63 días.
4. **Capital por estrategia** (pregunta abierta para Fede en el ticket de backend): hoy `models.py` no tiene estrategia ni capital asignado; posiciones y caja son de la cuenta, y `cartera_paper.py` reparte pesos por símbolo sobre una sola caja. Hasta decidir cómo se asigna capital, el rendimiento de una estrategia en ejecución no se puede aislar.

**Primer corte de implementación**: Rendimiento de **backtests** (incluida la comparación de dos) + lista mínima de Backtests con casillas. Cartera y estrategia en ejecución quedan detrás de sus flags.
