# Comparación crítica: 5 repos externos vs QuantAgent (roadmap post-M5)

Fecha: 2026-10-30. Solo análisis, sin cambios de código. Cada repo fue leído por una corrida de Codex (gpt-6.1-sol, esfuerzo high) contra `2026-10-28_roadmap_post_m5.md`; el contenido de los repos se trató como dato no confiable y no se ejecutó nada. Los análisis completos están en `anexos_repos_externos_2026-10-30/` (en inglés). Este documento es la síntesis.

## Veredicto corto

Ninguno de los cinco reemplaza ni acelera el núcleo de QuantAgent (embudo de hipótesis, admisión, DSR, snapshots Parquet). En eso estamos **por delante** de todos. Lo que sí sirve es chico y concreto:

1. **quantstats** para reportes de backtest y paper (la mejor adopción: una dependencia, ganancia inmediata).
2. Un **"recibo" de candidato** (qué hipótesis, qué código, qué datos, qué costos) tomado de ai-hedge-fund y tradesight.
3. Vocabulario de **evidencia y frescura** en la web (tradesight, QuantDinger, TradingAgents).
4. Un puñado de **tests de contabilidad** (tradesight, ai-hedge-fund).

Lo demás es sobreingeniería para nuestra escala (cuenta de USD 2.500, 5 hipótesis por mes).

## Qué es cada repo, en una línea

| Repo | Qué es | Madurez / lectura honesta |
|---|---|---|
| ai-hedge-fund | "Fondo" simulado con personas LLM (estilo Buffett, etc.) + backtest + paper local | ~64k estrellas, MIT, prueba de concepto educativa. Sin validación estadística (no hay walk-forward, ni DSR). Datos: un único proveedor pago (Financial Datasets) |
| TradingAgents | Debate de agentes LLM (bull/bear/riesgo) que emite una recomendación por ticker | Evalúa *recomendaciones*, no cuentas reales. Sin hipótesis formales. Datos: yfinance, Alpha Vantage, EDGAR, FRED |
| quantstats | Librería de métricas y reportes ("tearsheet") a partir de una serie de retornos | Chica, mantenida, CI con regresiones numéricas. Es lo único que es *componente*, no sistema |
| QuantDinger | Plataforma SaaS completa: IDE de estrategias con LLM, backtest propio, brokers, Kafka, MCP, marketplace | Enorme. Casi todo es producto, no investigación. Genera código con LLM y optimiza hasta que "da bien" |
| tradesight | Torneo de estrategias + optimizador + paper en Alpaca + dashboard Flask | Tiene bugs de contabilidad (la curva de equity omite el capital invertido). La "IA" es mutación aleatoria de código con `exec()` |

## Fuentes de datos históricos

Lo que encontró cada uno, y qué nos dice:

- **ai-hedge-fund**: solo Financial Datasets (API paga), con caché JSON. Filtra fundamentales por fecha de presentación (bien), pero no ajusta por splits/dividendos por su cuenta. Aporta una fuente curiosa: Kenneth French (breakpoints de capitalización).
- **TradingAgents**: yfinance (gratis, no oficial), Alpha Vantage (25 pedidos/día gratis; su adaptador usa un endpoint premium), SEC EDGAR (gratis, cobertura limitada), FRED/ALFRED (gratis, con "vintages"). Cache mutable de 5 años: **no sirve para validar históricos**.
- **QuantDinger**: abanico enorme (Yahoo, Finnhub, Twelve Data, Tiingo, FRED, BLS, BEA, GDELT, cripto). La mayoría irrelevante para ETFs/acciones US.
- **tradesight**: Alpaca feed IEX (gratis pero cobertura parcial) + yfinance. Alpaca consolidado cuesta USD 99/mes, o sea ~47% de un capital de USD 2.500: no se justifica.
- **quantstats**: descarga benchmark de Yahoo si le pasás un ticker. Hay que evitarlo y pasarle series propias.

**Conclusión para M-datos:** ninguno tiene algo mejor que snapshots Parquet inmutables con manifiesto. Nuestro diseño ya es el más serio. Lo único para robar: separar en cada campo **fecha de observación / fecha de publicación / fecha de descarga**, y distinguir "dato faltante" de "proveedor caído" de "vacío conocido". Y mantener la decisión del roadmap de refresco mensual completo (no construir ahora la ingesta diaria con corporate actions).

Fuentes gratuitas útiles que aparecieron y ya están en nuestro radar: FRED/ALFRED (macro con vintages), SEC EDGAR (fundamentales, solo para un prototipo acotado), Kenneth French (factores). Pagas: Sharadar (USD 499/año, ~20% del capital: decidir después de la prueba de cuenta chica, no antes).

## Ciclos para generar y testear hipótesis

Esto es lo que más te interesaba. Resultado: **nadie tiene un ciclo mejor que el nuestro.**

| | Cómo "genera ideas" | Cómo valida | Anti-sobreajuste |
|---|---|---|---|
| QuantAgent | Hipótesis YAML con racional, admisión, postmortems estructurales | Filtro walk-forward, validación detallada, veredicto, DSR, holdout con permisos | Sí: cuenta de intentos, reserva, regla de parada |
| ai-hedge-fund | Combinar personas a mano | Backtest + paper manual | No (el propio repo lo lista como "planeado") |
| TradingAgents | El LLM razona sobre un ticker | Mide hit-rate de recomendaciones vs SPY | No |
| QuantDinger | LLM escribe Python → backtest → optimizador (TPE propio) | Ranking por retorno, "OOS" elegido por retorno, nota de robustez | Débil: calibra umbrales automáticamente (lazo de optimización sin registro) |
| tradesight | Mutación aleatoria de código, parámetros de 4 familias | Split 70/30, cross-símbolo, Monte Carlo, penalización heurística | Parcial, pero reemplaza al campeón tras pérdidas |

**Ideas aprovechables (chicas):**
- tradesight: separar la **observación diaria barata** de la **búsqueda cara**, y dejar un recibo "no se buscó porque…". Encaja con tu ciclo mensual: cada ciclo deja un recibo aunque no promueva nada.
- QuantDinger: chequear restricciones de parámetros *antes* de simular; cargar el dataset una sola vez por trabajo.
- ai-hedge-fund: "deployment snapshot", un registro inmutable de lo desplegado.
- TradingAgents: reanudación idempotente (si el ciclo se cae, retoma sin repetir).

**A evitar:** "si pierde, reemplazar por el desafiante" (tradesight), notas de robustez con letra A–F que promedian chequeos (QuantDinger), y agregar debates bull/bear a cada hipótesis sin evidencia de que mejoren la selección (TradingAgents).

## Componentes de la comunidad: qué usar en vez de construir

| Necesidad | Opción | Estado en QuantAgent | Recomendación |
|---|---|---|---|
| Métricas y reporte de performance | **quantstats** | No lo usamos | **Adoptar** (ver abajo) |
| Optimización de parámetros (si hace falta) | **Optuna** | No | Solo si aparece la necesidad; QuantDinger escribió su propio TPE, no repetir |
| Indicadores técnicos | TA-Lib (ya está), stockstats | TA-Lib ya declarado | Consolidar en TA-Lib; no sumar otra librería |
| Backtest | vectorbt + Backtesting.py | Ya los usamos | No cambiar; ni QuantDinger ni tradesight tienen un motor superior |
| Calendario de mercado | pandas_market_calendars | Ya está | No agregar exchange-calendars |
| Markdown a HTML (reportes) | markdown-it-py | No | Opción para reportes de ciclo (activar el modo seguro: por defecto permite HTML crudo) |
| MCP read-only | SDK oficial MCP de Python | Planeado | Usar el SDK directo; **no** copiar el MCP de QuantDinger (tiene órdenes y operaciones de emergencia) |
| Checkpoints de ejecución | LangGraph checkpointer | LangGraph ya está | Sirve para reanudar, pero el registro de hipótesis va en nuestra base |

Ninguno de los repos esconde un motor maduro que podamos adoptar entero. Importar cualquiera completo crearía un segundo stack de trading.

## quantstats: el reporte, en concreto

Es la recomendación más fuerte. Se le pasa una **serie de retornos diarios** (más una de benchmark opcional) y genera un HTML con:

- **~50 métricas en la tabla básica**: período, retorno acumulado, CAGR, Sharpe, Sortino, Calmar, drawdown máximo y duración, ratio ganancia/dolor, factor de recuperación, Ulcer Index, retornos MTD/3M/6M/YTD/1Y.
- **Extras en el reporte completo**: volatilidad, asimetría, curtosis, VaR/CVaR, Kelly, mejores/peores días y meses, rachas, % de días/meses ganadores. Con benchmark: alfa, beta, correlación, R², information ratio.
- **Tablas**: 10 peores drawdowns, retorno por año contra benchmark.
- **14 gráficos**: retorno acumulado (normal y log), igualado por volatilidad, retornos por año, histograma mensual, retornos diarios, beta/volatilidad/Sharpe/Sortino móviles, drawdowns resaltados, curva de drawdown ("underwater"), heatmap mensual, boxplots por frecuencia.

**Cómo encaja:** `quantagent/backtesting/export.py` ya exporta la curva de equity (timestamp, equity, cash, drawdown). Hay que convertirla en retornos conservando el capital inicial. Para paper, la fuente son los `EquitySnapshot`.

**Trampas que encontró Codex (importantes):**
- Con 3 meses de paper igual muestra filas "3Y/5Y/10Y" calculadas con lo que haya: engañoso, hay que ocultarlas.
- "Time in Market" es solo un proxy; "Risk of Ruin" no mide riesgo de tu cuenta; el VaR por defecto asume distribución normal.
- Sharpe: nuestro backtest usa 2% de tasa libre, quantstats 0, y `sharpes_en_exceso()` usa DGS3MO. Si no se unifica, dos "Sharpe" distintos se presentan como lo mismo.
- No usar el HTML de varias estrategias (bug de embebido); un reporte por serie.
- No es para decidir promociones: las decisiones siguen en `fabrica/informe.py` y los criterios propios. quantstats es **apéndice descriptivo**.
- El tipo `EquitySnapshot` tiene `snapshot_date` único y sin id de cuenta: con varias carteras compartiendo tabla se rompe (hoy lo evitamos con una base por cuenta).
- Dependencias: mete seaborn y exige yfinance aunque se use offline. Importarlo solo en el comando de reporte, versión fija.
- El layout HTML es de escritorio: para el celular, enlazar el archivo, no incrustarlo en la web.

## Qué adoptar (ordenado por valor / esfuerzo)

1. **Adaptador equity→retornos + reporte quantstats** por candidato promovido y por cartera paper (reporte periódico). Con tests chicos de contabilidad (retorno compuesto conocido, pérdida en el primer período, depósitos, fechas faltantes). Baja complejidad, valor alto y visible.
2. **Recibo de candidato / de ciclo**: un registro con hipótesis+versión, hash de código, hash de snapshots, universo, costos, política de ejecución, supuestos de cuenta. Se reutiliza en reportes, shadow y API. Incluye "ciclo sin promoción" como resultado válido. Vive en nuestra persistencia, sin plataforma de eventos.
3. **Corrida de sensibilidad de cuenta chica (USD 1.000 / 2.500 / 5.000)** en el motor actual, contra SPY ejecutable. Ninguno de los cinco lo resuelve (ai-hedge-fund compra 0 acciones de un papel de USD 600 con 10% de peso, y no tiene fraccionarias ni liquidación). Hacerlo **antes** de pagar datos.
4. **Vocabulario de evidencia en la web**: campos separados `fuente`, `modo_de_trading`, `estado_de_verificación`, `observado_en`, `versión_de_datos`; vista "por qué esta operación" basada en lo guardado (tradesight lo hace bien: si no se guardó la razón, lo dice, no la inventa).
5. **Códigos de motivo de ejecución** de QuantDinger: unidad mínima, caja insuficiente, ejecución parcial, nocional mínimo, redondeo.
6. **Defensas de datos** de TradingAgents: el alcance de fechas lo fija el contexto de la corrida, no el agente; registrar qué proveedor respondió.

## Qué evitar

**En los repos (no copiar):** personas LLM como "diversificación" (mismos inputs, no señales independientes); jerarquía fondo/pod/CIO antes de tener alfa; Kafka, SaaS, marketplace, IDE de estrategias en navegador; mutación de código con `exec()`; reemplazo automático de campeón; notas de robustez promediadas; filtro de IA fail-open como autoridad de riesgo; MCP con órdenes.

**En nosotros (sobreingeniería a vigilar):**
- Un orquestador distribuido para 5 hipótesis por mes: alcanza con un corredor reanudable, un registro y reportes.
- Dos sistemas de verdad operativos (JSON/SQLite por un lado, Postgres por otro).
- Más agentes o etapas LLM sin evidencia de que suman evidencia independiente.
- Una librería propia de métricas (justo lo que quantstats ya hace).
- Montar ya la ingesta diaria con corporate actions: está bien como referencia, no como alcance.
- Agregar frecuencias intradía antes de validar datos y ejecución diarios.

## Hallazgos sobre nuestro propio repo (para revisar, no tocados acá)

Codex los vio al contrastar; verificar antes de actuar:

1. **Contradicción en el roadmap**: la incubación de 3 meses aparece en shadow local (embudo revisado) pero la decisión C3 dice 3 meses en Alpaca. Resolver antes de automatizar promociones.
2. **Paridad de ejecución**: `fabrica/validacion.py::correr_motor()` tiene una rama por subproceso del motor y otra de pesos objetivo que calcula retornos close-to-close directo. Confirmar que ambas comparten fills y costos antes de agregar más orquestación.
3. **Frescura falsa en la API**: en `apps/api/app.py` el resumen activo devuelve `stale=False` y `alarms=[]` fijos; mostrar Cuenta como autoridad operativa con eso engaña.
4. **Postgres vs SQLite**: el roadmap decide Postgres, pero hoy el registro de cuentas lee SQLite por cuenta. Documentar la decisión real en vez de migrar solo por coherencia.
5. **Tres meses no son prueba estadística** para una estrategia mensual (tres rebalanceos). Pre-registrar qué se necesita observar y cómo se trata la incertidumbre.
6. El README sigue describiendo arquitectura vieja.

## Quick wins propuestos (candidatos a tickets, sin crear ninguno)

| # | Qué | Esfuerzo |
|---|---|---|
| 1 | Adaptador equity→retornos + `quantagent report` con quantstats (un HTML por candidato/cartera, con metadatos y versiones) | chico |
| 2 | Página del manual con la definición de cada métrica y la convención de Sharpe/tasa libre única | chico |
| 3 | Recibo de ciclo/candidato (incluye "ciclo sin promoción") | chico-medio |
| 4 | Corrida de sensibilidad USD 1.000/2.500/5.000 vs SPY | medio (ya en roadmap, subirla de prioridad) |
| 5 | Corregir frescura/alarmas reales en el resumen de cuenta | chico |
| 6 | Tests de contabilidad: entrada/salida a precio igual, comisiones, gaps, caja compartida | chico |
| 7 | Resolver la contradicción C3 y la paridad de `correr_motor()` | chico (decisión + auditoría) |

Decisión pendiente tuya: si querés que estos quick wins pasen a tickets (lote nuevo) o se agreguen al plan de la etapa. Yo no creé ningún ticket.
