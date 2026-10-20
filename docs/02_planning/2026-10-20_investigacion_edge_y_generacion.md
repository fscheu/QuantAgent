# Investigación: por qué no pasa ninguna cartera, dónde buscar edge y cómo cambiar el generador

Fecha: 2026-10-20. Estado: **propuesta, sin decidir**. No cambia código.

Alcance: no se corrió ningún backtest ni el embudo, no se leyó ni escribió `local-data/`, no se tocó `intentos.csv` ni el candado. Las cifras de carteras salen de los IM ya publicados. La única cuenta nueva es una simulación con retornos sintéticos que mide cuánto discrimina el criterio (apéndice A). Las tres propuestas nuevas están en `catalogo/propuestas/`, sin aprobar, y pasan el chequeo de forma y de originalidad de `fabrica/propuestas.py`.

Beads: este trabajo no tiene issue; la pregunta 13 de §7 propone crear uno padre para el lote.

Relacionado: [DC criterio de la etapa 3 y loop](../04_decisions/2026-10-19-DC-criterio-etapa3-y-loop-de-mejora.md), [DC cartera de primas](../04_decisions/2026-10-17-DC-cartera-de-primas.md), [IM L16](../06_implementation/QuantAgent-vbnh-IM-criterio-etapa3.md), [IM L17](../06_implementation/QuantAgent-vbnh-IM-l17-punto-de-control.md), [IM capa macro L14](../06_implementation/QuantAgent-d4cq.75-IM-punto-de-control-capa-macro.md), informe de la curva `docs/06_implementation/QuantAgent-ro3w-IM-criterio-paper-curva.md` (rama `lote/m4-prep-curva-puv2`). Código leído: `quantagent/fabrica/{investigador,propuestas,criterio,cartera,validacion,postmortem,transversal}.py`, `quantagent/strategy/{momentum_12m,sma_cross}_strategy.py`, `catalogo/`, `hipotesis/`.

## Resumen

Las cuatro carteras cortan por una sola condición: en 2019–2022 su Sharpe no supera al de comprar y mantener los mismos activos. Lo demás (Sharpe positivo, 3 de 4 años, drawdown) se cumple. No son los costos ni la frecuencia, y el período explica solo una parte. Lo que más pesa es el tipo de idea: las cuatro son la mitad larga, sin apalancar y sobre los mismos activos del benchmark, de primas que en la literatura son largo-corto, apalancadas o que pagan en crisis largas. Contra 1/N de esos activos solo queda una diferencia chica, y 4 años no alcanzan para verla. Con una ventaja verdadera de +0,2 de Sharpe, el criterio actual la rechaza más de la mitad de las veces; con ventaja cero, la aprueba casi una de cada cinco.

Más volumen del mismo tipo de idea sube N sin subir la chance de encontrar algo. Lo que propongo cambiar, en este orden:

1. **Medición**, antes de abrir cualquier ventana: Sharpe en exceso de la tasa corta y benchmark con los mismos pesos como diagnóstico. En la reserva (2023+, tasas de 4–5 %), medir a tasa 0 una cartera cargada de SHY contra 1/N la favorece sin que la señal haga nada.
2. **Búsqueda** en fuentes de retorno que 1/N no tiene: flujos institucionales predecibles (rebalanceo de fin de mes), el lado corto de la tendencia y el carry en futuros. Tres propuestas YAML.
3. **Generador**: que el LLM conozca qué corta y contra qué, el inventario de datos y clases, y los efectos que ya decayeron, y que justifique por qué su idea le gana a 1/N de los mismos activos.
4. **Criterio**: mantener el actual con la medición corregida para carteras que pretenden reemplazar al benchmark, y abrir una vía de "diversificador" declarada antes de correr. Paper pasa a ser incubación con reglas de corte, porque 12 semanas de paper tienen un error estándar de Sharpe de 2.

Las decisiones de Fede están en §7.

## 1. Diagnóstico

### 1.1 Lo que pasó

| Cartera | Etapa 2 (2007–2018) | Benchmark etapa 2 | Etapa 3 (2019–2022) | Benchmark etapa 3 | Años positivos | Drawdown etapa 3 | Fuente |
|---|---:|---:|---:|---:|---:|---:|---|
| `cartera-tendencia-momentum` | 0,76 | 0,41 | 0,25 | 0,61 | 3 de 4 | 5,5 % | IM L16 |
| `cartera-transversal` | 0,58 | 0,41 | 0,56 | 0,61 | 3 de 4 | 3,7 % | IM L16 |
| `cartera-estacionalidad` | 0,45 | 0,39 | 0,18 | 0,57 | 3 de 4 | 34,3 % | IM L16 |
| `cartera-nucleo-curva` | 0,82 | 0,41 | 0,38 | 0,61 | 3 de 4 | 6,7 % | IM ro3w (L18) |

Tres hechos. Las cuatro le ganan al benchmark en la etapa 2 y pierden en la 3. El benchmark mejora de una etapa a la otra (0,41 → 0,61) mientras las carteras empeoran. Y el postmortem de L17 no encontró ninguna causa estructural (`causas: []` en las tres fichas): no fueron pocos trades, costos, rotación, una pierna dominante ni drawdown.

### 1.2 Factor por factor

| Factor | Peso | Por qué |
|---|---|---|
| Tipo de ideas | Alto | Overlays solo largos sobre los activos del benchmark; la cita define otra estrategia (§1.3) |
| Potencia de una ventana de 4 años | Alto | Diferencias de 0,2–0,3 de Sharpe no se distinguen de cero en 4 años (§1.4) |
| Etapa 2 dentro de la muestra de las citas | Medio | Varias citas llegan hasta 2009–2016; la etapa 3 es la primera ventana posterior a todas (§1.5) |
| Composición de 2019–2022 | Medio | Dos años alcistas fuertes, un crash en V (el peor caso para tendencia lenta) y 2022, donde la tendencia paga con cortos que estas carteras no tienen |
| Benchmark y medición | Medio | 1/N es difícil de batir; además pesa distinto que la cartera y todo se mide a tasa 0 (§1.6) |
| Universo | Medio | 29 ETFs solo largos y sin derivados: pocas apuestas independientes y ninguna prima que 1/N no tenga (§1.7) |
| Generador | Alto para lo que viene | El LLM no sabe qué corta, qué datos hay ni qué decayó; sus tres propuestas de L17 repiten el patrón (§1.8) |
| Costos | Bajo | 5 pb por lado (`COSTOS_PERFILES["etf"]`) y señales mensuales; el postmortem no marca `costos_comen_ganancia` |
| Frecuencia | Bajo | Las barras diarias alcanzan para primas mensuales. Ir a intradía no achica el error del Sharpe por año calendario y los efectos intradía más conocidos decayeron (§2.1) |

### 1.3 Lo que define la cita y lo que se probó

| Pierna | Cita | Lo que define la cita | Lo implementado |
|---|---|---|---|
| `momentum-12m` | Moskowitz, Ooi y Pedersen 2012 | Largo y corto en 58 futuros, escalado por volatilidad, retorno en exceso de la tasa | Solo largo en ETFs; en efectivo a 0 % cuando la señal es negativa |
| `sma-cross` 1/210 | Faber 2007; Hurst, Ooi y Pedersen 2017 | Faber decide a fin de mes; HOP es largo-corto en 67 mercados | Decisión diaria (el YAML lo aclara), solo largo |
| `momentum-transversal` | Jegadeesh y Titman 1993; Asness, Moskowitz y Pedersen 2013 | Ganadores menos perdedores (largo-corto) | Solo el tercio ganador, largo |
| `baja-volatilidad` | Frazzini y Pedersen 2014 y otros | Largo bajo riesgo apalancado y corto alto riesgo, neutral a beta | Solo el tercio menos volátil, sin apalancar |
| `halloween` | Bouman y Jacobsen 2002 | Diferencia de retorno nov–abr contra may–oct | Largo nov–abr; fiel a la cita (el YAML ya cita el decaimiento de McLean y Pontiff) |
| `curva-pendiente` | Cochrane y Piazzesi 2005; Koijen et al. 2018 | Exceso de retorno de bonos predicho por la curva | Largo bonos si T10Y2Y > 0; casi no cambió de estado en 2007–2018 (lo dice el YAML) |

La mitad larga de una prima largo-corto es, en lo esencial, exposición al mercado más media prima. Contra un benchmark que ya es el mercado de esos activos, lo que queda para ganar es esa media prima menos lo que se pierde por estar en efectivo cuando la señal es negativa. Es una diferencia chica, y en 4 años su signo es casi una moneda (§1.4).

En la tendencia, la ganancia de la cita se concentra en caídas largas, donde paga el lado corto. 2019–2022 tuvo un crash en V en marzo de 2020 (la tendencia lenta sale después de la caída y vuelve después del rebote) y, en 2022, el año bueno para la tendencia, la versión solo larga solo podía estar en efectivo.

### 1.4 Cuánto discrimina un criterio de 4 años

El Sharpe estimado en T años tiene un error estándar cercano a 1/√T, y la diferencia entre dos Sharpe depende además de la correlación entre las curvas. Simulé el criterio completo con retornos normales i.i.d. (apéndice A): Sharpe > 0 y mayor al benchmark en la etapa 2 (11 años efectivos) y en la 3 (4 años), y Sharpe positivo en 3 de 4 años. El drawdown no corta en estas simulaciones. Sharpe del benchmark: 0,5.

Probabilidad de pasar según la ventaja verdadera de la cartera:

| ΔSharpe verdadero | ρ = 0,3 | ρ = 0,6 | ρ = 0,85 |
|---:|---:|---:|---:|
| −0,2 | 0,08 | 0,05 | 0,02 |
| 0 | 0,20 | 0,18 | 0,18 |
| +0,1 | 0,28 | 0,28 | 0,34 |
| +0,2 | 0,38 | 0,43 | 0,52 |
| +0,3 | 0,48 | 0,54 | 0,69 |
| +0,5 | 0,68 | 0,75 | 0,84 |

Error estándar de ΔSharpe estimado en 4 años: 0,60 con ρ = 0,3; 0,43 con ρ = 0,6; 0,28 con ρ = 0,85; 0,16 con ρ = 0,95.

Qué dice:
- Una cartera sin ventaja pasa 18–20 % de las veces; una con +0,2, que ya sería una prima buena después del decaimiento, pasa 38–52 %. El criterio separa, pero poco.
- La brecha de `cartera-transversal` (0,56 contra 0,61) es un décimo de error estándar. No dice que sea peor que el benchmark ni que sea mejor.
- Si las cuatro tuvieran +0,2 de verdad (ρ = 0,6), la chance de que corten todas sería 0,59⁴ ≈ 0,12. Como dos comparten núcleo, son unas tres independientes y la cifra sube a ≈ 0,2. Las cuatro caídas son evidencia moderada de que la ventaja es cero o chica; no son prueba.
- Alargar la prueba a 8 años (2019–2026, pidiendo 6 de 8 años positivos) casi no cambia nada: 0,16 sin ventaja y 0,43 con +0,2 (ρ = 0,6). Lo que limita es la conjunción de condiciones sobre estimaciones puntuales.
- La palanca más grande es la probabilidad previa de la idea. Con potencia 0,45 y falsos positivos 0,19, una idea que entra con 20 % de chance de ser real sale con 37 % si pasa; una que entra con 50 % sale con 70 %. Mejorar el generador rinde más que ajustar el criterio.

### 1.5 La etapa 2 no es fuera de muestra para la literatura

Las muestras de varias citas llegan adentro de la etapa 2: Moskowitz, Ooi y Pedersen hasta 2009; Asness, Moskowitz y Pedersen hasta 2011; Frazzini y Pedersen y Koijen et al. hasta 2012; Hurst, Ooi y Pedersen hasta mediados de la década de 2010. Una prima elegida porque la literatura la documentó tiende a andar bien en esos años. La etapa 3 es la primera ventana entera después de todas las publicaciones, y el patrón observado (gana en la 2, pierde en la 3) es el que anticipan McLean y Pontiff (2016): en 97 predictores, 26 % menos de retorno fuera de muestra y 58 % menos después de publicado. Falck, Rej y Thesmar (2022) encuentran que fuera de muestra queda cerca de la mitad del Sharpe y que el año de publicación es lo que mejor predice el decaimiento.

### 1.6 Benchmark y medición

1/N es un rival difícil: DeMiguel, Garlappi y Uppal (2009) no encuentran, entre 14 modelos de cartera, ninguno que le gane de forma consistente fuera de muestra en Sharpe. Sobre eso hay dos detalles de medición que ensucian la comparación:

- **Pesos distintos.** La cartera pesa cada celda por 1/σ de 60 sesiones (`fabrica/cartera.py`), así que en capital queda dominada por bonos cortos y medios (SHY, AGG, IEF, TIP). El benchmark pesa igual a los 29 (`criterio.curva_benchmark`). ΔSharpe mezcla el efecto de la señal con el de pesar distinto. De ahí los drawdowns de 3,7–6,7 % de las carteras multiactivo.
- **Tasa 0.** El Sharpe de cartera y benchmark se calcula a tasa 0. A tasa 0, un activo que rinde como la tasa corta, como SHY, tiene un Sharpe muy alto. Con pesos 1/σ, SHY se lleva una parte grande del capital de la cartera; en el benchmark, 1/29. En 2019–2022 la tasa promedió cerca de 1 % y el efecto fue chico y de signo incierto. En la reserva, con tasas de 4–5 %, el efecto es grande y a favor de cualquier cartera cargada de bonos cortos, haga lo que haga su señal. Antes de abrir la reserva hay que medir en exceso de DGS3MO, que ya está en `macro-2026-10`.

1/N de los mismos activos sigue siendo una vara honesta para decidir si vale la pena operar: es lo que el operador tendría sin hacer nada. Pero como diagnóstico conviene separar la brecha en pesos, señal y tasa (ticket R04).

### 1.7 Universo

Los 29 ETFs incluyen 9 sectores del S&P 500 más SPY, QQQ e IWM, muy correlacionados entre sí. Grinold (1989) liga el ratio de información a la raíz de la cantidad de apuestas independientes; acá son del orden de 8 (estimación por clase de activo, no medida). Y como el universo es solo largo y sin derivados, no hay forma de cobrar primas que 1/N no tenga: carry, prima de seguro (venta de opciones) o el lado corto de la tendencia. Con este universo solo se puede elegir cuándo y cuánto de cada activo, que es lo que menos señal tiene.

### 1.8 El generador

`armar_pedido` (`fabrica/investigador.py`) le pasa al LLM el catálogo, las fichas de postmortem y N. No le dice:
- qué condición corta (superar a comprar y mantener los mismos activos);
- qué datos y qué clases de señal existen: las tres propuestas de L17 pedían clases nuevas y una pedía un calendario que no está;
- qué efectos ya decayeron: propuso la deriva pre-FOMC, que Kurov, Wolfe y Gilbert (2021) muestran desaparecida desde 2015, con una de sus dos citas apuntando a otro artículo;
- que una ficha con `causas: []` significa "la causa no es estructural", no "todo bien".

Resultado: las tres propuestas de L17 vuelven a ser exposición larga a acciones con pesos que cambian en el tiempo (cambio de mes, volatilidad gestionada, pre-FOMC), el patrón que ya cortó.

Hay un sesgo más difícil de ver. Un LLM entrenado hasta 2025–2026 sabe, a grandes rasgos, cómo les fue a las estrategias conocidas en 2019–2022 y después. El candado impide correr dos veces sobre la prueba; no impide proponer lo que el proponente ya sabe que anduvo. Para ideas generadas hoy, por el LLM o por mí en este documento, 2019–2022 no es una prueba ciega y la reserva tampoco del todo. La prueba ciega es paper.

### 1.9 Conclusión

El loop gira en círculos porque explora una región donde la ventaja esperada contra el benchmark es chica (overlays solo largos de primas publicadas, sobre los activos del benchmark), con un instrumento que no ve ventajas chicas en 4 años. Más volumen en esa región sube N sin subir la chance de encontrar algo. Hay que mover la búsqueda a fuentes de retorno que 1/N no tiene y que, por su mecanismo, prometen una ventaja más grande o están menos explotadas, y medir con reglas que sepan qué pueden ver.

## 2. Dónde es creíble el edge hoy

### 2.1 Decaimiento y múltiples pruebas

| Trabajo | Qué encuentra | Para el loop |
|---|---|---|
| McLean y Pontiff 2016 | 97 predictores: −26 % fuera de muestra y −58 % después de publicado; cae más lo que era más fuerte en muestra | Descontar al menos la mitad del efecto publicado |
| Falck, Rej y Thesmar 2022 | Fuera de muestra queda ~50 % del Sharpe; el año de publicación explica 30 % del decaimiento, y la complejidad de la señal junto con la sensibilidad del Sharpe a valores extremos, otro 15 % | Preferir señales simples; pedir muestra y año de la cita |
| Hou, Xue y Zhang 2020 | 65 % de 452 anomalías no pasan \|t\| = 1,96 con pesos por capitalización y cortes de NYSE; 82 % no pasan t = 2,78 | La mayoría de las anomalías chicas de acciones no se replica |
| Harvey, Liu y Zhu 2016 | Con más de 300 factores probados, uno nuevo necesita t > 3 | El umbral sube con N, que es lo que el registro de intentos ya hace |
| Chordia, Goyal y Saretto 2020 | Con millones de estrategias, los umbrales corregidos son del orden de t = 3,8 y sobreviven muy pocas | Igual |
| Jensen, Kelly y Pedersen 2023 | Con un modelo bayesiano, la mayoría de los factores se replica en 93 países, agrupados en 13 temas | Las primas amplias y diversificadas existen; lo frágil es la anomalía chica y aislada |
| Linnainmaa y Roberts 2018 | Muchas anomalías son débiles antes de la muestra original | El fuera de muestra hacia atrás también falla |
| Novy-Marx y Velikov 2016 | Las anomalías de rotación alta no sobreviven a los costos | Rotación mensual o menor, salvo efectos de liquidez con mecanismo |
| Sullivan, Timmermann y White 1999, 2001 | Corrigiendo por la búsqueda, ninguna de miles de reglas técnicas y ningún efecto de calendario es significativo | Calendario solo con mecanismo y evidencia reciente |
| DeMiguel, Garlappi y Uppal 2009 | Ninguno de 14 modelos le gana a 1/N fuera de muestra de forma consistente | El benchmark actual es exigente por diseño |
| Goyal y Welch 2008; Goyal, Welch y Zafirov 2024 | Casi ningún predictor macro le gana a la media histórica fuera de muestra | La capa macro tiene probabilidad previa baja |
| Huang, Li, Wang y Zhou 2020 | El momentum de series de tiempo es débil activo por activo; el t agregado no sobrevive a un bootstrap | La tendencia vale como cartera diversificada, no por activo |
| Cederburg, O'Doherty, Wang y Yan 2020 | Las versiones con volatilidad gestionada de 103 estrategias no le ganan a las originales fuera de muestra, aunque tienen alfa en regresiones de expansión | Volatilidad gestionada sirve como diversificador, no sola |

Efectos con decaimiento documentado, que el generador no debería volver a proponer sin evidencia posterior:
- **Deriva pre-FOMC:** desaparece después de 2015 (Kurov, Wolfe y Gilbert 2021).
- **Deriva nocturna en futuros del S&P 500:** la ventana de 2 a 3 de la mañana (hora de Nueva York) que rendía ~3,7 % anual en 1998–2020 promedia cerca de cero desde 2021, también en Nasdaq y Dow. La dispersión del desbalance de órdenes al cierre, que la explicaba, cayó a la mitad. Los dos ETFs "nocturnos" lanzados en 2022 cerraron a los 14 meses (Boyarchenko, Larsen y Whelan, Liberty Street Economics, 2026).
- **Deriva posterior a resultados (PEAD):** ausente en acciones grandes desde 2006 según Martineau (2022); en 2025 hubo trabajos en contra, sin consenso.
- **Carry de cripto (base futuro-contado):** rentable en 2020–2024 y con Sharpe negativo en 2025 según una revisión de 2025 (arXiv 2510.14435), sobre el trabajo de Schmeling, Schrimpf y Todorov (BIS, 2023).
- **Halloween y cambio de mes:** los propios trabajos y McLean y Pontiff anticipan decaimiento, y Sullivan, Timmermann y White (2001) no los encuentran significativos tras corregir por la búsqueda. Etula, Rinne, Suominen y Vaittinen (2020) le dan al cambio de mes un mecanismo (necesidades de caja institucionales de fin de mes) que lo sostiene mejor que la estacionalidad pura.

### 2.2 Qué ventaja tiene un operador chico

Tiene capacidad: puede operar efectos que a un fondo grande no le cambian el resultado. No tiene riesgo de carrera: puede sostener una cartera que pierde contra el índice durante años. Y no tiene mandato: puede vender corto, usar opciones o quedarse en efectivo. No tiene infraestructura de baja latencia, paga los datos caros en proporción a su capital y sufre la granularidad: una opción sobre SPY o XSP equivale a unos USD 65 mil de nominal y un futuro de bono del Tesoro, a USD 100–200 mil. Con Alpaca opera acciones, ETFs, opciones y cripto; los futuros figuran en su hoja de ruta, no como producto.

Conclusión práctica: la ventaja de un operador chico está en flujos y liquidez de capacidad limitada y en primas que los grandes no pueden o no quieren cobrar, no en información ni en velocidad.

### 2.3 Mapa por fuente de datos

| Fuente de datos | Edge creíble hoy | Evidencia | Estado después de publicado | Costo y origen del dato | Encaje operativo | Veredicto |
|---|---|---|---|---|---|---|
| ETFs diarios (lo actual) | Flujos de fin de mes (rebalanceo 60/40, necesidades de caja); lado corto de la tendencia; cambio de mes | Harvey, Mazzoleni y Melone 2025; Etula et al. 2020; Moskowitz, Ooi y Pedersen 2012 | Rebalanceo: el efecto aparece desde 1997 y crece con los fondos balanceados; publicado en 2025. Cambio de mes: débil pero con mecanismo | Gratis (Yahoo, ya integrado) | Alpaca sí; cortos con cuenta margin de USD 2.000 o más, solo en ETFs fáciles de tomar en préstamo y sin costo de préstamo en ellos | Prioridad 1 |
| FRED / ALFRED (lo actual) | Tasa corta para medir bien; pendiente de curva como capa (ya probada) | Goyal y Welch 2008, 2024 | Predicción macro con probabilidad previa baja | Gratis con clave | No aplica | Usar para medir, no como fuente de ideas |
| Futuros: tendencia y carry | Tendencia largo-corto diversificada; carry de materias primas y monedas | Hurst, Ooi y Pedersen 2017; Koijen et al. 2018; Lustig et al. 2011; Szymanowska et al. 2014 | Tendencia: buena en 2022, floja en 2023–2025 (SG Trend Index ≈ −4 % en 2023 y +2 % en 2025, fuente secundaria). Carry: prima por riesgo de crisis, sin decaimiento por publicación documentado | Norgate Futures (contratos individuales, más de 30 años; precio en su sitio) o Databento (cobro por GB, USD 125 de crédito inicial, más de 15 años de CME); broker con futuros (IBKR) | No con Alpaca | Prioridad 2: depende de una decisión de datos y broker |
| Opciones y volatilidad implícita | Prima de varianza: vender seguro contra caídas | Carr y Wu 2009; Bakshi y Kapadia 2003; Bollerslev, Tauchen y Zhou 2009; Ilmanen 2012; Israelov y Nielsen 2015 | Es una prima por riesgo, no una anomalía. Pero el índice PUT de Cboe rindió +13,5 / +2,1 / +21,8 / −7,7 % en 2019–2022 contra +31,5 / +18,4 / +28,7 / −18,1 % del S&P 500 con dividendos: en esa ventana su Sharpe quedó por debajo del subyacente. En 1986–2015, 0,67 contra 0,47 | Índices de Cboe (PUT, BXM, VIX) gratis en CSV; cadenas históricas en Alpaca (historia corta) o proveedores pagos a cotizar; OPRA en vivo con Alpaca Algo Trader Plus, USD 99 por mes | Alpaca sí (puts cubiertas desde el nivel 1); un contrato equivale a ~USD 65 mil | Prioridad 3: no para el criterio actual; candidata a diversificador con más capital |
| Intradía | Momentum intradía del mercado (la primera media hora predice la última) | Gao, Han, Li y Zhou 2018; Baltussen, Da, Lammers y Martens 2021 (cobertura de gamma de los dealers) | Muy sensible a costos; con el auge de opciones de vencimiento diario la dinámica de gamma cambió y la evidencia reciente es mixta. La deriva nocturna decayó | Alpaca gratis (barras de minuto desde 2016; el plan gratis restringe los últimos 15 minutos) | Alpaca sí; el motor no es intradía | No ahora |
| Acciones individuales y factores | Rentabilidad y calidad, valor, momentum; capacidad en acciones chicas | Jensen, Kelly y Pedersen 2023; Novy-Marx 2013; Hou, Xue y Zhang 2020 | Los factores amplios se replican; las anomalías chicas no. Contra 1/N de las mismas acciones la ventaja esperada vuelve a ser chica, el mismo problema de hoy | Precios sin sesgo de supervivencia: Norgate Platinum, USD 630 por año (deslistadas desde 1990); fundamentales punto en el tiempo: a cotizar | Alpaca sí | No ahora: mucha inversión de datos para un ΔSharpe esperado chico |
| Cripto | Momentum de series de tiempo | Liu y Tsyvinski 2021; Liu, Tsyvinski y Wu 2022 | Historia corta (líquida desde ~2014); el carry de base decayó en 2025; drawdowns por encima de 35 % aun con tendencia | Gratis (APIs de exchanges y de Alpaca) | Alpaca solo contado y solo largo | No con el embudo actual: no existe 2007–2018 |
| Datos alternativos | Cambios en el texto de los 10-K, compras de directivos, short interest agregado | Cohen, Malloy y Nguyen 2020; Cohen, Malloy y Pomorski 2012; Rapach, Ringgenberg y Zhou 2016 | Los dos primeros son de acciones individuales y pagan mucho en el lado corto; el tercero predice el mercado | EDGAR gratis; short interest de FINRA gratis | Requiere el universo de acciones | Interesante para un LLM (texto), pero después de tener acciones y con el sesgo de mirada al futuro del modelo controlado (Glasserman y Lin 2023) |

### 2.4 Qué sube la probabilidad previa de una idea

1. Trae una fuente de retorno que 1/N de los mismos activos no tiene: lado corto, carry, prima de seguro o un flujo predecible al que la estrategia le provee liquidez.
2. Tiene una contraparte que no puede dejar de hacer lo que hace (mandatos, calendario de pagos, coberturistas), en lugar de un sesgo de comportamiento que el arbitraje corrige al publicarse.
3. La muestra de la cita termina antes de la ventana de prueba, o la prueba se hace en datos posteriores a la muestra (reserva y paper).
4. Hay evidencia de que siguió después de publicada, o una razón concreta para que no decaiga.
5. La implementación es fiel a la cita: si la cita es largo-corto, se prueba largo-corto.

## 3. Revisión del criterio de paper

### 3.1 Lo que se mantiene

Pre-registro, cero parámetros libres, todo intento cuenta, una sola apertura por hipótesis y el DSR como indicador con N efectivo. Nada de lo que sigue cambia eso.

### 3.2 Correcciones de medición, antes de cambiar el criterio

- **M1. Sharpe en exceso de la tasa corta** (DGS3MO) para cartera y benchmark, como columna nueva del informe. El registro de intentos sigue a tasa 0 (D7 del DC de cartera de primas) para no romper la comparabilidad del DSR. Obligatorio antes de abrir la reserva.
- **M2. Benchmark con los mismos pesos** de la cartera, siempre invertido, informado al lado de 1/N, con la brecha separada en pesos, señal y tasa. Es diagnóstico, no corte.
- **M3. Error estándar de ΔSharpe** (Ledoit y Wolf 2008, o bootstrap por bloques) junto a la cifra puntual, para que "0,56 contra 0,61" se lea como lo que es.

### 3.3 Tres criterios posibles

| | C0: el actual, con M1–M3 | C1: diversificador declarado | C2: no inferioridad e incubación |
|---|---|---|---|
| Regla | Sharpe > 0 y > benchmark en etapas 2 y 3; 3 de 4 años positivos; drawdown ≤ 35 % | Solo para carteras que declaran `rol: diversificador` en el YAML antes de correr: alfa contra el benchmark > 0 en etapas 2 y 3 (equivale a que la mezcla mejore el Sharpe, es decir Sharpe propio > ρ × Sharpe del benchmark; Huberman y Kandel 1987); la mezcla 50/50 en riesgo con Sharpe positivo en 3 de 4 años; drawdown de la mezcla ≤ el del benchmark | El histórico solo descarta lo claramente peor (ΔSharpe estimado < −1 error estándar, Sharpe ≤ 0, menos de 3 de 4 años, drawdown); la decisión pasa a paper, con tamaño chico y reglas de corte escritas antes |
| Pasa sin ventaja | 18–20 % | 18–19 % (alfa cero) | 44–48 % |
| Pasa con +0,2 / +0,3 | 38–52 % / 48–69 % | 34–38 % / 43–48 % (alfa en unidades de Sharpe) | 66–70 % / 73–78 % |
| Con más datos | Prueba de 8 años (2019–2026) en lugar de 4, pidiendo 6 de 8 años: 16 % sin ventaja y 43 % con +0,2 (ρ = 0,6), casi igual | Exigiendo además alfa > 0 en la reserva: 9 % sin ventaja; 22–26 % / 32–34 % con +0,2 / +0,3 | No aplica |
| Qué selecciona | Carteras que reemplazan al benchmark | Carteras que se suman a una base comprada | Casi todo lo razonable va a incubación |
| Riesgo principal | Rechaza la mayoría de lo bueno y premia la suerte de 4 años | El operador tiene que mantener de verdad la base comprada; una cartera con Sharpe propio bajo puede pasar | Más carteras en paper y más tiempo de Fede; exige disciplina para cortar; un falso positivo con suerte en paper llega a vivo |

Cifras de C0, C1 y C2: simulación del apéndice A.

### 3.4 Recomendación

C0 con M1–M3 para las carteras que pretenden reemplazar al benchmark, y C1 como vía separada para las que declaran `rol: diversificador` en el YAML antes de correr. C1 no relaja en tasa de error: los falsos positivos son los mismos que en C0. Cambia lo que se busca. Si el operador de todos modos tiene una base comprada, una cartera con alfa contra esa base mejora su resultado total aunque sola no le gane al índice. Cederburg et al. (2020) muestran exactamente ese caso con la volatilidad gestionada: no le gana a la estrategia original, pero suma alfa a una cartera que la contiene.

C2 lo descarto por ahora. Desplaza todo el filtro a paper, y paper hoy no tiene reglas de corte ni la duración que haría falta para sostener ese rol.

### 3.5 Riesgos de relajar

- **Rescate retroactivo.** Con las cifras publicadas, `cartera-transversal` (0,56 contra 0,61) pasaría C1 si su correlación con el benchmark en 2019–2022 fue menor a 0,92, y `cartera-nucleo-curva` (0,38) si fue menor a 0,62. Aplicar C1 a la prueba ya abierta para salvarlas es elegir el criterio mirando el resultado. Si Fede quiere darles otra oportunidad, la vía honesta es la reserva con el criterio escrito antes de abrirla.
- **Más falsos positivos en vivo** si se baja la vara sin bajar el tamaño. El recargo por DSR que ya decidió Fede (50 % y 12 semanas con DSR < 0,50) cubre parte de esto.
- **Cambio de rol después de correr.** Una cartera que corta como independiente no puede volver como diversificador. El rol se declara en el YAML y se guarda con el hash del universo en `hipotesis-universos.csv`.

### 3.6 Paper: incubación, no confirmación

Error estándar del Sharpe anual estimado según la duración: 6 semanas, 2,9; 12 semanas, 2,1; 26 semanas, 1,4; 52 semanas, 1,0; 104 semanas, 0,7. Doce semanas de paper no confirman ni descartan una ventaja de 0,3. Lo que paper sí puede verificar en semanas es la implementación. Propuesta, con los números a fijar por Fede:
- **Éxito de paper = seguimiento:** correlación diaria entre los retornos de paper y los de la simulación sobre los mismos días de al menos 0,9; costo real por operación de no más del doble del supuesto; ninguna orden fuera de regla.
- **Corte escrito antes, en paper y en vivo:** la curva acumulada cae bajo el percentil 5 de la distribución simulada de la cartera para ese horizonte, o el drawdown supera el límite.
- **Vivo con tamaño reducido** (el recargo vigente) después del seguimiento mínimo; tamaño completo solo después de 12–24 meses sin activar el corte.

### 3.7 La reserva 2023+

Es la única ventana histórica sin abrir: unos 3,8 años desde `RESERVA_DEFAULT = "2023-01-01"`. Se puede usar una sola vez por hipótesis. Propuesta: abrirla una vez, con M1 aplicado y el criterio decidido, para una lista corta (máximo 3) escrita antes. Candidatas: `cartera-transversal` (la más cercana en 2019–2022), `rebalanceo-fin-de-mes` si replica en la etapa 2 y `cambio-de-mes` si pasa la etapa 2. La reserva también cae dentro del entrenamiento del LLM y del mío, así que es más ciega para hipótesis definidas antes de este documento que para las propuestas acá.

## 4. Familias priorizadas

Las probabilidades son juicio, no medición. Se anclan en la tabla de §1.4 (lo máximo realista es 0,5–0,7) y en el decaimiento de §2.1. C0 y C1 son los criterios de §3.3.

| # | Familia | Hipótesis económica | Cita verificable | Universo | Señal (0 parámetros libres) | Datos y costo | N que agrega | P(pasa C0) | P(pasa C1) | Estado |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | Presión de rebalanceo de fin de mes | Los fondos balanceados (~USD 20 billones) rebalancean a fin de mes contra el ganador del mes; quien les provee liquidez cobra | Harvey, Mazzoleni y Melone 2025, [10.3386/w33554](https://doi.org/10.3386/w33554); Etula et al. 2020, [10.1093/rfs/hhz054](https://doi.org/10.1093/rfs/hhz054) | SPY, IEF | 50/50 el resto del mes; en los últimos 5 días hábiles, 100 % en el activo que la señal favorece (desvío del peso de acciones de una 60/40 que deriva desde el último cierre de mes) | Actuales, gratis. Clase nueva y pesos objetivo por fecha | 1 cartera, hasta 3 filas (etapas 1 a 3) | 0,4–0,6 en 2007–2022, dentro de la muestra del paper (poco informativo); 0,3–0,4 en 2023+ | 0,4–0,5 | YAML `rebalanceo-fin-de-mes` |
| — | Cambio de mes | Flujos de caja de fin de mes compran acciones | Lakonishok y Smidt 1988; McConnell y Xu 2008; Etula et al. 2020 | 8 ETFs de índices | Largo de −1 a +3 | Actuales | Ya en L18 | 0,2–0,35 | 0,3 | Aprobada, en L18 (Q02–Q04) |
| F2 | Tendencia larga y corta | La prima de tendencia se cobra en las dos direcciones; la versión solo larga está en efectivo cuando paga | Moskowitz, Ooi y Pedersen 2012, [10.1016/j.jfineco.2011.11.003](https://doi.org/10.1016/j.jfineco.2011.11.003); Hurst, Ooi y Pedersen 2017, [10.3905/jpm.2017.44.1.015](https://doi.org/10.3905/jpm.2017.44.1.015) | Los 29 ETFs | Signo del retorno de 252 sesiones en exceso de DGS3MO, decisión a fin de mes, largo o corto, pesos 1/σ de la cartera | Actuales. Cortos en el motor; en Alpaca, cuenta margin | 1 cartera, hasta 3 filas; puede compartir grupo con la tendencia ya probada | 0,3–0,45 | 0,45–0,55 | YAML `tendencia-larga-corta` |
| F3 | Carry en futuros de materias primas y monedas | Prima por riesgo de crisis y de liquidez: largo carry alto, corto carry bajo, neutral dentro de cada clase | Koijen et al. 2018, [10.1016/j.jfineco.2017.11.002](https://doi.org/10.1016/j.jfineco.2017.11.002); Lustig et al. 2011, [10.1093/rfs/hhr068](https://doi.org/10.1093/rfs/hhr068); Szymanowska et al. 2014, [10.1111/jofi.12096](https://doi.org/10.1111/jofi.12096) | 7 materias primas y 6 monedas de CME | Materias primas: pendiente anualizada entre 1.º y 2.º vencimiento; monedas: prima del futuro sobre el contado; pesos por ranking dentro de la clase; igual riesgo por clase | Futuros pagos (Norgate o Databento) y broker con futuros | 1 cartera, hasta 3 filas | 0,3–0,4 | 0,45–0,55 | YAML `carry-futuros`; fase 2 |
| — | Volatilidad gestionada (pendiente, de L17) | Escalar por la inversa de la varianza mejora el Sharpe | Moreira y Muir 2017 | SPY | Inversa de la varianza del mes previo | Actuales; para ser fiel necesita exposición mayor a 1 en meses calmos | 1 cartera, hasta 3 filas | 0,15–0,25 | 0,3–0,45 | Solo como diversificador, y con la constante de escala calculada con datos hasta cada fecha (la de Moreira y Muir usa la muestra completa) |
| — | Prima de varianza (índice PUT) | Vender seguro contra caídas paga | Carr y Wu 2009; Bakshi y Kapadia 2003; Ilmanen 2012 | SPY (índice PUT de Cboe) | Reglas fijas de Cboe | Índice gratis; ejecución con opciones | — | ≤ 0,15 | ≤ 0,25 | No la propongo ahora: en 2019–2022 quedó por debajo del subyacente según cifras públicas de Cboe |
| — | Cripto, deriva nocturna, pre-FOMC, PEAD | | | | | | | | | No: historia corta o decaimiento documentado (§2.1) |

### F1. Presión de rebalanceo de fin de mes

Harvey, Mazzoleni y Melone simulan una cartera 60/40 de futuros del S&P 500 y del bono a 10 años y miden cuánto se aleja su peso de acciones del 60 % a medida que pasan los días del mes. Cuando las acciones quedan sobreponderadas, los fondos venden acciones y compran bonos, y el retorno de las acciones cae del orden de 17 pb al día siguiente; la presión se disuelve en unas dos semanas. La señal de calendario es más fuerte 4 y 5 días antes del fin de mes (t cercano a −2,8 y −3,1 en el gráfico de calibración de los autores) y coincide con la reversión que Etula et al. (2020) ubican alrededor del último día de liquidación del mes. Una estrategia que toma el otro lado (largo-corto) tiene Sharpe 1,08 en 1997–2023, con saltos grandes en 2008 y en 2020. En 1961–1997 el efecto no es significativo, lo que coincide con el mecanismo (los fondos balanceados crecieron después) y a la vez es una advertencia.

Por qué puede ganarle a 1/N: la cartera es el benchmark (50/50 SPY e IEF) salvo los últimos 5 días hábiles del mes, cuando pasa entera al activo que la presión favorece. La diferencia con el benchmark mide solo la señal, y esa señal es una fuente de retorno (proveer liquidez a un flujo forzado) que el benchmark no tiene.

Riesgos: la muestra del paper (hasta marzo de 2023) cubre las etapas 2 y 3, así que pasarlas es una réplica, no una confirmación; la prueba que informa es la reserva y paper. Publicado en 2025, puede decaer como la deriva nocturna si más gente toma el mismo lado; los autores cuentan que algunos fondos ya anticipan su propio rebalanceo. La versión del YAML usa el signo de la señal; el paper pondera por la magnitud. El paper predice el retorno de cierre a cierre del día siguiente; con llenado en la apertura siguiente se pierde la parte nocturna del primer día, así que conviene ejecutar al cierre. Antes de aprobar hay que verificar en el texto del paper la definición exacta de la "cuarta semana".

Qué la mataría: que no replique en 2007–2018 con nuestros datos (barato de saber), o que en la reserva el desvío de la 60/40 no prediga el retorno del día siguiente.

### F2. Tendencia larga y corta

Es la estrategia de la cita sin el recorte a solo largo: decisión mensual, largo si el retorno de 12 meses supera a la tasa corta del mismo período y corto si no. Usa los mismos 29 ETFs y los mismos pesos de cartera. La fuente de retorno nueva es el lado corto, que el benchmark no tiene. Su correlación con 1/N debería ser baja, por eso la propongo con `rol: diversificador`.

Riesgos: la propongo sabiendo que en 2022 la tendencia con cortos ganó, así que 2019–2022 no es una prueba ciega; la evaluación que vale es la reserva y paper, y en 2023–2025 la tendencia diversificada anduvo floja. Es la tercera cartera con la prima de tendencia: comparte tema con `cartera-tendencia-momentum` y probablemente grupo en el N efectivo. Hace falta que el motor admita celdas cortas y, en Alpaca, una cuenta margin; los cortos solo se abren en ETFs fáciles de tomar en préstamo.

Qué la mataría: que el lado corto no aporte en 2007–2018 (la etapa 2 incluye 2008, donde debería pagar) o una correlación con 1/N parecida a la de la versión solo larga.

### F3. Carry en futuros (fase 2)

Es la prima con más evidencia en varias clases de activos que el universo actual no puede cobrar. Neutral en dólares dentro de cada clase, no tiene beta de acciones. Su riesgo conocido es el de crisis y de liquidez: el carry de monedas cae en las huidas hacia la calidad. Se eligieron materias primas y monedas porque su carry se mide con la propia curva de futuros, sin datos externos.

Lo que exige: contratos individuales de futuros (no series continuas), un motor de roll, un broker con futuros (IBKR) y capital suficiente para la granularidad de los contratos, a estimar en el ticket de datos. No conviene empezar hasta que una cartera con datos actuales llegue a paper.

### Contaminación de estas propuestas

Elegí F1 y F2 sabiendo, a grandes rasgos, cómo les fue a esas familias en 2019–2022: que el front-running del rebalanceo tuvo un salto grande en 2020 y que la tendencia con cortos ganó en 2022. F3 está menos contaminada porque no conozco el resultado del carry en esos años con precisión. Por eso propongo que, para F1 y F2, la etapa 3 se lea como réplica y que la decisión de paper dependa de la reserva con M1 aplicado.

## 5. Cambios al generador

El objetivo es subir la probabilidad previa de lo que llega al embudo, no la cantidad. Ninguno de estos cambios le pasa al LLM resultados de la prueba.

### 5.1 El pedido (`armar_pedido`)

Borrador del texto nuevo, que reemplaza las tres líneas de reglas actuales:

```text
Sos investigador de una cartera de trading de un operador chico. Proponé entre 0 y 3 entradas
nuevas para el catálogo bajo la clave `propuestas:`. Si no tenés una idea que cumpla todo, devolvé
`propuestas: []`; una propuesta débil cuesta un intento en el registro y sube la vara para todas.

Cómo se evalúa (no cambia por tu propuesta):
- Cada cartera se compara con comprar y mantener los mismos activos con pesos iguales y rebalanceo
  mensual, por Sharpe en exceso de la tasa corta, en 2007-2018 y en una prueba única posterior, y
  después en paper trading. Una cartera que solo decide cuándo estar comprada en los mismos activos
  casi nunca supera esa vara: decí qué fuente de retorno tiene tu propuesta que el benchmark no tenga.
- `rol`: `independiente` (tiene que superar al benchmark) o `diversificador` (tiene que tener alfa
  contra el benchmark). Se declara ahora y no cambia.

Qué hay:
- Activos: {lista de ETFs del snapshot con fecha de inicio}. Series macro: {series del snapshot macro}.
  Velas diarias con apertura y cierre. No hay intradía, opciones ni futuros.
- Clases de señal existentes: {clases de registry.py y de transversal.METRICAS}. Una clase nueva es
  posible pero cuesta un ticket: marcá `clase_nueva: true`.
- Motor: llenado en la apertura siguiente, 5 pb por lado, {cortos: sí/no}.

Qué no proponer:
- Efectos con decaimiento documentado: {lista del catálogo con su cita}.
- Temas ya probados dos veces: {temas}, salvo para corregir un desvío de la cita.

Reglas: 0 parámetros libres; el universo sale de dónde está definida la prima; cada DOI tiene que
existir y ser del artículo que nombrás en `cita_texto` (se verifica título y autor). Si la muestra de
la cita llega a 2019 o después, la propuesta no tiene prueba ciega en 2019-2022 y va directo a la
reserva y a paper. Descontá la mitad del efecto publicado al estimar `efecto_esperado`.
```

Lo que va entre llaves se arma con el manifiesto del snapshot, el del snapshot macro, `registry.py`, `transversal.METRICAS` y el catálogo, no a mano. El test de texto que hoy verifica que el pedido no tenga Sharpe, DSR ni ganancias se mantiene.

### 5.2 Campos nuevos de la propuesta

Además de los actuales (`id`, `idea`, `clase`, `ventana`, `hipotesis_economica`, `fuente`, `universo`, `parametros_libres`):

```yaml
clase_nueva: true                 # la clase no existe en registry.py ni en transversal.METRICAS
rol: diversificador               # independiente | diversificador; no cambia después
tema: flujos-institucionales      # de una lista cerrada (ver 5.4)
fuente_de_retorno: provision-de-liquidez  # prima-de-riesgo | provision-de-liquidez | sesgo-de-comportamiento | friccion
por_que_no_esta_en_el_benchmark: >-
  El benchmark está comprado todo el mes; la cartera cambia de activo solo cuando hay un flujo forzado.
cita_texto:                       # uno por DOI de `fuente`, en el mismo orden
  - Harvey, Mazzoleni y Melone (2025). The Unintended Consequences of Rebalancing
  - Etula, Rinne, Suominen y Vaittinen (2020). Dash for Cash
muestra_de_la_cita: 1997-2023
evidencia_posterior: []           # DOI o URL de evidencia después de publicada, o vacío
fidelidad: igual a la cita        # o "desvío: <cuál>"
datos_nuevos: []                  # [{fuente: ..., costo: ...}] si algún activo o serie no está en los snapshots
efecto_esperado: ΔSharpe +0,2 contra el benchmark, ya con la mitad descontada
```

### 5.3 Chequeos (`validar_propuesta` y `escribir_propuestas`)

1. Campos nuevos presentes y con valores de la lista cerrada.
2. Cada DOI resuelve en Crossref y su título comparte al menos el 60 % de las palabras con `cita_texto`. Este chequeo habría rechazado la segunda cita de `deriva-pre-fomc`. Sin red, la propuesta queda "sin verificar" y no pasa.
3. `universo.activos` está en el snapshot o figura en `datos_nuevos`.
4. `clase` existe o `clase_nueva: true`.
5. Lista negra: si `tema` o `idea` coinciden con un efecto con decaimiento documentado del catálogo, se rechaza salvo que `evidencia_posterior` cite algo publicado después del decaimiento.
6. Cupo por tema: máximo una propuesta por tema por lote, y los temas probados dos veces se bloquean salvo `fidelidad` que corrija un desvío documentado. La originalidad estructural de hoy (misma clase, activos y ventana) es demasiado estrecha: la volatilidad gestionada pasa como "nueva" aunque sea del mismo tema que la baja volatilidad.
7. `muestra_de_la_cita` que termina en 2019 o después marca la propuesta como "sin prueba ciega en 2019–2022".

### 5.4 Catálogo (`catalogo/primas.yaml`)

Campos nuevos por entrada: `tema`, `muestra`, `publicacion`, `fidelidad` (por ejemplo, para `momentum-series-de-tiempo`: "desvío: la cita es largo-corto; acá solo largo") y `decaimiento` (cita o vacío). Temas, de una lista cerrada inspirada en los 13 de Jensen, Kelly y Pedersen más los nuestros: tendencia, momentum transversal, baja volatilidad, valor, carry, calendario, flujos institucionales, prima de volatilidad, reversión y liquidez, macro. Entradas nuevas: las tres familias de §4 y una sección de descartadas por decaimiento (pre-FOMC, deriva nocturna, PEAD en acciones grandes, carry de cripto) con su cita.

### 5.5 Admisión y postmortem

Lo que hoy busca el postmortem (pocos trades, costos, rotación) es útil pero no encontró nada. Hay causas estructurales que se conocen antes de correr y no usan resultados, así que no filtran la prueba. Conviene mostrarlas como advertencias de la etapa 0, en el informe y en el pedido:
- `desvio_de_la_cita`: la implementación es solo larga y la cita es largo-corto, o la frecuencia difiere (sale del campo `fidelidad`).
- `muestra_solapada`: la muestra de la cita termina dentro de la etapa 2.
- `overlay_solo_largo`: todas las piernas son solo largas sobre activos del benchmark.

Con estas tres, las cuatro carteras que cortaron habrían llegado con advertencias antes de abrir la prueba.

### 5.6 Flujo

1. El LLM propone entre 0 y 3 con el pedido nuevo.
2. Un segundo pedido, con otro rol, intenta refutar cada propuesta: busca evidencia de decaimiento, réplicas fallidas, costos y desvíos de la cita. Su texto se guarda junto a la propuesta.
3. Chequeos automáticos de §5.3, con Crossref.
4. Fede aprueba o rechaza con `hipotesis propuestas aprobar|rechazar`, como hoy.

## 6. Plan del lote siguiente

Lote de búsqueda; el número lo asigna el PM después de L18. Filas con prefijo R para no chocar con las existentes. Tipo como en `PLAN-CONTINUACION.md` §4: leer, probar o decidir.

| Fila | Qué | Oráculo | Tipo | Dónde | Depende de |
|---|---|---|---|---|---|
| R01 | Decisiones de §7 y DC nuevo que las registre | DC con cada decisión y su fecha | decidir | doc | — |
| R02 | M1: Sharpe en exceso de DGS3MO para cartera y benchmark en el informe; el registro sigue a tasa 0 | Test sintético: un activo que rinde exactamente la tasa da Sharpe en exceso ≈ 0; cartera y benchmark usan la misma serie; `intentos.csv` sin cambios de formato | leer | nube | R01 |
| R03 | M2 y M3: benchmark de mismos pesos con descomposición de la brecha, y error estándar de ΔSharpe | Tests sintéticos con brecha conocida por construcción; el error estándar coincide con la fórmula en datos i.i.d. | leer | nube | R01 |
| R04 | Atribución de las cuatro carteras con las curvas guardadas: pesos, señal y tasa | Tabla por cartera; md5 de `intentos.csv`, `hipotesis.csv` e `hipotesis-candado.csv` iguales antes y después (como Q01) | decidir | VM | R02, R03 |
| R05 | Catálogo con campos nuevos, temas y descartadas | Carga sin errores; cada entrada tiene `tema` de la lista cerrada | leer | nube | R01 |
| R06 | `armar_pedido` nuevo (§5.1) con inventario armado desde manifiestos y registro | El pedido contiene el criterio, las clases, los activos y la lista negra; sigue sin Sharpe, DSR ni ganancias | leer | nube | R05 |
| R07 | `validar_propuesta` y `escribir_propuestas` nuevos (§5.3) con verificación en Crossref | Rechaza la segunda cita de `deriva-pre-fomc` con una respuesta de Crossref simulada; acepta las tres propuestas de este PR una vez completados sus campos nuevos | probar | nube | R05 |
| R08 | Advertencias de admisión (§5.5) | Las cuatro carteras que cortaron salen con sus advertencias; una propuesta fiel a la cita sale sin ellas | leer | nube | R05 |
| R09 | Si F1 se aprueba: pesos objetivo por fecha en la cartera y clase `rebalanceo-fin-de-mes` | Señal sintética conocida da los pesos esperados; no mira el futuro; el resto del mes la curva coincide con el benchmark 50/50 | leer | nube | R01 |
| R10 | F1 por las etapas 0 a 2 (2007–2018) como réplica | IM con veredicto; ninguna entrada nueva en el candado; pide a Fede el OK para la reserva | decidir | VM | R02, R09 |
| R11 | Si F2 se aprueba: celdas cortas en la cartera y en el control contra `backtesting.py` | Celda corta sintética gana cuando el precio baja; costos simétricos; control coincide | probar | nube | R01 |
| R12 | F2 por las etapas 0 a 2 | IM con veredicto y correlación con 1/N; ninguna entrada nueva en el candado | decidir | VM | R02, R11 |
| R13 | Protocolo de la reserva (lista corta, criterio, una apertura) y de incubación en paper (seguimiento, cortes, duración) | Documento con los números de Fede antes de abrir nada | decidir | doc | R04 |
| R14 | Opcional: datos y broker de futuros para F3 | Comparación de costo y cobertura (Norgate, Databento; IBKR) y capital mínimo por granularidad | decidir | doc | R13 |

Orden: R01; después R02, R03 y R05 en paralelo; R04, R06, R07 y R08; R09 a R12 según lo que se apruebe; R13 antes de abrir la reserva. Ningún ticket de nube toca `local-data`. Solo R04, R10 y R12 corren en la VM.

Recomiendo que `cambio-de-mes` (L18, Q04) no abra su etapa 3 hasta que estén R01 a R03: la apertura es única y conviene que use la medición corregida.

## 7. Decisiones para Fede

1. **Criterio de la etapa 3.** ¿C0 con M1–M3 (actual corregido), C0 más la vía C1 para carteras que declaran `rol: diversificador` antes de correr, o C2? Recomiendo C0 más C1.
2. **Sharpe en exceso de la tasa corta (M1).** ¿Se adopta en el informe antes de abrir cualquier ventana nueva, con el registro de intentos a tasa 0? Recomiendo sí.
3. **Benchmark de mismos pesos (M2).** ¿Se agrega como diagnóstico y 1/N sigue siendo el corte? Recomiendo sí.
4. **Rol de paper.** ¿Paper pasa a ser incubación con éxito por seguimiento y reglas de corte escritas antes? Si es sí, hacen falta tres números: correlación mínima de seguimiento (propongo 0,9), costo máximo contra el supuesto (propongo el doble) y percentil de corte (propongo 5).
5. **Reserva 2023+.** ¿Se abre una sola vez para una lista corta escrita antes? ¿Cuáles y con qué criterio? Propongo `cartera-transversal`, más F1 y `cambio-de-mes` si pasan la etapa 2, con el criterio de la pregunta 1 y M1.
6. **`cambio-de-mes` (L18, Q04).** ¿Su etapa 3 espera a R01–R03? Recomiendo que espere.
7. **F1 `rebalanceo-fin-de-mes`.** ¿Se aprueba, con la etapa 2 leída como réplica y la decisión en la reserva? Recomiendo sí, después de verificar en el texto del paper la definición de la cuarta semana.
8. **F2 `tendencia-larga-corta`.** ¿Se aprueba como diversificador? ¿Cuenta como variante de la tendencia (consume cupo) o como corrección de fidelidad a la cita (en N cuenta igual)? ¿Se habilitan cortos en el motor y una cuenta margin en Alpaca? Recomiendo sí a las tres, contándola como variante.
9. **F3 `carry-futuros`.** ¿Se abre ahora la discusión de datos pagos y broker de futuros, o se espera a que una cartera con datos actuales llegue a paper? Recomiendo esperar.
10. **`volatilidad-gestionada` (pendiente desde L17).** ¿Se rechaza, o se deja solo como diversificador con la constante de escala calculada con datos hasta cada fecha? Recomiendo dejarla como diversificador y no correrla antes que F1 y F2.
11. **Lección escrita para el generador.** ¿Entra al catálogo una regla escrita por personas, sin cifras de la prueba, del tipo "overlays solo largos sobre los activos del benchmark no superaron a 1/N"? Filtra un bit de la prueba; recomiendo sí, porque el pedido nuevo ya dice lo mismo desde el criterio.
12. **Generador.** ¿Se adoptan el pedido, los campos y los chequeos de §5 antes del próximo `hipotesis proponer`? Recomiendo sí, y no correr `proponer` hasta tenerlos.
13. **Beads.** ¿Qué ID lleva este trabajo? Propongo crear un issue padre para el lote con R01–R14 como hijos.

## Apéndice A. Simulación del criterio

Retornos diarios normales i.i.d. para estrategia y benchmark, 252 sesiones por año, con Sharpe anual y correlación fijados; 4.000 repeticiones por celda. Etapa 2 de 11 años (2007–2018 menos el año de arranque de las señales) y etapa 3 de 4 años. Supuestos que favorecen al criterio: sin colas gruesas, sin cambios de régimen y sin autocorrelación; con datos reales, los errores son mayores. No usa datos de mercado ni el motor.

```python
import numpy as np

D, rng = 252, np.random.default_rng(7)

def sim(sr_s, sr_b, rho, anios, n):
    z = rng.standard_normal((n, anios * D, 2)) @ np.linalg.cholesky([[1, rho], [rho, 1]]).T
    return z + np.array([sr_s, sr_b]) / np.sqrt(D)

def sharpe(x):
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(D)

def criterio_actual(r2, r3):
    s2, b2, s3, b3 = sharpe(r2[..., 0]), sharpe(r2[..., 1]), sharpe(r3[..., 0]), sharpe(r3[..., 1])
    anios_pos = (r3[..., 0].reshape(len(r3), 4, D).mean(axis=2) > 0).sum(axis=1)
    return (s3 > 0) & (s3 > b3) & (s2 > 0) & (s2 > b2) & (anios_pos >= 3)

for rho in (0.3, 0.6, 0.85):
    for dsr in (-0.2, 0.0, 0.1, 0.2, 0.3, 0.5):
        p = criterio_actual(sim(0.5 + dsr, 0.5, rho, 11, 4000), sim(0.5 + dsr, 0.5, rho, 4, 4000)).mean()
        print(rho, dsr, round(p, 2))
```

C1 reemplaza la comparación de Sharpe por el t del alfa de la regresión de la estrategia contra el benchmark (> 0 en las dos etapas) y pide 3 de 4 años positivos a la mezcla 50/50. C2 pide ΔSharpe estimado > −√((2 − 2ρ̂)/T) en las dos etapas, Sharpe > 0 y 3 de 4 años. El error estándar de paper es 1/√(semanas/52).

## Fuentes

DOIs verificados contra api.crossref.org el 2026-10-20 (título y primer autor coinciden). Las fuentes sin DOI llevan URL.

Decaimiento, réplicas y múltiples pruebas
- Bailey, D. y López de Prado, M. (2014). The Deflated Sharpe Ratio. JPM. [10.3905/jpm.2014.40.5.094](https://doi.org/10.3905/jpm.2014.40.5.094)
- Bailey, D., Borwein, J., López de Prado, M. y Zhu, Q. (2014). Pseudo-Mathematics and Financial Charlatanism. Notices of the AMS. [10.1090/noti1105](https://doi.org/10.1090/noti1105)
- Chordia, T., Goyal, A. y Saretto, A. (2020). Anomalies and False Rejections. RFS. [10.1093/rfs/hhaa018](https://doi.org/10.1093/rfs/hhaa018)
- DeMiguel, V., Garlappi, L. y Uppal, R. (2009). Optimal Versus Naive Diversification. RFS. [10.1093/rfs/hhm075](https://doi.org/10.1093/rfs/hhm075)
- Falck, A., Rej, A. y Thesmar, D. (2022). When do systematic strategies decay? Quantitative Finance. [10.1080/14697688.2022.2098810](https://doi.org/10.1080/14697688.2022.2098810)
- Goyal, A. y Welch, I. (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. RFS. [10.1093/rfs/hhm014](https://doi.org/10.1093/rfs/hhm014)
- Goyal, A., Welch, I. y Zafirov, A. (2024). A Comprehensive 2022 Look at the Empirical Performance of Equity Premium Prediction. RFS. [10.1093/rfs/hhae044](https://doi.org/10.1093/rfs/hhae044)
- Grinold, R. (1989). The Fundamental Law of Active Management. JPM. [10.3905/jpm.1989.409211](https://doi.org/10.3905/jpm.1989.409211)
- Harvey, C. y Liu, Y. (2015). Backtesting. JPM. [10.3905/jpm.2015.42.1.013](https://doi.org/10.3905/jpm.2015.42.1.013)
- Harvey, C., Liu, Y. y Zhu, H. (2016). … and the Cross-Section of Expected Returns. RFS. [10.1093/rfs/hhv059](https://doi.org/10.1093/rfs/hhv059)
- Hou, K., Xue, C. y Zhang, L. (2020). Replicating Anomalies. RFS. [10.1093/rfs/hhy131](https://doi.org/10.1093/rfs/hhy131)
- Huberman, G. y Kandel, S. (1987). Mean-Variance Spanning. JF. [10.1111/j.1540-6261.1987.tb03917.x](https://doi.org/10.1111/j.1540-6261.1987.tb03917.x)
- Jensen, T., Kelly, B. y Pedersen, L. (2023). Is There a Replication Crisis in Finance? JF. [10.1111/jofi.13249](https://doi.org/10.1111/jofi.13249)
- Ledoit, O. y Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. JEF. [10.1016/j.jempfin.2008.03.002](https://doi.org/10.1016/j.jempfin.2008.03.002)
- Linnainmaa, J. y Roberts, M. (2018). The History of the Cross-Section of Stock Returns. RFS. [10.1093/rfs/hhy030](https://doi.org/10.1093/rfs/hhy030)
- McLean, R. y Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? JF. [10.1111/jofi.12365](https://doi.org/10.1111/jofi.12365)
- Novy-Marx, R. y Velikov, M. (2016). A Taxonomy of Anomalies and Their Trading Costs. RFS. [10.1093/rfs/hhv063](https://doi.org/10.1093/rfs/hhv063)
- Sullivan, R., Timmermann, A. y White, H. (1999). Data-Snooping, Technical Trading Rule Performance, and the Bootstrap. JF. [10.1111/0022-1082.00163](https://doi.org/10.1111/0022-1082.00163)
- Sullivan, R., Timmermann, A. y White, H. (2001). Dangers of Data Mining: The Case of Calendar Effects in Stock Returns. J. of Econometrics. [10.1016/S0304-4076(01)00077-X](https://doi.org/10.1016/S0304-4076(01)00077-X)
- White, H. (2000). A Reality Check for Data Snooping. Econometrica. [10.1111/1468-0262.00152](https://doi.org/10.1111/1468-0262.00152)

Tendencia, momentum y carry
- Asness, C., Moskowitz, T. y Pedersen, L. (2013). Value and Momentum Everywhere. JF. [10.1111/jofi.12021](https://doi.org/10.1111/jofi.12021)
- Babu, A., Levine, A., Ooi, Y. H., Pedersen, L. y Stamelos, E. (2019). You Can't Always Trend When You Want. SSRN. [10.2139/ssrn.3487134](https://doi.org/10.2139/ssrn.3487134)
- Faber, M. (2007). A Quantitative Approach to Tactical Asset Allocation. JWM. [10.3905/jwm.2007.674809](https://doi.org/10.3905/jwm.2007.674809)
- Gorton, G. y Rouwenhorst, K. G. (2006). Facts and Fantasies about Commodity Futures. FAJ. [10.2469/faj.v62.n2.4083](https://doi.org/10.2469/faj.v62.n2.4083)
- Huang, D., Li, J., Wang, L. y Zhou, G. (2020). Time Series Momentum: Is It There? JFE. [10.1016/j.jfineco.2019.08.004](https://doi.org/10.1016/j.jfineco.2019.08.004)
- Hurst, B., Ooi, Y. H. y Pedersen, L. (2017). A Century of Evidence on Trend-Following Investing. JPM. [10.3905/jpm.2017.44.1.015](https://doi.org/10.3905/jpm.2017.44.1.015)
- Jegadeesh, N. y Titman, S. (1993). Returns to Buying Winners and Selling Losers. JF. [10.1111/j.1540-6261.1993.tb04702.x](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
- Kim, A., Tse, Y. y Wald, J. (2016). Time Series Momentum and Volatility Scaling. JFM. [10.1016/j.finmar.2016.05.003](https://doi.org/10.1016/j.finmar.2016.05.003)
- Koijen, R., Moskowitz, T., Pedersen, L. y Vrugt, E. (2018). Carry. JFE. [10.1016/j.jfineco.2017.11.002](https://doi.org/10.1016/j.jfineco.2017.11.002)
- Lustig, H., Roussanov, N. y Verdelhan, A. (2011). Common Risk Factors in Currency Markets. RFS. [10.1093/rfs/hhr068](https://doi.org/10.1093/rfs/hhr068)
- Moskowitz, T., Ooi, Y. H. y Pedersen, L. (2012). Time Series Momentum. JFE. [10.1016/j.jfineco.2011.11.003](https://doi.org/10.1016/j.jfineco.2011.11.003)
- Rattray, S., Granger, N., Harvey, C. y Van Hemert, O. (2020). Strategic Rebalancing. JPM. [10.3905/jpm.2020.1.150](https://doi.org/10.3905/jpm.2020.1.150)
- Szymanowska, M., de Roon, F., Nijman, T. y van den Goorbergh, R. (2014). An Anatomy of Commodity Futures Risk Premia. JF. [10.1111/jofi.12096](https://doi.org/10.1111/jofi.12096)
- SG Trend Index 2023 y 2025 (fuente secundaria): [Top Traders Unplugged, diciembre de 2023](https://www.toptradersunplugged.com/?p=14211), [enero de 2025](https://www.toptradersunplugged.com/trend-following-performance-report-january-2025/)

Calendario y flujos
- Bouman, S. y Jacobsen, B. (2002). The Halloween Indicator. AER. [10.1257/000282802762024683](https://doi.org/10.1257/000282802762024683)
- Cieslak, A., Morse, A. y Vissing-Jorgensen, A. (2019). Stock Returns over the FOMC Cycle. JF. [10.1111/jofi.12818](https://doi.org/10.1111/jofi.12818)
- Etula, E., Rinne, K., Suominen, M. y Vaittinen, L. (2020). Dash for Cash: Monthly Market Impact of Institutional Liquidity Needs. RFS. [10.1093/rfs/hhz054](https://doi.org/10.1093/rfs/hhz054)
- Harvey, C., Mazzoleni, M. y Melone, A. (2025). The Unintended Consequences of Rebalancing. NBER WP 33554. [10.3386/w33554](https://doi.org/10.3386/w33554); presentación en EDHEC, febrero de 2026: [slides](https://www.edhec.edu/sites/default/files/2026-03/slides_EDHEC_Michele%20Mazzoleni%20%281%29.pdf)
- Kurov, A., Wolfe, M. y Gilbert, T. (2021). The Disappearing Pre-FOMC Announcement Drift. FRL. [10.1016/j.frl.2020.101781](https://doi.org/10.1016/j.frl.2020.101781)
- Lakonishok, J. y Smidt, S. (1988). Are Seasonal Anomalies Real? RFS. [10.1093/rfs/1.4.403](https://doi.org/10.1093/rfs/1.4.403)
- Lucca, D. y Moench, E. (2015). The Pre-FOMC Announcement Drift. JF. [10.1111/jofi.12196](https://doi.org/10.1111/jofi.12196)
- McConnell, J. y Xu, W. (2008). Equity Returns at the Turn of the Month. FAJ. [10.2469/faj.v64.n2.11](https://doi.org/10.2469/faj.v64.n2.11)

Intradía y nocturno
- Baltussen, G., Da, Z., Lammers, S. y Martens, M. (2021). Hedging Demand and Market Intraday Momentum. JFE. [10.1016/j.jfineco.2021.04.029](https://doi.org/10.1016/j.jfineco.2021.04.029)
- Bondarenko, O. y Muravyev, D. (2023). Market Return Around the Clock: A Puzzle. JFQA. [10.1017/s0022109022000783](https://doi.org/10.1017/s0022109022000783)
- Boyarchenko, N., Larsen, L. y Whelan, P. (2023). The Overnight Drift. RFS. [10.1093/rfs/hhad020](https://doi.org/10.1093/rfs/hhad020)
- Boyarchenko, N., Larsen, L. y Whelan, P. (2026). The Disappearing Overnight Drift. Liberty Street Economics. [10.59576/lse.20260701](https://doi.org/10.59576/lse.20260701)
- Gao, L., Han, Y., Li, S. y Zhou, G. (2018). Market Intraday Momentum. JFE. [10.1016/j.jfineco.2018.05.009](https://doi.org/10.1016/j.jfineco.2018.05.009)
- Lou, D., Polk, C. y Skouras, S. (2019). A Tug of War: Overnight versus Intraday Expected Returns. JFE. [10.1016/j.jfineco.2019.03.011](https://doi.org/10.1016/j.jfineco.2019.03.011)

Volatilidad y opciones
- Bakshi, G. y Kapadia, N. (2003). Delta-Hedged Gains and the Negative Market Volatility Risk Premium. RFS. [10.1093/rfs/hhg002](https://doi.org/10.1093/rfs/hhg002)
- Bollerslev, T., Tauchen, G. y Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. RFS. [10.1093/rfs/hhp008](https://doi.org/10.1093/rfs/hhp008)
- Carr, P. y Wu, L. (2009). Variance Risk Premiums. RFS. [10.1093/rfs/hhn038](https://doi.org/10.1093/rfs/hhn038)
- Cederburg, S., O'Doherty, M., Wang, F. y Yan, X. (2020). On the Performance of Volatility-Managed Portfolios. JFE. [10.1016/j.jfineco.2020.04.015](https://doi.org/10.1016/j.jfineco.2020.04.015)
- Harvey, C., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. y Van Hemert, O. (2018). The Impact of Volatility Targeting. JPM. [10.3905/jpm.2018.45.1.014](https://doi.org/10.3905/jpm.2018.45.1.014)
- Ilmanen, A. (2012). Do Financial Markets Reward Buying or Selling Insurance and Lottery Tickets? FAJ. [10.2469/faj.v68.n5.7](https://doi.org/10.2469/faj.v68.n5.7)
- Israelov, R. y Nielsen, L. (2015). Covered Calls Uncovered. FAJ. [10.2469/faj.v71.n6.1](https://doi.org/10.2469/faj.v71.n6.1)
- Johnson, T. (2017). Risk Premia and the VIX Term Structure. JFQA. [10.1017/s0022109017000825](https://doi.org/10.1017/s0022109017000825)
- Moreira, A. y Muir, T. (2017). Volatility-Managed Portfolios. JF. [10.1111/jofi.12513](https://doi.org/10.1111/jofi.12513)
- Cboe: historia diaria del índice PUT, [CSV](https://cdn-api.cboe.com/api/global/us_indices/daily_prices/PUT_History.csv); ficha con retornos anuales, [factsheet](https://cdn.cboe.com/resources/indices/factsheet/CboeGlobalIndices_PUT-Index.pdf); estudio de Bondarenko, [resumen de Cboe](https://www.cboe.com/insights/posts/white-paper-shows-volatility-risk-premium-facilitated-higher-risk-adjusted-returns-for-put-index/) e [Indexology](https://www.indexologyblog.com/2016/02/18/paper-by-professor-bondarenko-has-intriguing-new-analysis-of-put-and-wput-indexes/)

Acciones, factores, cripto y datos alternativos
- Asness, C., Frazzini, A. y Pedersen, L. (2012). Leverage Aversion and Risk Parity. FAJ. [10.2469/faj.v68.n1.1](https://doi.org/10.2469/faj.v68.n1.1)
- Cohen, L., Malloy, C. y Nguyen, Q. (2020). Lazy Prices. JF. [10.1111/jofi.12885](https://doi.org/10.1111/jofi.12885)
- Cohen, L., Malloy, C. y Pomorski, L. (2012). Decoding Inside Information. JF. [10.1111/j.1540-6261.2012.01740.x](https://doi.org/10.1111/j.1540-6261.2012.01740.x)
- Frazzini, A. y Pedersen, L. (2014). Betting Against Beta. JFE. [10.1016/j.jfineco.2013.10.005](https://doi.org/10.1016/j.jfineco.2013.10.005)
- Glasserman, P. y Lin, C. (2023). Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis. [arXiv:2309.17322](https://arxiv.org/abs/2309.17322)
- Liu, Y. y Tsyvinski, A. (2021). Risks and Returns of Cryptocurrency. RFS. [10.1093/rfs/hhaa113](https://doi.org/10.1093/rfs/hhaa113)
- Liu, Y., Tsyvinski, A. y Wu, X. (2022). Common Risk Factors in Cryptocurrency. JF. [10.1111/jofi.13119](https://doi.org/10.1111/jofi.13119)
- Lopez-Lira, A. y Tang, Y. (2026). Can ChatGPT Forecast Stock Price Movements? JFE. [10.1016/j.jfineco.2026.104335](https://doi.org/10.1016/j.jfineco.2026.104335)
- Martineau, C. (2022). Rest in Peace Post-Earnings Announcement Drift. Critical Finance Review. [10.1561/104.00000122](https://doi.org/10.1561/104.00000122)
- Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability Premium. JFE. [10.1016/j.jfineco.2013.01.003](https://doi.org/10.1016/j.jfineco.2013.01.003)
- Rapach, D., Ringgenberg, M. y Zhou, G. (2016). Short Interest and Aggregate Stock Returns. JFE. [10.1016/j.jfineco.2016.03.004](https://doi.org/10.1016/j.jfineco.2016.03.004)
- Schmeling, M., Schrimpf, A. y Todorov, K. (2023, revisado 2025). Crypto Carry. BIS Working Paper 1087. [bis.org/publ/work1087.htm](https://www.bis.org/publ/work1087.htm); decaimiento en 2025: [arXiv:2510.14435](https://arxiv.org/abs/2510.14435)

Datos y operación
- Alpaca: [margin y venta corta](https://docs.alpaca.markets/docs/margin-and-short-selling), [sin costo de préstamo en ETB](https://alpaca.markets/blog/zero-borrow-fees-on-short-selling-etb-stock-shares-alpaca-trading-api/), [opciones](https://docs.alpaca.markets/docs/options-trading-overview), [planes de datos](https://docs.alpaca.markets/docs/about-market-data-api), [productos y hoja de ruta](https://docs.alpaca.markets/docs/about-alpaca)
- Databento: [precios](https://databento.com/pricing), [créditos y cobro por uso](https://databento.com/docs/knowledge-base/new-users/usage-based-pricing-and-data-credits)
- Norgate Data: [paquetes de acciones](https://norgatedata.com/stockmarketpackages.php), [suscripción y futuros](https://norgatedata.com/subscribe)
