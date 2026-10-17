# Cartera de primas de riesgo: qué se prueba, a qué nivel y con qué datos

Fecha: 2026-10-17. Estado: **propuesta, espera decisión de Fede**. No cambia código.
Origen: conversación de Fede con el PM (fork de la sesión PM) al cerrar L11 (PR #239).
Relacionado: [`QuantAgent-ljoj`](../02_planning/QuantAgent-ljoj-PL-regimenes-y-macro.md) (régimen y macro),
[retrospectiva y "oficina de inversión"](../02_planning/2026-09-16_retrospectiva_autodev_y_plan_oficina_inversion.md)
§3 (capas L1 Macro a L4 Estrategia), [`QuantAgent-iuf`](./QuantAgent-iuf-DC-datos-m2.md) y
[`QuantAgent-7nle`](./QuantAgent-7nle-DC-datos-para-operar-m3.md) (fuentes de datos),
[`QuantAgent-d4cq`](./QuantAgent-d4cq-DC-motor-de-la-fabrica.md) (fábrica y embudo).

## Resumen

Hoy el embudo prueba **una estrategia contra los 12 ETFs, activo por activo**. La propuesta es probar
**carteras**: estrategias de primas de riesgo documentadas, cada una sobre el universo donde está definida, combinadas
con pesos por igual riesgo y juzgadas en conjunto. Lo que protege contra engañarse (registro de intentos, walk-forward,
candado, reserva desde 2023, Sharpe deflactado) no cambia.

| # | Decisión | Recomendación del PM |
|---|---|---|
| D1 | Qué estrategias entran a la cola | Primas de riesgo con literatura; no combinaciones de indicadores |
| D2 | Universo de cada estrategia | Fijado en el YAML con cita, antes de correr; nunca elegido por resultados |
| D3 | Nivel del veredicto | La cartera; los trades se cuentan sumando todo |
| D4 | Pesos dentro de la cartera | Igual riesgo (volatilidad); sin optimizar pesos |
| D5 | Capa macro | Segunda capa, hipótesis aparte con datos punto en el tiempo; se mide contra el núcleo |
| D6 | Datos | Más ETFs de Yahoo (unos 35) y FRED/ALFRED; Alpaca sigue para paper; futuros más adelante |

## Por qué (evidencia)

- **Dos fuentes de ganancia.** Las primas de riesgo (tendencia, momentum, carry, value) son un pago por cargar un
  riesgo, con evidencia larga y en muchos mercados: Hurst, Ooi y Pedersen, *A Century of Evidence on
  Trend-Following* (2017); Moskowitz, Ooi y Pedersen, *Time Series Momentum* (2012, 58 futuros); Asness, Moskowitz y
  Pedersen, *Value and Momentum Everywhere* (2013). El alpha de ineficiencias (arbitraje estadístico, alta frecuencia)
  exige datos e infraestructura fuera de alcance, y decae al publicarse: McLean y Pontiff (2016).
- **Combinar indicadores fabrica falsos positivos.** Sullivan, Timmermann y White (1999): corregida la búsqueda entre
  miles de reglas técnicas, la mejor no supera al mercado después de 1986. Harvey, Liu y Zhu (2016): la mayoría de los
  factores publicados no pasan un umbral corregido. Bailey y López de Prado: sobreajuste de backtests.
- **La ventaja se arma con amplitud.** Grinold y Kahn (*Active Portfolio Management*): el ratio de información crece
  con la raíz de la cantidad de apuestas independientes. Una prima gana poco por activo y se ve en la cartera.
- **Evidencia propia (L11, I10).** `momentum-12m` tiene Sharpe ≥ 0,10 en 7 de 12 activos y `maximo-52-semanas` en 9
  de 12, pero cortan en la etapa 1 porque hacen 4 a 14 trades por activo en 12 años y la regla pide 30 por activo.
  `rsi` y `sma` (reglas técnicas) no mostraron esa señal.

## D1. Qué entra a la cola

- **A (recomendada).** Solo hipótesis con una prima documentada y su cita: tendencia (series de tiempo), momentum
  (12 meses, 52 semanas), y más adelante carry y value cuando haya datos (D6).
- B. Seguir sumando reglas técnicas además de las primas. Cada una suma intentos y sube la vara del Sharpe deflactado
  para todas.

## D2. Universo por estrategia

El YAML de cada hipótesis declara su universo con cita, antes de la primera corrida. El criterio es **dónde está
definida** la estrategia, no dónde le fue mejor: tendencia en cualquier activo líquido; carry solo donde hay un
diferencial de tasa (bonos, monedas, futuros); value solo con una medida de valuación. Elegir activos mirando los
backtests propios es sobreajuste y queda prohibido. Retoma la idea de "activos por hipótesis" anotada para L12.

## D3. Veredicto a nivel cartera

- **A (recomendada).** El embudo evalúa la cartera (estrategias × universos con los pesos de D4): Sharpe, drawdown y
  Sharpe deflactado sobre la curva combinada. El mínimo de 30 trades se cuenta sumando toda la cartera. Cada cartera
  probada es un intento para el Sharpe deflactado.
- B. Mantener el veredicto por activo y solo bajar el mínimo de trades para estrategias lentas. Más barato, pero sigue
  pidiendo que cada activo gane solo, que es lo que estas primas no prometen.

Sigue pendiente unificar el Sharpe guardado en `intentos.csv` (la etapa 1 guarda tasa 0; las etapas 2 y 3 restan la
tasa libre de riesgo). Con el veredicto en la cartera hace falta igual: el Sharpe deflactado usa esa varianza.

## D4. Pesos

- **A (recomendada).** Igual riesgo: cada estrategia y cada activo aportan la misma volatilidad esperada, medida solo
  con datos pasados (Carver, *Systematic Trading*). Reglas fijas, sin parámetros que ajustar.
- B. Mismos dólares. Más simple; los bonos cortos casi no aportan y las materias primas dominan el riesgo.
- C. Pesos optimizados (media-varianza). Descartada: es otra forma de sobreajuste.

## D5. Capa macro

Orden por capas: (1) núcleo de primas sin predicción; (2) capa macro como hipótesis aparte; (3) un LLM leyendo
noticias, solo hacia adelante en paper.

- La capa macro usa reglas escritas sobre datos **punto en el tiempo** (ALFRED guarda cada versión publicada). Ejemplo
  con literatura: la pendiente de la curva anticipa el exceso de retorno de los bonos (Fama y Bliss 1987; Cochrane y
  Piazzesi 2005).
- Se acepta solo si mejora el Sharpe deflactado de la cartera respecto del núcleo solo. La evidencia pide humildad:
  Goyal y Welch (2008) encuentran que casi ninguna variable macro predice acciones fuera de muestra; Campbell y
  Thompson (2008), que aporta poco y con reglas prudentes.
- Un LLM no se puede backtestear en 2019–2022: conoce lo que pasó (sesgo de mirada al futuro del preentrenamiento).
- `QuantAgent-ljoj` (régimen medido en el mercado, sin macro) sigue siendo el primer paso barato de esta capa.

## D6. Datos

| Fuente | Para qué | Historia | Costo |
|---|---|---|---|
| Yahoo, ETFs diarios | Ampliar de 12 a unos 35 activos multiactivo | Completa | Gratis; ya integrado (uso personal) |
| FRED / ALFRED | Tasas, curva, inflación, empleo para D5 | Larga, con versiones | Gratis, API con clave |
| Alpaca | Paper trading (M3, ya decidido en `7nle`) | Acciones desde 2016 | Gratis |
| Futuros continuos | El estándar de la literatura (carry, más mercados) | Larga | Pago; solo si la cartera con ETFs lo justifica |

- **Alpaca no sirve para investigar**: arranca en 2016 y el esquema entrena con 2007–2018 (`iuf`, opción C).
  Adelantarla solo tiene sentido para correr el núcleo en paper en paralelo, no para ampliar la historia.
- **Universo según lo que existía en 2007**, no lo que existe hoy, para no quedarse solo con los sobrevivientes.
  Candidatos a verificar contra el snapshot: bonos (IEF, TIP, LQD, AGG, HYG), materias primas (SLV, USO, DBA),
  monedas (UUP, FXE, FXY), acciones por región y sector (EWJ, EWZ, FXI, QQQ, resto de los sectores SPDR), REITs (VNQ).
- Los términos de Yahoo valen para uso personal; con dinero de terceros hay que cambiar de fuente.

## Qué implicaría para L12 (boceto, a confirmar)

1. Unificar el Sharpe de `intentos.csv` (semántica; afecta al Sharpe deflactado).
2. Universo por hipótesis en el YAML, con cita.
3. Snapshot ampliado a unos 35 ETFs que existían en 2007.
4. Pesos por igual riesgo y curva de cartera.
5. Veredicto de cartera en las etapas 1 a 3 (reemplaza el corte por activo).
6. Primera cartera: tendencia y momentum sobre el universo ampliado; prueba 2019–2022 solo con OK de Fede.

FRED/ALFRED y la capa macro quedan para el lote siguiente.

## Decisiones de Fede

Pendientes.
