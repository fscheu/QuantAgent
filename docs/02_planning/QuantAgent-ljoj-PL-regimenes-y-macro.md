# QuantAgent-ljoj: régimen de mercado, macro y elección de estrategia

Fecha: 2026-10-14 · Estado: idea registrada para M3 y siguientes; un ticket de investigación (`QuantAgent-ljoj`).
Origen: conversación de Fede con el PM. Este documento guarda la idea, el análisis y las referencias para retomarlas
cuando se avance con la capa macro (L1 de la "oficina de inversión").

Relacionado:
- `docs/02_planning/2026-09-16_retrospectiva_autodev_y_plan_oficina_inversion.md` §3: capas L1 Macro y L2 Asignación.
- `docs/02_planning/2026-09-17_borrador_skill_revision_macro_semanal.md` y `QuantAgent-wwi`: nota macro semanal (L1 sin código).
- `docs/02_planning/phase2_roadmap.md` §2E: sketch original Macro Agent → Sector Agent → Portfolio Optimizer.

## 1. La idea

Una etapa analiza variables macroeconómicas y estima qué sectores o regiones se ven favorecidos y cuáles perjudicados.
De ahí salen activos con una hipótesis de movimiento fuerte y dirección. Con ese esquema (ciclo económico + activo)
se elige qué tipo de estrategia técnica se adapta mejor, en lugar de hacer el backtest de todas las estrategias en
todos los activos.

## 2. Análisis del PM

**Lo que tiene sustento.** El rendimiento de cada tipo de estrategia depende del régimen. El seguimiento de tendencia
(momentum de series de tiempo) gana en movimientos largos y en crisis: Hurst, Ooi y Pedersen lo muestran en un siglo
de datos, con buen resultado en 8 de las 10 peores crisis de un portafolio 60/40. La reversión a la media rinde mejor
en mercados laterales. El momentum sufre en los rebotes que siguen a un mercado bajista (Daniel y Moskowitz,
"Momentum crashes"). No se encontró un estudio que compare directamente tendencia contra reversión por régimen de
volatilidad: ese vínculo es plausible pero inferido.

**Lo que es débil.** La rotación sectorial guiada por el ciclo económico. Stangl, Jacobsen y Visaltanachoti
(1948–2007): aun con el ciclo fechado a la perfección y sin costos, el mejor caso es unos 2 puntos anuales de alfa,
que desaparecen con errores de timing realistas y costos. Cambiar a efectivo al entrar en recesión rinde más que rotar
sectores. Molchanov y Stangl (2023) no encuentran el patrón sectorial que la creencia popular espera.

**Riesgos y falacias a evitar.**
1. El mercado se adelanta a la macro: tener razón sobre la macro no alcanza, hay que tenerla antes o distinto que el
   consenso.
2. No reemplaza el backtest, lo hace más difícil: la regla régimen → estrategia es un modelo con parámetros y se
   valida con pocos episodios (en 2007–2022 hay tres o cuatro). Más riesgo de encontrar una regla que encaja por
   casualidad, no menos (ver V35, Sharpe deflactado).
3. Hipótesis encadenadas: el error de cada eslabón se multiplica.
4. Datos macro revisados: un backtest con la serie revisada usa información que no existía. Hace falta la versión
   publicada en cada fecha (ALFRED).
5. Una historia convincente no es una predicción.

**Propuesta.** Empezar por regímenes medidos en el propio mercado (volatilidad, tendencia, correlación): sin demora,
sin revisiones y con muchos más episodios que los ciclos económicos. La macro entra después, como filtro lento del
universo o de la exposición total, no para elegir la técnica. Primer paso barato con datos que ya existen:
`QuantAgent-ljoj` (§4).

## 3. Referencias para la capa macro (inspiración, no dependencias)

Las cifras de rendimiento de productos comerciales son de sus propios autores y no están verificadas de forma
independiente.

**Macro → señales de trading**
- **J.P. Morgan Macrosynergy Quantamental System (JPMaQS).** Indicadores macro diarios "point-in-time" (lo que se
  sabía en cada fecha, sin revisiones posteriores) para crecimiento, inflación, balances externos, etc., en muchos
  países, pensados para backtests de renta fija, monedas, acciones y materias primas. Comercial, para institucionales;
  hay datos de muestra gratuitos para investigación académica. Paquete Python `macrosynergy` y una academia con
  notebooks. Lo más útil como inspiración: el diseño point-in-time y los indicadores compuestos simples sin optimizar.
  [J.P. Morgan](https://www.jpmorgan.com/insights/global-research/jpmaqs) ·
  [Quantamental Academy](https://academy.macrosynergy.com) ·
  [Handbook (PDF)](https://macrosynergy.com/wp-content/uploads/2025/04/Macrosynergy-Macro-Quantimental-Investment-Handbook.pdf)
- **Hedgeye GIP Model ("Quads").** Clasifica la economía en cuatro cuadrantes según si crecimiento (PBI) e inflación
  (IPC) aceleran o desaceleran, con una respuesta de política típica por cuadrante, y asocia a cada uno ganadores y
  perdedores por clase de activo, sector y factor. Producto comercial; el historial es afirmación de la firma.
  Inspiración: la clasificación por tasa de cambio, simple y explicable.
  [Entrevista a Keith McCullough](https://www.cmcmarkets.com/en/opto/keith-mccullough-on-using-growth-and-inflation-quads-and-debunking-old-wall-misinformation) ·
  [Cómo modelan la economía](https://app.hedgeye.com/insights/55467-chart-of-the-day-how-we-model-the-u-s-economy/print)
- **Fidelity, Business Cycle Approach to Sector Investing.** Fases del ciclo (temprana, media, tardía, recesión)
  estimadas con probabilidades y asignación sectorial por fase. Aclara que el ciclo no siempre avanza en orden.
  [Fidelity Viewpoints](https://fidelity.com/viewpoints/investing-ideas/sector-investing-business-cycle) ·
  [PDF institucional](https://www.fidelity.com/webcontent/ap101883-markets_sectors-content/18.09.0/business_cycle/Business_Cycle_Sector_Approach.pdf)
- **Merrill Lynch Investment Clock.** Cuatro fases por brecha de producto e inflación, con rotación de activos por
  fase. No se encontró una validación fuera de muestra sólida; un test en Taiwán (1997–2011) da resultados mixtos.
  [Estudio de rotación sectorial (Cranfield)](https://dspace.lib.cranfield.ac.uk/items/91e34a7c-c083-4179-8938-4a02adaf6b55/full)

**Régimen → modelo o estrategia**
- **Two Sigma, "A Machine Learning Approach to Regime Modeling" (Botte y Bao, 2021).** Regímenes detectados por los
  datos con un modelo de mezcla gaussiana sobre los factores de su Factor Lens, en lugar de etiquetas definidas a mano
  (boom/crisis, volatilidad alta/baja, risk-on/off). Inspiración para la etapa en que el régimen deje de ser una regla
  fija. [Artículo](https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling) ·
  [PDF](https://www.twosigma.com/wp-content/uploads/2021/10/Machine-Learning-Approach-to-Regime-Modeling_.pdf)
- **Modelos de cambio de régimen y de Markov ocultos** (Hamilton, 1989). A diferencia de la mezcla gaussiana, tienen
  memoria del régimen anterior. Implementaciones abiertas en Python (por ejemplo `hmmlearn`, `statsmodels`).
- **Meta-labeling (López de Prado).** Un segundo modelo no predice dirección: decide si tomar o no la señal de la
  estrategia primaria. Es la forma más cercana a "usar el contexto para decidir cuándo aplicar una técnica". Riesgo
  conocido: fuga de información entre los dos modelos si no se purga la validación.
  [Hudson & Thames, ejemplo](https://hudsonthames.org/meta-labeling-a-toy-example/) ·
  [¿Suma a la señal?](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy/) ·
  [QuantConnect: no es una bala de plata](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)

**Evidencia académica citada**
- Seguimiento de tendencia en crisis: [Man Group / Oxford-Man](https://www.man.com/insights/trend-following-equity-and-bond-crisis-alpha) ·
  [CXO Advisory, tests de momentum de largo plazo](https://www.cxoadvisory.com/momentum-investing/long-term-tests-of-intrinsic-momentum-across-asset-classes)
- Rotación sectorial: [CXO Advisory, "perfect sector rotation"](https://www.cxoadvisory.com/economic-indicators/perfect-sector-rotation) ·
  [The myth of business cycle sector rotation (Massey)](https://mro.massey.ac.nz/handle/10179/69590)

**Datos macro sin mirada al futuro**
- **ALFRED (Fed de St. Louis).** Guarda cada versión publicada de las series de FRED: permite reconstruir qué se sabía
  en cada fecha. API gratuita con clave. No todas las fuentes originales publican versiones previas.
  [ALFRED a los 15 años](https://fredblog.stlouisfed.org/2021/04/alfred-at-15-archiving-fred-data-since-2006/)

## 4. Ticket de M3

`QuantAgent-ljoj` (decidir, ~100 líneas, después de L7): ¿cambia el resultado de cada estrategia según el régimen de
mercado? Dos regímenes de SPY medidos sin mirar el futuro (volatilidad realizada a 20 días contra su mediana de un
año; precio sobre o bajo la media de 200 días), trades de las 48 celdas de 2007–2018 asignados al régimen del día de
entrada, y una tabla estrategia × régimen. Si la diferencia es grande y del mismo signo en la mayoría de los ETFs, se
sigue con una regla de asignación validada fuera de muestra como una estrategia más. Si no, la idea se archiva antes
de construir la capa macro.
