# QuantAgent-d4cq: plan M3, fábrica de estrategias

- **Estado:** borrador del PM para revisión de Fede (2026-10-14)
- **Antes:** M2 cerrado con el merge de L7 (PR #132). El validador funciona, pero ninguna de las 4 estrategias con sus parámetros por defecto tiene ventaja (Sharpe deflactado de 0,25 como máximo, sobre 48 celdas).
- **Pedido de Fede:** "construir un sistema completo para evaluar cientos de estrategias y encontrar ganadoras o por lo menos aprender en el camino".
- **Decisión de motor:** `docs/04_decisions/QuantAgent-d4cq-DC-motor-de-la-fabrica.md`.

## Objetivo

Una hipótesis nueva entra a la fábrica como un archivo, recorre un embudo de etapas cada vez más caras y sale con un veredicto y un informe. La fábrica cuenta todos los intentos y nunca deja abrir dos veces el mismo dato para la misma idea.

Punto de control de M3: **50 hipótesis distintas pasaron por el embudo, cada una tiene informe y el Sharpe deflactado se calcula contra el total de intentos**. Encontrar una ganadora no es condición para cerrar M3, porque eso no depende de nosotros. Lo que sí es condición es que el embudo deje registrado qué se aprendió de cada hipótesis.

## Numeración de hitos (decide Fede)

Propuesta: M3 pasa a ser la fábrica de estrategias y la conexión con Alpaca en papel pasa a M4. Todo lo de M4 y M5 se corre un lugar. Así queda escrito lo que ya decidiste el 2026-10-06: no conectar un broker hasta tener una estrategia que lo justifique. Los tickets de datos de Alpaca (T1–T9, PR #116) quedan en espera hasta M4.

## El embudo

| Etapa | Datos | Qué corre | Corta si | Costo por hipótesis |
|---|---|---|---|---|
| 0. Admisión | ninguno | validación del archivo de hipótesis | falta la justificación o hay más de 6 parámetros libres | nulo |
| 1. Filtro | 2007–2018, 12 ETFs | walk-forward con el motor del filtro (B: `vectorbt`) | Sharpe fuera de muestra del walk-forward < 0,10 en más de 8 de 12 activos | segundos |
| 2. Validación | 2007–2018 | motor propio con control `backtesting.py` | el control no coincide sin una causa escrita, o la regla de V30 falla en todos los activos | minutos |
| 3. Prueba | 2019–2022, una vez por hipótesis | motor propio, veredicto completo | Sharpe deflactado contra el total de intentos < 0,95 | minutos |
| 4. Reserva | 2023 en adelante, solo finalistas | motor propio (+ LEAN como control, opcional) | el veredicto de la reserva da NO PASA | minutos |

Cada etapa escribe una fila en `intentos.csv` con el id de la hipótesis y la etapa. Los umbrales de las etapas 0 y 1 son una propuesta del PM. Tienen que quedar fijados antes de correr la primera hipótesis real, y eso lo decidís vos (L9).

## Lotes

Cada ticket es de 100 líneas o menos, como hasta ahora. Los tickets se escriben en BEADS cuando apruebes este plan.

**L8 · `lote/motor-del-filtro`.** Punto de control: las dos condiciones de la decisión de motor, medidas en la VM.
- Perfilar la estrategia que tarda 115 s por celda y corregir su cuello de botella, con oráculos sin cambios.
- Adaptador de `vectorbt` para las 4 estrategias de M2 (señales nuestras → entradas y salidas de `vectorbt`).
- Medición: tiempo total de 48 celdas y Spearman del orden por Sharpe contra el motor propio.
- Decisión de Fede con los números: B o A.

**L9 · `lote/hipotesis`.** Punto de control: dos hipótesis escritas como archivo corren de punta a punta en la etapa 1.
- Esquema del archivo de hipótesis (YAML): id, autor, justificación económica, clase de estrategia, rango de cada parámetro, activos y timeframe.
- Cargador y admisión (etapa 0), con el tope de parámetros.
- `backtest hipotesis filtrar <archivo>`: etapa 1 con walk-forward por activo, más un registro de intentos con el id de hipótesis.
- Candado por hipótesis: la prueba y la reserva se abren una sola vez por id y la segunda vez falla.
- Decisión de Fede: umbrales de las etapas 0 y 1.

**L10 · `lote/embudo`.** Punto de control: `fabrica correr <archivo>` lleva una hipótesis por las etapas 0 a 3 y se detiene donde corresponde.
- Orquestador de etapas con corte temprano.
- La etapa 2 reutiliza `compare` + `compare_control` + `verdict`.
- El Sharpe deflactado entra al veredicto de la etapa 3, con N = total de intentos de `intentos.csv`.
- Estado por hipótesis (`hipotesis.csv`: etapa alcanzada, motivo de corte, fecha).

**L11 · `lote/informe-y-tanda`.** Punto de control: una tanda nocturna de 10 hipótesis deja 10 informes y un resumen.
- Informe por hipótesis (HTML estático como la página del punto de control de L7: veredicto, indicadores, glosario y por qué cortó).
- Resumen de la tanda: cuántas cortaron en cada etapa y las causas más frecuentes.
- Cola de hipótesis y corrida en lote en la VM, partida para la nube cuando la etapa 1 no tenga Postgres.

**L12 · `lote/generador`.** Punto de control: el generador propone 20 hipótesis admisibles y la fábrica las procesa.
- Catálogo semilla: 30 hipótesis de la literatura (momentum, reversión, estacionalidad, volatilidad, régimen), cada una con su fuente.
- Generador con un LLM: lee el catálogo y los informes anteriores y propone hipótesis nuevas en el formato de L9. Es uno de los cuatro usos de IA generativa que elegiste el 2026-10-06.
- Crítico: revisa cada hipótesis antes de admitirla (justificación, sesgo de supervivencia, mirar el futuro).
- Las estrategias nuevas se escriben como clases contra nuestra interfaz de señales, nunca contra la API de un motor.

## Reglas que no cambian

- 2019–2022 y la reserva se abren una vez por hipótesis. Las 4 estrategias de M2 con sus parámetros por defecto ya la usaron.
- Cada corrida cuenta como intento. El Sharpe deflactado usa el total.
- El motor propio sigue siendo la referencia del veredicto hasta que una decisión diga otra cosa.
- Entregas chicas, PM revisa, Fede revisa lotes.

## Riesgos

1. **El filtro descarta ganadoras.** Se mide con la correlación de L8 y, cada tanto, se pasa una muestra de las descartadas por la etapa 2.
2. **Hipótesis repetidas con otro nombre.** El generador puede proponer variantes de la misma idea. El crítico compara contra el catálogo, y las variantes cuentan como intentos de la misma familia.
3. **Cómputo.** 2 núcleos en la VM. La nube sirve para la etapa 1 si no usa Postgres.
4. **Datos.** 12 ETFs diarios de Yahoo. Ampliar el universo es probablemente un lote propio después de L10.

## Qué decide Fede para arrancar

1. Aprobar este plan, con cambios o sin cambios.
2. La numeración de hitos: fábrica = M3 y Alpaca = M4 (recomendado) | otra.
3. El motor del filtro (documento de decisión), o esperar a los números de L8 (recomendado).
