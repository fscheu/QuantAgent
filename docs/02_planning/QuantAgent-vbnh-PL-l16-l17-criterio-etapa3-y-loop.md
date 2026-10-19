# QuantAgent-vbnh: plan de L16 (criterio de la etapa 3) y L17 (loop investigador)

- **Estado:** plan y recomendaciones de §6 aprobados por Fede el 2026-10-19. No abre los lotes ni toca `.beads/issues.jsonl`: la VM crea los tickets y las ramas si Fede aprueba. Los códigos `E01` a `E08` e `I01` a `I08` son del plan; BEADS les asigna su id.
- **Origen:** [`2026-10-19-DC-criterio-etapa3-y-loop-de-mejora`](../04_decisions/2026-10-19-DC-criterio-etapa3-y-loop-de-mejora.md), decidido por Fede el 2026-10-19 (mergeado en #322). Bug: `QuantAgent-vbnh`.
- **Reemplaza:** nada. Reescribe como `I04` e `I05` los tickets `QuantAgent-d4cq.42` y `.43` (LLM investigador, aprobado por Fede el 2026-10-19), según [`QuantAgent-d4cq-PL-l13-l14-reescritura-de-tickets`](./QuantAgent-d4cq-PL-l13-l14-reescritura-de-tickets.md) líneas 186-188.
- **No fija dinero:** el tamaño de posición y las semanas de paper salen del DC (50 % / 75 % y 12 / 6 semanas); acá solo se calculan e informan, no se aplican.

## 1. Qué decide el DC y qué falta construir

| Decisión del DC | Hoy en el código | Falta |
|---|---|---|
| El DSR se informa y no corta | `validacion.py` `validar_etapa3` corta con `dsr >= 0.95` (cartera y por activo) | Criterio nuevo de pase (E04) |
| Registro limpio | `local-data/intentos.csv` tiene 10 filas `sintetico` con Sharpe ~1e13; `verdict.py:dsr_contra_intentos` las usa | Limpieza (E01) y que no vuelva a pasar (E02) |
| N efectivo junto al N crudo | `dsr_contra_intentos` usa `N = len(filas)` | `n_efectivo` y DSR con ese N (E03) |
| Sharpe positivo en ≥ 3 de 4 años del holdout | no existe | Función de sub-períodos (E05) |
| Recargo por DSR bajo (tamaño y semanas de paper) | no existe | Función `recargo_paper` (E06) |
| Informe con todo eso | `informe.py` muestra DSR e intentos | Columnas nuevas (E07) |
| Loop: postmortem sin Sharpe, ideas de literatura, originalidad, cuota de 2 variantes, nada corre solo | no existe | L17 |

## 2. Orden de lotes y choques

1. **L16 `lote/criterio-etapa3` va primero**, porque todo veredicto posterior depende de que el DSR sea correcto (`vbnh`). Sale de `main`.
2. **L17 `lote/loop-investigador` va después y espera #277** (L13): el catálogo `catalogo/primas.yaml`, `catalogo/propuestas/` y el crítico (`P01`, `P02`) están en `lote/catalogo-de-primas`, no en `main`.
3. **Choque con L13/L14.** `P05` (en #277) toca `validacion.py`, `filtro.py` y `cartera.py`; M05 (en #278) toca `cli/hipotesis.py` e `informe.py`. E04 y E07 tocan `validacion.py` e `informe.py`. Por eso los ticket puros (E01, E02, E03, E05, E06) se pueden hacer en paralelo con #277 y #278; **E04 y E07 esperan a que #277 esté mergeado** y salen de `main` actualizado.
4. **Regla de aceptación de la capa macro (L14).** M07 usa ΔDSR con margen 0,05. Con el DSR pasando a indicador, esa regla debería leer también ΔSharpe y los sub-períodos. No se cambia acá; queda como pregunta Q5.

## 3. L16 · `lote/criterio-etapa3`

**Punto de control.** `quantagent backtest hipotesis informe <id>` de una cartera ya validada muestra: DSR con N crudo y con N efectivo, Sharpe por año del holdout, el recargo de paper (tamaño y semanas) y el veredicto según el criterio del DC. Corrido sobre `cartera-tendencia-momentum`, `cartera-transversal` y `cartera-estacionalidad` (ver E08).

### E01 · Limpiar `intentos.csv` (`QuantAgent-vbnh`, parte de datos)

- **Objetivo:** copia fechada `local-data/intentos-AAAAMMDD-previo-a-vbnh.csv` y retiro de las 10 filas con snapshot `sintetico` (líneas 88 a 97 al 2026-10-19). Fede aprobó la limpieza en el DC.
- **Aceptación:** [ ] copia fechada creada antes de tocar el archivo; [ ] `grep -c sintetico local-data/intentos.csv` da 0; [ ] el resto de las filas queda igual (mismo orden, mismo contenido); [ ] `python -c` que recalcula el DSR de la etapa 2 de `cartera-transversal` ya no da 0,00 por culpa del SR0.
- **Depende de:** nada. **Toca:** `local-data/intentos.csv` (fuera del repo). **Máquina:** VM. **No toca:** código.

### E02 · Que las pruebas sintéticas no escriban en el registro real (`vbnh`, parte de código)

- **Objetivo:** (1) fixture automática en `tests/conftest.py` que apunta `QUANTAGENT_SNAPSHOT_DIR` a un `tmp_path` en toda la suite, para que `registrar_intento` (`cli/sweep.py:118`) nunca escriba en `local-data/`; (2) `dsr_contra_intentos` ignora Sharpe no finitos y los de snapshot `sintetico`.
- **Aceptación:**
  - [ ] Test: correr `hipotesis correr` con `--snapshot sintetico` deja `local-data/intentos.csv` sin cambios (se compara el hash antes y después sobre un directorio temporal que simula `local-data/`).
  - [ ] Test: un `intentos.csv` con una fila de Sharpe 1e13 y otra `inf` da el mismo DSR que sin esas filas.
  - [ ] Los tests existentes de `tests/test_cli_hipotesis.py` y `tests/test_hipotesis.py` siguen verdes.
  - [ ] ≤ 100 líneas.
- **Depende de:** nada. **Toca:** `tests/conftest.py`, `quantagent/backtesting/verdict.py`, tests. **Máquina:** nube. **Con L13/L14:** paralelo (no toca los archivos de la cadena).

### E03 · N efectivo (nuevo)

- **Objetivo:** función pura `n_efectivo(filas)` en `quantagent/backtesting/verdict.py` y parámetro opcional `n_intentos` en `dsr_contra_intentos` para usarlo. **Regla v1 (recomendada, ver Q2):** cuenta intentos distintos por `(estrategia, params, activo, desde, hasta)`; quita las repeticiones exactas (la misma corrida registrada dos veces) y no afirma nada sobre parecido de curvas. El agrupado por correlación de curvas es v2 y queda fuera hasta guardar las curvas por intento.
- **Aceptación:**
  - [ ] Test: 5 filas, 2 idénticas, da `n_efectivo = 4`; el N crudo sigue siendo 5.
  - [ ] Test: dos corridas de la misma cartera en etapas 1, 2 y 3 con distintas ventanas cuentan como 3 (distinto `desde`/`hasta`).
  - [ ] `dsr_contra_intentos(..., n_intentos=k)` con `k` menor da un DSR mayor o igual al de N crudo para el mismo Sharpe.
  - [ ] ≤ 100 líneas.
- **Depende de:** E02 (comparte `verdict.py`). **Toca:** `quantagent/backtesting/verdict.py`, tests. **Máquina:** nube. **Con L13/L14:** paralelo.

### E04 · Criterio de pase de la etapa 3 (nuevo)

- **Objetivo:** `validar_etapa3` pasa a decidir con el criterio del DC en vez de `dsr >= 0.95`. Pasa si: Sharpe de cartera > 0; Sharpe de cartera mayor al del benchmark (ver Q1) en etapa 2 y en etapa 3; Sharpe positivo en ≥ 3 de los 4 años del holdout (E05); drawdown dentro del límite V30 vigente. El DSR se calcula con N efectivo (E03) y N crudo, se guarda en la fila y no bloquea. Los motivos de corte dejan de incluir "DSR < 0.95".
- **Aceptación:**
  - [ ] Test con curvas sintéticas: una cartera con DSR bajo pero Sharpe > 0, mayor al benchmark, con 3 de 4 años positivos y drawdown bajo **pasa**.
  - [ ] Test: la misma cartera con solo 2 de 4 años positivos **corta**, con el motivo escrito.
  - [ ] Test: Sharpe de cartera menor al del benchmark corta aunque el DSR sea alto.
  - [ ] Test que falla si se vuelve a poner el corte por DSR (una cartera con DSR 0,00 que cumple lo demás pasa).
  - [ ] El caso por activo (una sola pierna) se decide en Q6; mientras no esté decidido, se mantiene su comportamiento actual.
- **Depende de:** E03, E05, E06 y **#277 mergeado** (toca `validacion.py`). **Toca:** `quantagent/fabrica/validacion.py`, tests. **Máquina:** nube para la función pura, VM para la integración. Probable partición en dos (función de decisión pura; integración), como `.50.1` y `.50.2`.

### E05 · Sharpe por año del holdout (nuevo)

- **Objetivo:** función pura `sharpe_por_anio(equity, fechas)` y `anios_positivos(...)` que devuelve cuántos años del tramo tienen Sharpe > 0.
- **Aceptación:** [ ] test con una curva de 4 años donde 3 suben y 1 baja: `3 de 4`; [ ] un año con menos de 30 sesiones se descarta y se informa; [ ] sin lectura de datos del futuro (recibe la curva ya calculada); [ ] ≤ 100 líneas.
- **Depende de:** nada. **Toca:** `quantagent/fabrica/criterio.py` (nuevo), tests. **Máquina:** nube. **Con L13/L14:** paralelo.

### E06 · Recargo de paper según DSR (nuevo)

- **Objetivo:** función pura `recargo_paper(dsr)` que devuelve `(tamano_pct, semanas)` según el DC: DSR < 0,50: (50, 12); 0,50 ≤ DSR < 0,95: (75, 6); DSR ≥ 0,95: (100, 0 semanas extra). Solo informa; no ejecuta nada ni fija capital.
- **Aceptación:** [ ] test de los tres tramos y de los bordes 0,50 y 0,95; [ ] DSR `None` devuelve el tramo más conservador (50, 12); [ ] ≤ 50 líneas.
- **Depende de:** nada. **Toca:** `quantagent/fabrica/criterio.py`, tests. **Máquina:** nube. **Con L13/L14:** paralelo.

### E07 · Informe con el criterio nuevo (nuevo)

- **Objetivo:** `informe.py` muestra N crudo y N efectivo, DSR con ambos, años positivos del holdout, benchmark, y el recargo de paper; el glosario cambia "pasa con 0,95" por la descripción del criterio.
- **Aceptación:** [ ] test del HTML con una fila de etapa 3: aparecen las cuatro cifras y el recargo; [ ] el glosario ya no dice "pasa con 0,95"; [ ] se lee bien en el teléfono (tabla angosta, una columna por cifra); [ ] ≤ 100 líneas.
- **Depende de:** E04 y **#277 y #278 mergeados** (M05 toca `informe.py`). **Toca:** `quantagent/fabrica/informe.py`, tests. **Máquina:** nube.

### E08 · Punto de control de L16 (nuevo)

- **Objetivo:** recalcular con el criterio nuevo las tres carteras que ya cortaron (tendencia-momentum, transversal, estacionalidad) y publicar el resultado en `docs/06_implementation/QuantAgent-vbnh-IM-criterio-etapa3.md`. **Sin abrir el candado de nuevo:** usa las curvas ya guardadas de las etapas 2 y 3, y la prueba 2019–2022 ya está gastada (ver Q3).
- **Aceptación:** [ ] el informe de cada cartera muestra el veredicto nuevo y el recargo; [ ] el documento dice cuáles pasan y cuáles no, y por qué motivo; [ ] ninguna corrida nueva sobre 2019–2022 (se verifica con el candado por id); [ ] `python -m pytest -q -m oraculo` verde.
- **Depende de:** E01, E04, E07. **Toca:** `docs/06_implementation/`. **Máquina:** VM.

## 4. L17 · `lote/loop-investigador`

**Punto de control.** Un ciclo completo sin correr nada en el embudo: la ficha de postmortem de las carteras de L16, el pedido al investigador, hasta 3 propuestas en `catalogo/propuestas/`, cada una con el chequeo de hipótesis y de originalidad, y un comando que las lista para que Fede las revise. Nada pasa a `hipotesis/` sin su OK.

**Reglas del DC que el lote hace cumplir:** el generador no lee Sharpe ni DSR del holdout; las ideas salen de la literatura; cada propuesta lleva hipótesis económica, cita con DOI verificado, universo fijado antes de correr y 0 parámetros libres; máximo 2 variantes derivadas de una misma idea por lote; todo intento suma al registro.

### I01 · Ficha de postmortem (nuevo)

- **Objetivo:** función pura `ficha_postmortem(h, filas)` que clasifica por qué cortó una cartera en causas estructurales con umbrales fijos: `pocos_trades`, `costos_comen_ganancia`, `rotacion_alta`, `una_pierna_domina`, `drawdown`. Escribe `catalogo/postmortem/<id>.yaml`.
- **Aceptación:**
  - [ ] Test con una cartera sintética de pocos trades: la causa es `pocos_trades`.
  - [ ] **Test de contenido:** el YAML serializado no contiene claves ni valores de Sharpe, DSR ni pnl del holdout (se buscan por nombre de campo y por el valor numérico).
  - [ ] Los umbrales están escritos en el código como constantes con una línea de justificación; no se ajustan por resultados.
  - [ ] ≤ 100 líneas.
- **Depende de:** #277 (usa `catalogo/`). **Toca:** `quantagent/fabrica/postmortem.py` (nuevo), tests. **Máquina:** nube.

### I02 · Chequeo de propuestas (nuevo; amplía `P02`)

- **Objetivo:** validador de una entrada `catalogo/propuestas/<id>.yaml`: exige `hipotesis_economica`, `fuente` con DOI, `universo` con criterio de definición, 0 parámetros libres y el campo `idea` (de qué idea deriva). Cuenta las variantes por `idea` y rechaza la tercera dentro del mismo lote.
- **Aceptación:** [ ] test: propuesta sin DOI se rechaza con el motivo; [ ] test: sin hipótesis económica se rechaza; [ ] test: la tercera variante de la misma `idea` se rechaza y la primera y segunda pasan; [ ] ≤ 100 líneas.
- **Depende de:** #277 (`P01`, `P02`). **Toca:** `quantagent/fabrica/propuestas.py` (nuevo), tests. **Máquina:** nube.

### I03 · Chequeo de originalidad (nuevo)

- **Objetivo:** compara una propuesta con las entradas de catálogo y las hipótesis ya probadas por estructura (clase de señal, universo, ventana) y marca "parecida a `<id>`" si coincide en los tres. v1 solo estructural; el agrupado por correlación de curvas queda para cuando haya curvas guardadas (comparte el problema de E03 v2).
- **Aceptación:** [ ] test: una propuesta idéntica en señal, universo y ventana a una existente se marca parecida; [ ] test: cambiar solo el universo a otra clase de activo no la marca; [ ] una propuesta marcada parecida no se corre y su motivo queda escrito; [ ] ≤ 100 líneas.
- **Depende de:** I02. **Toca:** `quantagent/fabrica/propuestas.py`, tests. **Máquina:** nube.

### I04 · Investigador-1: el pedido y la escritura de propuestas (reescribe `QuantAgent-d4cq.42`)

- **Objetivo:** arma el pedido para el LLM con el catálogo, las reglas D1 y D2 del DC, el N de intentos y las fichas de I01 (sin Sharpe ni trades), y escribe cada propuesta devuelta como entrada en `catalogo/propuestas/<id>.yaml`, nunca en `hipotesis/`.
- **Aceptación:** [ ] test con una respuesta del LLM simulada: se escriben N archivos de propuesta; [ ] **test de contenido:** el texto del pedido no contiene ninguna cifra de Sharpe, DSR ni pnl; [ ] una respuesta que no es un YAML válido no escribe nada y devuelve el motivo; [ ] ≤ 100 líneas.
- **Depende de:** I01, I02. **Toca:** `quantagent/fabrica/investigador.py` (nuevo), tests. **Máquina:** nube. El LLM real no se llama en la nube.

### I05 · Investigador-2: el comando `proponer` (reescribe `QuantAgent-d4cq.43`)

- **Objetivo:** `quantagent backtest hipotesis proponer` llama al LLM vía `QUANTAGENT_LLM_CMD`, pasa la respuesta por I04, I02 e I03 y lista qué entró y qué se rechazó con su motivo.
- **Aceptación:** [ ] test con un `QUANTAGENT_LLM_CMD` de mentira (un script que imprime un YAML): el comando lista la propuesta aceptada; [ ] con un comando que falla, sale con un error claro y no escribe nada; [ ] no se ejecuta ninguna hipótesis; [ ] ≤ 100 líneas.
- **Depende de:** I04, I03 y **#277 mergeado** (toca `cli/hipotesis.py`). **Toca:** `quantagent/cli/hipotesis.py`, tests. **Máquina:** VM.

### I06 · Revisión humana de propuestas (nuevo)

- **Objetivo:** `hipotesis propuestas aprobar <id>` mueve la propuesta de `catalogo/propuestas/` a `hipotesis/` y deja el registro del OK con fecha y quién; `rechazar <id> --motivo` la archiva con el motivo. Sin aprobación no se corre nada.
- **Aceptación:** [ ] test: aprobar mueve el archivo y escribe el registro; [ ] test: una propuesta no aprobada no aparece en `hipotesis/` ni la toma `tanda`; [ ] rechazar sin motivo falla; [ ] ≤ 100 líneas.
- **Depende de:** I05. **Toca:** `quantagent/cli/hipotesis.py`, tests. **Máquina:** nube para la función, VM para el comando.

### I07 · Retrospectiva por lote (nuevo)

- **Objetivo:** script que arma `docs/06_implementation/retrospectiva-<lote>.md` con una tabla simple de causas de corte por cartera (de las fichas de I01) y cuántos intentos sumó el lote al registro. Sin Sharpe ni DSR en la tabla.
- **Aceptación:** [ ] test con 3 fichas sintéticas: la tabla cuenta cada causa; [ ] la salida no contiene cifras de Sharpe ni DSR; [ ] ≤ 100 líneas.
- **Depende de:** I01. **Toca:** `scripts/retrospectiva_lote.py` (nuevo), tests. **Máquina:** nube.

### I08 · Punto de control de L17 (nuevo)

- **Objetivo:** correr el ciclo del punto de control sobre las tres carteras de L16 y entregar a Fede la lista de propuestas (hasta 3) para que decida.
- **Aceptación:** [ ] fichas de postmortem de las tres carteras; [ ] hasta 3 propuestas en `catalogo/propuestas/`, cada una con el resultado de I02 e I03; [ ] retrospectiva del lote L16 generada; [ ] ninguna hipótesis en `hipotesis/` que Fede no haya aprobado.
- **Depende de:** I05, I06, I07. **Toca:** `catalogo/`, `docs/06_implementation/`. **Máquina:** VM.

## 5. Mapa de máquinas y paralelismo

| Ticket | Lote | Máquina | Con #277 / #278 |
|---|---|---|---|
| E01 limpiar CSV | L16 | VM | paralelo (datos) |
| E02 aislar registro y DSR robusto | L16 | nube | paralelo |
| E03 N efectivo | L16 | nube | paralelo |
| E05, E06 funciones puras | L16 | nube | paralelo |
| E04 criterio de pase | L16 | nube + VM | **espera #277** |
| E07 informe | L16 | nube | **espera #277 y #278** |
| E08 punto de control | L16 | VM | espera E04 y E07 |
| I01, I02, I03, I04, I07 | L17 | nube | **espera #277** (catálogo) |
| I05, I06 | L17 | VM | **espera #277** |
| I08 punto de control | L17 | VM | espera I05, I06, I07 |

L16 son 8 tickets y L17 son 8, dentro del rango de 5 a 10 por lote (PLAN-CONTINUACION §3.7).

## 6. Decisiones de Fede (2026-10-19)

Fede leyó el plan y aprobó las seis recomendaciones. Quedan como decididas:

- **Q1 · ¿Qué es el "control" del criterio?** En el repo, "control" es la comparación del motor propio contra la referencia (`compare_control.py`), que no sirve como vara de rendimiento. **Decidido:** benchmark = misma cartera de activos con pesos iguales, comprada y mantenida con rebalanceo mensual, sin señal.
- **Q2 · Regla v1 de N efectivo (E03).** **Decidido:** quitar solo las repeticiones exactas por `(estrategia, params, activo, desde, hasta)`. Baja poco el N (de 109 a unos 100). El agrupado por correlación baja más y requiere guardar curvas por intento: sería un ticket aparte.
- **Q3 · Recalcular las tres carteras (E08).** Ya gastaron su holdout 2019–2022, así que el recálculo no revela nada nuevo, pero un PASA con el criterio nuevo sale de datos ya vistos. **Decidido:** hacerlo como informativo y que el paper trading sea la vara fresca, como dice el DC.
- **Q4 · Orden.** L16 primero y L17 después de #277, con E02/E03/E05/E06 en paralelo mientras #277 y #278 esperan tu revisión.
- **Q5 · Capa macro (L14).** M07 acepta o descarta una capa con ΔDSR ≥ 0,05. Con el DSR como indicador, **Decidido:** la regla de aceptación de la capa debe leer también ΔSharpe y los años positivos. El cambio no está en L16 ni L17: va como ticket aparte en el lote de L14 cuando se retome.
- **Q6 · Hipótesis de una sola pierna.** `validar_etapa3` tiene un camino por activo (sin cartera). **Decidido:** mantenerlo como está hasta que haya una hipótesis de una sola pierna que lo necesite.

## 7. Qué queda fuera

- Agrupado por correlación de curvas (E03 v2 e I03 v2): requiere guardar la curva de cada intento.
- Aplicar el recargo de tamaño y semanas en paper trading: es de M4, no de este lote.
- Cambios a `PLAN-CONTINUACION.md` (cola §4): se hacen después de que #277 esté mergeado, o solo en la rama del lote.
