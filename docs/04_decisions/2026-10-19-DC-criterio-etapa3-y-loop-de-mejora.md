# Criterio de pase de la etapa 3 y loop de investigación de carteras

Fecha: 2026-10-19. Estado: **propuesta, aprobada en lo conceptual por Fede el 2026-10-19; falta su OK sobre los números**. No cambia código.
Origen: sesión de discusión de Fede con Claude. Relacionado: [`2026-10-17-DC-cartera-de-primas`](./2026-10-17-DC-cartera-de-primas.md), [`QuantAgent-d4cq-DC-motor-de-la-fabrica`](./QuantAgent-d4cq-DC-motor-de-la-fabrica.md), `QuantAgent-d4cq.42/.43` (LLM investigador).

## Resumen

Todas las carteras cortaron en etapa 3 con DSR 0,00. Se encontró que (1) el registro `intentos.csv` está contaminado y rompe el DSR, y (2) aun limpio, el DSR ≥ 0,95 sobre 4 años pide un Sharpe anual de ~1,4, que una prima real y honesta (0,3–0,7) no alcanza. Se propone cambiar el rol del DSR y diseñar un loop de mejora que no gaste el holdout.

## Hallazgos (verificados el 2026-10-19)

- `local-data/intentos.csv` líneas 88-97: 10 filas con snapshot `sintetico` y Sharpe de 6,6e13 y 1,3e14. `verdict.py:dsr_contra_intentos` las usa para la varianza entre intentos; la vara por azar (SR0) se vuelve astronómica y todo DSR da 0,00. Las etapas 3 de L12, transversal y estacionalidad corrieron después de esas filas.
- Con filas limpias (N=109, varianza de los 13 Sharpe tasa_0 limpios, muestra chica, orden de magnitud): Sharpe anual mínimo ~1,4 con holdout de 4 años, ~1,1 con 8 años, ~2,7 si la varianza usa los 99 intentos limpios. Transversal (0,56) y estacionalidad (0,18) no pasaban igual.
- Bug aparte, registrado como `QuantAgent-vbnh` (corrección, no criterio): sacar las filas sintéticas, impedir que corridas con snapshot sintético escriban en el `intentos.csv` real, y que `dsr_contra_intentos` ignore Sharpe no finitos o absurdos. Quitar filas toca el registro de intentos: requiere OK de Fede.

## D1. Rol del DSR en la etapa 3

- **A (recomendada, elegida en lo conceptual por Fede).** El DSR se muestra siempre y deja de ser corte. Pase a paper trading si, con todo definido antes de correr:
  - Sharpe de cartera > 0 y mayor al de la cartera de control, en etapa 2 y en etapa 3.
  - Sharpe positivo en la mayoría de los sub-períodos (por ejemplo, cada año del holdout).
  - Drawdown dentro del límite V30 vigente.
  - Un DSR bajo no bloquea, pero obliga a tamaño de posición reducido y a más semanas de paper antes de pasar a vivo (cifras a fijar por Fede).
- B. Mantener DSR ≥ 0,95 como corte. Casi nada pasa.
- C. Mantener el corte pero con intentos efectivos (ver D2).

Lógica: paper trading es la vara fresca y sin riesgo de plata; el filtro histórico descarta lo malo, no certifica lo bueno.

## D2. Intentos efectivos

N crudo cuenta como independientes intentos casi idénticos (12 activos × variantes de una regla, la misma cartera en etapas 1, 2 y 3, filas repetidas). Contar grupos de intentos parecidos (por correlación de curvas, agrupado de forma automática y no a mano) da un N efectivo menor y una vara más justa. Se informa junto al N crudo. Recomendación: informar ambos y usar el efectivo en el DSR mostrado.

## D3. Loop de mejora

Principio: aprender de los fallos sin que el holdout 2019–2022 influya en lo que se prueba.

1. **Ficha de postmortem por cartera, con causas estructurales solamente.** Registrar por qué cortó: pocos trades, costos que se comen la ganancia, rotación alta, una pierna que arrastra, drawdown, sensibilidad al régimen. Sin Sharpe del holdout en la ficha que lee el generador de ideas. Es el mismo criterio del pedido del investigador-1 (sin Sharpe ni trades).
2. **Las ideas nuevas salen de la literatura, no de ajustar lo que falló.** El postmortem dice qué no repetir; no elige parámetros ni activos.
3. **Cada propuesta es una entrada de catálogo con hipótesis económica, cita con DOI verificado, universo fijado antes de correr y 0 parámetros libres** (`catalogo/propuestas/`). Un paso de chequeo rechaza propuestas sin hipótesis o sin cita.
4. **Chequeo de originalidad:** una propuesta cuya curva en etapa 2 se parece demasiado a una ya probada no se corre (comparte grupo en D2), para no subir N en vano.
5. **Todo intento suma al registro, sin excepciones.**
6. **Datos de confirmación frescos:** 2023 en adelante reservado, mercados no usados (internacionales, futuros) y paper trading como vara final.
7. **Cuota de iteraciones por cartera** (por ejemplo, máximo 2 variantes derivadas de una misma idea) y revisión de Fede y el crítico antes de correr nada. El LLM propone, nada se ejecuta solo.
8. **Cadencia:** una retrospectiva por lote, no por cartera, con una tabla simple de causas de corte para ver patrones.

Riesgo residual: aunque el generador no vea el Sharpe, Fede y el crítico sí lo ven y deciden. Eso es aceptable si el pase final depende de paper trading y no de seguir ajustando contra 2019–2022.

## Pendiente de Fede (marcado como abierto)

- **Pendiente** · OK para limpiar `intentos.csv` (D bug).
- **Pendiente** · Números de D1: tamaño reducido y semanas de paper si el DSR es bajo.
- **Pendiente** · Si D2 se informa solo o también reemplaza al N crudo en el DSR.
- **Pendiente** · Cuota de variantes por idea (D3.7).
