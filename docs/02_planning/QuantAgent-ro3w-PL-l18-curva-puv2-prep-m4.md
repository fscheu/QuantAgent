# QuantAgent-ro3w: plan del lote L18 (curva contra el criterio de paper, `cambio-de-mes` y preparación de M4)

Fede aprobó los tres frentes el 2026-10-20. Rama `lote/m4-prep-curva-puv2`; filas Q01–Q12 en `PLAN-CONTINUACION.md` §4.

## Por qué este lote es mixto

Ninguna cartera pasa hoy el criterio de paper de la etapa 3. Las tres de L16 cortaron en `QuantAgent-n96l` porque su Sharpe de 2019–2022 no supera al de comprar y mantener los mismos activos (0,25, 0,56 y 0,18 contra 0,61, 0,61 y 0,57). La capa curva, aceptada en L14, no se evaluó con ese criterio. M4 (paper con Alpaca) necesita una cartera de entrada, así que primero se mide esa.

## Frente 1: la cartera núcleo + curva contra el criterio de paper (Q01)

Criterio (DC `2026-10-19-DC-criterio-etapa3-y-loop-de-mejora`): Sharpe de cartera mayor a 0 y mayor al benchmark en la etapa 2 y en la 3, Sharpe positivo en al menos 3 de 4 años del holdout, drawdown de 0,35 o menos. El benchmark es la misma lista de activos con pesos iguales y rebalanceo mensual. El DSR se informa y no corta.

Lo que ya se sabe de L14: Sharpe 0,38 en 2019–2022 y 3 de 4 años positivos. Falta el benchmark de esos activos y el drawdown. Con las otras carteras el benchmark de 2019–2022 estuvo en 0,57–0,61, y 0,38 no lo superaría; hay que medirlo con los activos de esta cartera antes de afirmarlo.

Alcance: el ticket no corre el motor, no abre ni escribe ningún candado y no escribe en `intentos.csv`. Lee las curvas que la corrida única de L14 ya generó (copiadas a `local-data/curvas-l14-curva/{etapa2,etapa3}`) y los cierres del snapshot de los mismos activos en la misma ventana ya abierta para esta cartera. Si el cálculo exigiera cualquier otra lectura de 2019–2022, el PM frena y escala a Fede.

Si pasa, es la cartera de entrada a M4 con su recargo de paper (50 % del tamaño y 12 semanas si el DSR es menor a 0,50). Si no pasa, el IM dice por cuánto y M4 queda sin cartera hasta que otra hipótesis cruce el criterio.

## Frente 2: `QuantAgent-puv2`, turn-of-the-month (Q02–Q04)

Partido en tres entregas de 100 líneas o menos: la señal (Q02), el YAML que carga y vuelve a `hipotesis/` (Q03) y la corrida hasta la etapa 2 sobre 2007–2018 (Q04, en la VM). La etapa 3 sólo con el OK escrito de Fede por esta hipótesis. Al cerrar Q04 se cierra `QuantAgent-puv2`.

## Frente 3: preparación de M4 que no depende de una cartera (Q05–Q12)

Datos de Alpaca (documento `QuantAgent-7nle-DC-datos-para-operar-m3.md` §6): T1, T2, T8 y T9 ya están cerrados. Quedan T3 (Q05), T6 (Q06), T7 (Q07), T4 (Q08) y T5 (Q09). T5 toca las dos consultas del backtest y es "decidir": cualquier cambio en una referencia de M1 se escala.

Diagnósticos de paper `say` (Q10), `oc0h` (Q11) y `beo` (Q12). Los tres son de primera entrega sin tocar `quantagent/` y terminan en una decisión de Fede; el lote sigue con las filas independientes mientras esperan.

Las claves de Alpaca paper y de FRED ya están en `~/secrets` (`alpaca.env`, `fred.env`).

## Qué queda afuera y por qué

- `QuantAgent-0gh7` (sizing por riesgo): tiene decisiones abiertas de Fede (comportamiento sin stop real y `max_position_pct`), y lo bloquean `say`, `oc0h` y `beo`.
- Broker (`13a`, `yme`, `qr6`, `y62`, `b41`): `13a` diseña la interfaz `Broker` y su respuesta depende de lo que Fede decida sobre reversiones y regla de salida (Q10–Q12); `yme`, `y62` y `b41` cuelgan de `13a`. Van en L19.
- `8659` y `55x`: diagnósticos sin cartera de por medio, pero `8659` pide medir Alpaca paper ante una orden que excede el poder de compra y conviene hacerlo junto con `13a`. Van en L19.

## Cómo corre

Un ticket por vez con `hermes_launch.sh --run --lote lote/m4-prep-curva-puv2`. Antes de cada ticket el PM mira `agy -p "/usage"`, anota los pools en `~/.local/state/quantagent-loop/pm-estado.md` y elige modelo por dificultad (Opus, Sonnet, Gemini 4 Argon, Gemini 3.8 Flash; con menos de 15 % de cupo un pool queda afuera). Los tickets que escriben en `local-data` (Q01, Q04, Q09) van a Codex. El PM no implementa. Las entregas de sólo documentos no tienen CI y pasan por Fede para el merge a la rama del lote. Todo corre en la VM porque `intentos.csv` (el registro que alimenta N en el DSR) sólo existe ahí.

Cierre: PR del lote a `main` marcado listo y pedido de `R: lote-m4-prep-curva-puv2 vi: ... decido: merge`.
