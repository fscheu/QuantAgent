# QuantAgent-j7bl · IM · Punto de control de L17 (loop investigador)

Corrió el PM en la VM el 2026-10-20 sobre el snapshot `etf-1d-2026-10-amplio` y la rama `lote/loop-investigador`. No se abrió 2019–2022 y no se corrió nada del embudo: solo la etapa 2 (2007–2018) de las tres carteras de L16, para guardar las celdas que la ficha necesita.

## Qué se hizo

1. **Celdas guardadas.** `validar_etapa2` ahora deja `celdas.csv` por corrida (I09) con trades, ganancia y caída máxima por pierna y activo, más la caída máxima de la cartera (I10). Se volvió a correr la etapa 2 de `cartera-tendencia-momentum`, `cartera-transversal` y `cartera-estacionalidad` (dos veces: la primera mostró el error de I10, abajo). Copia previa del registro: `local-data/intentos-20261020-previo-a-i08.csv`.
2. **Fichas de postmortem** (`catalogo/postmortem/<id>.yaml`): las tres salen **sin causa estructural** (`causas: []`). No cortaron por pocos trades, costos, rotación, una pierna dominante ni caída máxima de la cartera (caída máxima de la cartera en la etapa 2: 0,06 / 0,12 / 0,30 contra el límite de 0,35). Lo que las cortó fue rendir poco contra el benchmark y contra el DSR, que la ficha no mira a propósito. Es un resultado, no una falla: el postmortem estructural descarta causas, no inventa una.
3. **Retrospectiva de L16** (`retrospectiva-L16.md`): ninguna causa en las tres carteras; 20 intentos registrados de esas carteras.
4. **Pedido al investigador** (`armar_pedido`): 4.637 caracteres, 111 intentos registrados, catálogo y fichas; sin Sharpe, DSR ni ganancias (se buscó por texto). El LLM fue `claude-opus-5-5-medium` vía `agy`, con un script de prueba como `QUANTAGENT_LLM_CMD`.
5. **Tres propuestas** en `catalogo/propuestas/`, las tres aceptadas por el chequeo de forma y de originalidad. Nada pasó a `hipotesis/`.

## Propuestas y verificación de citas (Crossref)

| Propuesta | Idea | Citas | Verificación del PM |
|---|---|---|---|
| `cambio-de-mes` | Retorno concentrado entre el último día del mes y los 3 primeros del siguiente, en 8 ETFs de acciones | 10.1093/rfs/1.4.403 · 10.2469/faj.v64.n2.11 | Existen: Lakonishok y Smidt 1988; McConnell y Xu 2008. Coinciden con la idea. |
| `volatilidad-gestionada` | Escalar la exposición a SPY por la inversa de la varianza del mes anterior | 10.1111/jofi.12513 | Existe: Moreira y Muir 2017. Coincide. |
| `deriva-pre-fomc` | Retorno de SPY concentrado en los días de anuncio programado de la FOMC | 10.1111/jofi.12196 · 10.1017/S0022109013000410 | La primera existe (Lucca y Moench 2015). **La segunda es un artículo distinto** (Bali et al., asimetría neutral al riesgo en opciones): cita equivocada. |

Dos problemas prácticos antes de aprobar nada:
- Las tres clases de señal (`turn-of-the-month`, `vol-managed`, `pre-fomc`) **no existen todavía en el motor**; aprobarlas lleva un ticket de implementación cada una.
- `deriva-pre-fomc` necesita un calendario de fechas de la FOMC que el snapshot no tiene.

Recomendación del PM: aprobar `cambio-de-mes` (datos ya disponibles, cita verificada), dejar `volatilidad-gestionada` para después y rechazar `deriva-pre-fomc` por la cita equivocada y el calendario faltante. Es decisión de Fede (`hipotesis propuestas aprobar|rechazar <id> --quien <nombre> [--motivo ...]`).

## Encontrado en el camino

- `ficha_postmortem` de I01 marcaba `drawdown` si cualquier ETF suelto pasaba 0,35 (18 de 58, 9 de 46 y 8 de 17 celdas): las tres carteras salían igual, sin información. Se corrigió en I10 (#345) usando la caída máxima de la cartera; no se tocó ningún umbral.
- I01 no tenía quién le diera las filas por celda: se agregó I09 (#344) con OK de Fede.
- Las dos corridas de la etapa 2 suman filas al registro: 111 filas, 96 distintas (antes 99 y 95). El N efectivo casi no se mueve.
- `agy` rechaza los tickets heredados con la etiqueta `fuera-loop` (d4cq.42 y .43); se la saqué.

## Qué no se hizo

- Aprobar o correr ninguna propuesta. Calendario FOMC. Clases de señal nuevas.
- Verificar con un humano que las citas respaldan lo que dice cada propuesta; solo se comprobó que el DOI existe y a qué artículo apunta.
