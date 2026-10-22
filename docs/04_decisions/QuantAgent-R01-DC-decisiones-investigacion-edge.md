# QuantAgent-R01 — Decisiones de Fede sobre investigación de edge, generador y sizing

**Estado:** Aceptado (decidido por Fede el 2026-10-21/22, en conversación; este documento solo las registra).
**Origen:** `docs/02_planning/2026-10-20_investigacion_edge_y_generacion.md` §7 (PR #352), `QuantAgent-0gh7` (sizing por riesgo) y la consulta sobre retención fiscal de DBC/USO.
**Ticket:** R01 (ID de Beads del lote: pendiente, el PM lo completa).

## Marco que Fede dejó explícito

Avanzar ahora tiene como fin tener estrategias para probar paper trading. Paper es parte de la construcción y prueba del sistema, no una evaluación realista de las estrategias (con una evaluación estricta no pasarían). Ningún resultado de paper promueve una estrategia a dinero real ni cuenta como evidencia de edge.

## Decisiones

| # | Tema (pregunta de §7) | Decisión |
|---|---|---|
| 1 | Criterio de la etapa 3 (Q1) | **C0 + correcciones de medición.** Sin C1 (vía "diversificador") ni C2. C1 puede agregarse después, siempre declarando el rol antes de correr. |
| 2 | M1, Sharpe en exceso de la tasa corta (Q2) | Adoptado como columna del informe; el registro de intentos sigue a tasa 0. |
| 3 | M2, benchmark con los mismos pesos (Q3) | Adoptado como diagnóstico; 1/N sigue siendo el corte. M3 (error estándar de ΔSharpe) también se adopta. |
| 4 | Rol de paper (Q4) | **Paper liviano = prueba de sistema.** Chequeos: pesos iguales a la sombra salvo redondeo, ninguna orden rechazada sin explicar, dividendos bien tratados; si algo no coincide, se frena y se investiga. La escalera de incubación (§3.6) y sus números (tracking error, percentiles de corte, meses por etapa) se deciden **antes de poner dinero real**, no ahora. |
| 5 | Reserva 2023+ (Q5) | **Tope de 3** hipótesis, lista escrita antes: `cartera-transversal`, F1, F2. F1 y F2 entran solo si pasan la etapa 2. `cambio-de-mes` queda afuera (falló la etapa 1). Se evalúan con C0 + M1; el intento cuenta en N. Una sola apertura por hipótesis. |
| 6 | `cambio-de-mes` (Q6) | Sin etapa 3; fuera de la reserva. |
| 7 | F1 `rebalanceo-fin-de-mes` (Q7) | **Aprobada.** Etapas 2 y 3 se leen como réplica (muestra del paper hasta 2023-03); la decisión está en la reserva. Antes: verificar en el texto del paper la definición de la cuarta semana. |
| 8 | F2 `tendencia-larga-corta` (Q8) | **Aprobada**, medida con C0 (no hay vía de diversificador). Cuenta como variante de la tendencia (cupo máximo de 2 por idea por lote). Se habilitan **cortos en el motor y cuenta margin solo en paper**; la cuenta margin con dinero real se decide antes de operar en vivo. |
| 9 | F3 `carry-futuros` (Q9) | En espera hasta que una cartera con datos actuales llegue a paper. |
| 10 | `volatilidad-gestionada` (Q10) | **Rechazada.** Su razón de ser era la vía C1. Reabrible si algún día se adopta C1 declarando el rol antes de correr. No cuenta como intento (no se corre). |
| 11 | Lección escrita para el generador (Q11) | **Sí**: regla escrita por personas, sin cifras de la prueba, en el catálogo. |
| 12 | Generador: pedido, campos, chequeos de §5 (Q12) | **Sí**, con el rol `diversificador` fuera del pedido y de los campos. No correr `hipotesis proponer` hasta tenerlos implementados. |
| 13 | Beads (Q13) | Issue padre para el lote con R01–R15 como hijos; lo crea el PM. |
| 14 | Datos gratis (Q14) | **Filas 1, 2 y 3 de §2.5**: factores AQR/French (cortados en 2018-12, AQR no se versiona en el repo), ETFs ampliados con regla fechada en 2007, CSV de pysystemtrade. **Subastas del Tesoro (fila 4) rechazadas.** Un ticket por fuente, armado por el PM. Nada pago hasta que una familia pase la etapa 2 con datos gratis. |
| 15 | Modelo del generador (Q15) | **Sí**: modelo más capaz disponible con esfuerzo alto, refutador de otra familia, búsqueda web para verificar citas; el refutador corre en Codex por cuota. |

- Orden de ejecución de F1, F2 y la reserva: indistinto (Fede, 2026-10-22). Fede no tiene una base de ETFs comprados aparte, lo que confirma dejar C1 afuera.

## Sizing por riesgo (`QuantAgent-0gh7`)

- Fórmula: `cantidad (acciones) = (capital × riesgo_por_trade_pct) / |precio_entrada − precio_stop|`. El numerador está en USD y el divisor en USD por acción.
- **`TRADING_RIESGO_POR_TRADE_PCT` por defecto: 0,25 %** (no el 1 % del borrador).
- **`max_position_pct` se mantiene en 10 %.** Se descartó subirlo a 25–30 %: demasiada concentración (cuatro pérdidas seguidas sacan de juego).
- **Stop por defecto ±2 %** para las estrategias sin stop real (`sma_cross`, `halloween`, `momentum_12m`, `senal_externa`). Se aplica antes de dimensionar, en un solo lugar compartido, para que tamaño y monitor de posición usen el mismo stop; el informe lo marca como "stop por defecto". Ninguna estrategia queda fuera de la regla.
- El stop lo define la estrategia (en la señal); el risk manager solo valida el tope de posición.
- Detalles menores aprobados: la confianza se ignora en la regla `"riesgo"`; piso de 0,5 % para la distancia al stop; error explícito si el stop está del lado equivocado.

## DBC / USO (retención fiscal)

- **Paper sin cambios.**
- Antes de operar con dinero real: reemplazar DBC por PDBC (sin K-1) y quitar USO, o antes si Alpaca confirma que aplica la retención del 10 % (sección 1446(f)) sobre el bruto de ventas de PTP a cuentas extranjeras.
- Argentina no tiene tratado con EE.UU.: 30 % de retención sobre dividendos de todos los ETFs, no modelado en los backtests; modelarlo en el informe antes de dinero real.
- Fede envió un mail al soporte de Alpaca con las consultas; pendiente de respuesta.

## Pendiente fuera de este documento

- `R:` de Fede sobre el lote #360 cuando termine L19.
- PR de docs #334 (dashboards de bonos de eldashboard): para más adelante, sin tickets hasta terminar el plan actual.

## Propuesta de actualización de README (no aplicada)

Agregar a `docs/04_decisions/README.md`, sección "Active Per-Change Decisions": `QuantAgent-R01-DC-decisiones-investigacion-edge.md` — Decisiones de investigación de edge, generador, sizing por riesgo y DBC/USO (Fede, 2026-10-22).
