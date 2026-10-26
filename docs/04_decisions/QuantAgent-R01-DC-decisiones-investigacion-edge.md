# QuantAgent-R01 — Decisiones de Fede sobre investigación de edge, generador y sizing

**Estado:** Aceptado (decidido por Fede el 2026-10-21/22, en conversación; este documento solo las registra).
**Origen:** `docs/02_planning/2026-10-20_investigacion_edge_y_generacion.md` §7 (PR #352), `QuantAgent-0gh7` (sizing por riesgo) y la consulta sobre retención fiscal de DBC/USO.
**Ticket:** R01 = `QuantAgent-e4jc` (padre del lote L20; hijos en `PLAN-CONTINUACION.md`, filas X01–X20).

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
- Fede envió un mail al soporte de Alpaca con las consultas; respuesta recibida el 2026-10-25 (ver abajo).

### Respuesta de Alpaca (2026-10-25)

Alpaca aclara que no da asesoramiento fiscal y comparte reglas generales para un no residente en EE.UU. con W-8BEN (Argentina):

- Reciben el formulario 1042-S por ingresos de fuente estadounidense. Los dividendos pagan normalmente 30 % de retención; un tratado podría bajarla, y Alpaca no tiene datos del tratado con Argentina.
- Las ventas de PTP (publicly traded partnerships) por extranjeros sufren una retención del 10 % sobre el **bruto** de la venta, y ciertas distribuciones de PTP también retienen. La venta de un PTP es la única venta de títulos que genera 1042-S. Los dividendos de al menos USD 0,50 en el año generan 1042-S.
- Las operaciones comunes (compra/venta de acciones y ETFs que no son PTP) no generan comprobante fiscal.
- Alpaca **no sabe** si DBC o USO son PTP, no conoce un proceso de "qualified notice" ni una lista oficial de símbolos PTP.
- Paper es simulado: sin retención ni reporte. Las reglas aplican al operar en vivo.

### Qué cambia (y qué no)

- **No bloquea paper.** Nada cambia en el motor ni en las estrategias en paper.
- **Se confirma el 30 % sobre dividendos** para el modelado antes de dinero real (el tratado con Argentina no ayuda según lo ya anotado arriba; Alpaca no aportó nada en contrario).
- **DBC y USO son PTP** (verificado en fuentes públicas, 2026-10-25): DBC es un fondo constituido como sociedad que reparte por K-1 y su 10-K lo trata como partnership; USO es un limited partnership (USCF lo declara así y emite K-1). PDBC se vende como "No K-1 ETF" (1099). Estas fuentes no son asesoramiento fiscal; el 10-K y la página de impuestos de cada emisor mandan.
- **Matiz que Alpaca no conocía:** los emisores publican "qualified notices". Invesco informó (aviso de diciembre 2025) que sus fondos, DBC incluido, cumplen la excepción y quedan exentos de la retención 1446(f), y USCF informó que USO cumple la "excepción del 10 %" porque no operó un negocio en EE.UU. Esa exención se renueva por año y el broker tiene que aplicarla; no se sabe si Alpaca la aplica. Hay que preguntárselo antes de operar en vivo.
- **Decisión vigente (sin cambios):** antes de dinero real, reemplazar DBC por PDBC y quitar USO, salvo que Alpaca confirme por escrito que aplica los qualified notices. Alpaca no dio esa confirmación, así que la regla no se relaja.
- Pregunta pendiente para Alpaca antes de dinero real: (a) si aplican los qualified notices de Invesco y USCF para evitar el 10 %; (b) si hay retención en las distribuciones de PTP más allá de los dividendos.

Fuentes: respuesta de soporte de Alpaca (mail de Fede, 2026-10-25); 10-K de Invesco DB Commodity Index Tracking Fund y de United States Oil Fund, LP en SEC EDGAR; aviso de qualified notices de Invesco (2025-12-09); nota de CIBC Mellon sobre 1446(f).

## Pendiente fuera de este documento

- `R:` de Fede sobre el lote #360 cuando termine L19.
- PR de docs #334 (dashboards de bonos de eldashboard): para más adelante, sin tickets hasta terminar el plan actual.

## Propuesta de actualización de README (no aplicada)

Agregar a `docs/04_decisions/README.md`, sección "Active Per-Change Decisions": `QuantAgent-R01-DC-decisiones-investigacion-edge.md` — Decisiones de investigación de edge, generador, sizing por riesgo y DBC/USO (Fede, 2026-10-22).
