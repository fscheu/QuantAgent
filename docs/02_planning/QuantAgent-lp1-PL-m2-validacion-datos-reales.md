# M2 — Validación con datos reales (plan posterior a M1)

Ticket: `QuantAgent-lp1` (T25 de `PLAN-CONTINUACION.md`). Estado: **borrador**, escrito el 2026-10-06.
Se cierra después de T23 (ADR del engine): si el engine cambia, las filas de los lotes A a C se reescriben.
Los tickets de BEADS se crean cuando Fede apruebe este documento; hasta entonces las filas usan `V01`…`V41`.

---

## 1. Por qué esta etapa

M1 deja un backtester que calcula bien, probado sobre datos sintéticos (`tests/fixtures/spy-90d.csv` tiene
velas 24/7 y volumen fijo) y sin ninguna estrategia que gane plata (RSI sobre ese fixture: profit factor 0.61).
Conectar un broker en ese estado solo prueba la plomería.

Decisiones de Fede del 2026-10-06:

1. Antes del broker va una etapa de validación: datos históricos reales, costos realistas, comparación de
   N estrategias y prueba fuera de muestra.
2. La estrategia de 4 agentes LLM es una estrategia más. No se le reserva presupuesto ni trabajo propio.
3. La IA generativa entra en otras instancias: informe de corrida, revisor crítico, nota macro semanal y
   generador de hipótesis.
4. Se valida sobre ETFs en velas diarias **y** sobre un par de activos en 4h.
5. Numeración de milestones nueva (§2).

## 2. Milestones

| Milestone | Nombre | Criterio de completitud |
|---|---|---|
| M1 | Backtesting estable | 3 estrategias, resultados reproducibles, métricas auditadas, ADR del engine (T23) |
| M2 | Validación con datos reales | §3 de este documento |
| M3 | Broker Alpaca en paper | Integración funcionando contra la cuenta paper (`13a`, `yme`, `qr6`, `y62`) |
| M4 | Paper trading estable | 2 semanas sin intervención manual (`b41`) |
| M5 | Live con capital mínimo | Primera semana con dinero real, sin errores críticos |

"Arquitectura multi-estrategia", el M2 anterior de `milestone-tracker.md`, queda absorbido por el lote C.

## 3. Criterio de salida de M2

M2 termina cuando existen las dos cosas:

1. **La herramienta:** un comando compara estrategias sobre datos reales con costos, contra comprar y mantener,
   y emite un veredicto dentro y fuera de muestra con su informe.
2. **El veredicto:** al menos una estrategia pasa los umbrales sobre la reserva final, o Fede decide por
   escrito pasar a M3 sin estrategia validada (o cortar el proyecto en esa capa).

Los umbrales se deciden en V21. Propuesta inicial: después de costos, sobre el período de prueba, Sharpe mayor
que el de comprar y mantener y drawdown máximo no mayor. Los umbrales de 2025 (win rate ≥ 40%, Sharpe ≥ 1.0,
drawdown ≤ 15%) no comparaban contra una referencia.

## 4. Reglas de la etapa

- **Modo de trabajo:** el de `PLAN-CONTINUACION.md` §3 (entregas de ~100 líneas, revisión del PM, un PR de lote
  por punto de control).
- **Reserva final:** un tramo de la historia que ninguna corrida de ajuste ni de comparación puede leer. Se abre
  una sola vez, en V41. El CLI lo hace cumplir (V22).
- **Intentos contados:** cada combinación estrategia × parámetros probada queda registrada. Un resultado bueno
  entre cien intentos no vale lo mismo que entre tres.
- **La IA no calcula:** los números salen de código determinista. Los modelos leen artefactos (CSV, JSON) y
  escriben texto; un verificador comprueba que cada número del texto existe en el artefacto.
- **Sin clave también funciona:** `backtest run`, `compare` y el veredicto corren sin API key. El informe y la
  crítica son un paso aparte.
- **La IA no ve la reserva:** ni el generador de hipótesis ni el revisor crítico reciben datos de ese tramo.

## 5. Lotes y cola

Tipos de revisión: **leer**, **decidir**, **probar** (igual que en `PLAN-CONTINUACION.md` §4).

| Lote | Rama | Filas | Punto de control | Decisiones de Fede |
|---|---|---|---|---|
| A | `lote/datos-reales-diarios` | V01–V07 | Las 3 estrategias corren sobre un snapshot versionado de ETFs diarios reales, con reporte de calidad | V01 |
| B | `lote/costos-y-referencia` | V08–V13 | Cada corrida informa costos y se compara contra comprar y mantener | V08, V13 |
| C | `lote/comparacion` | V14–V20 | Un comando corre la matriz estrategia × activo; agregar una estrategia no toca el engine | V19 |
| D | `lote/fuera-de-muestra` | V21–V26 | Veredicto PASA / NO PASA por estrategia, dentro y fuera de muestra, con la reserva bajo candado | V21 |
| E | `lote/informe-y-critico` | V27–V31 | Cada comparación trae un informe en lenguaje llano y una crítica con evidencia | V27 |
| F | `lote/intradia-4h` | V32–V35 | La misma comparación y veredicto sobre 2 activos en 4h | V32 |
| G | `lote/generador-de-hipotesis` | V36–V41 | Un modelo propone estrategias, se implementan como código determinista y pasan solas por el veredicto | V36, V41 |

Corte de control después del lote D: con la herramienta completa y los primeros veredictos, Fede decide si E, F
y G siguen en este orden. G depende de D: sin prueba fuera de muestra no hay forma de distinguir una idea buena
de un sobreajuste.

| # | Objetivo | Aceptación binaria | Revisión | Lote |
|---:|---|---|---|---|
| V01 | Decisión: lista de ETFs, rango de fechas y dónde vive el snapshot | Doc ≤40 líneas con opciones y tamaño en disco | decidir | A |
| V02 | `data snapshot`: baja diarios y escribe snapshot con manifiesto (símbolo, rango, filas, sha256) | Releer el snapshot da el mismo hash; con `--offline` no hay llamadas de red | probar | A |
| V03 | Reporte de calidad: huecos contra el calendario de mercado, velas inválidas, saltos anómalos | Test: a un snapshot se le borra un día y el reporte lo nombra | leer | A |
| V04 | Precios ajustados por splits y dividendos, documentado y testeado | Test con un split conocido: no hay salto de precio en la fecha | leer | A |
| V05 | `backtest run --snapshot <nombre> --symbol <s> --timeframe 1d` | Corre RSI sobre SPY diario real, exit 0 | probar | A |
| V06 | Las 3 estrategias de M1 sobre SPY diario real | Cada una con trades > 0 y `backtest verify` en `OK reproducible` | probar | A |
| V07 | Test golden sobre el snapshot real | Falla si cambia un trade o una métrica | leer | A |
| V08 | Decisión: comisión y slippage por clase de activo | Doc con 2–3 perfiles y el efecto de cada uno sobre una corrida | decidir | B |
| V09 | Perfil de costos por clase de activo; el CLI imprime el perfil usado | Cambiar el perfil cambia el PnL en el monto que predice el recálculo | probar | B |
| V10 | Referencia comprar y mantener: misma ventana, mismos costos | `recalc_metrics.py` reproduce la referencia sin importar `quantagent` | probar | B |
| V11 | Métricas nuevas: retorno anualizado, exceso sobre la referencia, % del tiempo en mercado | Recálculo independiente = CLI | probar | B |
| V12 | Diagnóstico: cómo afectan el tamaño de posición y el límite diario de pérdida en velas diarias | Doc con una corrida con y sin límite | decidir | B |
| V13 | Aplicar lo decidido en V12 | Según la decisión | probar | B |
| V14 | `backtest compare --strategies … --symbols … --snapshot …` | Tabla en stdout y CSV, una fila por estrategia × activo, con la referencia | probar | C |
| V15 | Agregar una estrategia = un archivo + una línea de registro | Test que registra una estrategia nueva sin tocar `quantagent/backtesting/` | leer | C |
| V16 | Parámetros de estrategia desde un archivo | La misma estrategia con dos archivos da dos filas distintas en `compare` | leer | C |
| V17 | Estrategia clásica diaria 1: cruce de medias 50/200 | Trades > 0 sobre SPY diario; test de señal a mano | probar | C |
| V18 | Estrategia clásica diaria 2: momentum absoluto de 12 meses | Ídem | probar | C |
| V19 | Diagnóstico: ¿el engine simula una cartera con capital compartido entre activos? | Doc con una corrida de 2 activos y la respuesta | decidir | C |
| V20 | La comparación se guarda como reporte en markdown | El archivo tiene la tabla y los parámetros de la corrida | leer | C |
| V21 | Decisión: partición (ajuste / prueba / reserva final) y umbrales de salida | Doc con fechas concretas y umbrales | decidir | D |
| V22 | `--from` / `--to` y candado de la reserva | Sin el flag explícito el CLI se niega a leer la reserva; cada acceso queda registrado | probar | D |
| V23 | Barrido de parámetros con registro de intentos | El registro tiene una fila por combinación probada | probar | D |
| V24 | Walk-forward: parámetros elegidos en cada ventana de ajuste, aplicados en la de prueba | Test con serie conocida; ninguna ventana de prueba se usa para elegir | probar | D |
| V25 | Veredicto por estrategia: tabla dentro / fuera de muestra y PASA / NO PASA | La salida coincide con los umbrales de V21 aplicados a mano | probar | D |
| V26 | Estadísticas de robustez: cantidad de trades, concentración del PnL, resultado por año, sensibilidad a ±20% de parámetros | Test con una serie donde un solo trade explica todo el PnL | leer | D |
| V27 | Decisión: proveedor, modelo y tope de gasto por informe | Doc con costo medido de un informe | decidir | E |
| V28 | `report explain`: informe en lenguaje llano desde los artefactos | El verificador confirma que cada número del informe existe en el artefacto | probar | E |
| V29 | Glosario: cada término técnico se explica la primera vez que aparece | Informe de ejemplo revisado por Fede | leer | E |
| V30 | `report critique`: revisor crítico con las estadísticas de V26 | Una estrategia sobreajustada a propósito recibe la objeción correcta | probar | E |
| V31 | `compare` y el veredicto adjuntan informe y crítica; sin API key siguen funcionando | `env -u` de las claves: exit 0 sin informe | probar | E |
| V32 | Decisión: los 2 activos en 4h y la fuente de datos | Doc: Yahoo (730 días) contra una fuente alternativa | decidir | F |
| V33 | Snapshot intradía con sesiones de mercado y reporte de calidad | Sin velas fuera de horario para activos con horario | probar | F |
| V34 | Anualización del Sharpe verificada con datos reales con horario | Recálculo independiente = CLI | leer | F |
| V35 | `compare` y veredicto sobre el snapshot intradía | Tabla y veredicto para los 2 activos | probar | F |
| V36 | Decisión: formato de una hipótesis (idea, fuente, regla, parámetros, qué la refutaría) | Doc con 2 ejemplos escritos a mano | decidir | G |
| V37 | `research propose`: el modelo genera hipótesis en ese formato, sin código | Las hipótesis validan contra el esquema | leer | G |
| V38 | `research implement`: hipótesis → estrategia determinista con tests | La estrategia se registra y corre en `compare` | probar | G |
| V39 | Cada candidata pasa sola por ajuste y prueba; suma al registro de intentos | Ninguna corrida del generador toca la reserva | probar | G |
| V40 | Tabla de todas las candidatas probadas, incluidas las descartadas | La cantidad de filas = intentos registrados | leer | G |
| V41 | Cierre de M2: las que pasaron se evalúan una vez sobre la reserva final | Informe final y decisión de Fede | decidir | G |

**Fuera del loop**

- `QuantAgent-wwi`, nota macro semanal (capa L1): arranca en paralelo. Instalar el skill en `~/.hermes` requiere
  confirmación de Fede y las 4 decisiones de `2026-09-17_borrador_skill_revision_macro_semanal.md` §2.
- Siguen fuera: `u0w`, la UI de Streamlit, `kkj.10`, `kkj.11`, datos en tiempo real y todo M3.

## 6. Qué depende de M1

| De M1 | Afecta |
|---|---|
| T23, ADR del engine | Lotes A a C: comandos y archivos cambian si se adopta `backtesting.py` |
| T13 y T14, anualización del Sharpe | V11 y V34 |
| T08c, slippage por defecto | V08 y V09 parten de ese valor |
| T15 y T20, `verify` y test golden | V06 y V07 los reutilizan |

## 7. Preguntas abiertas

- V19: el engine recorre un activo y después el siguiente (`quantagent/backtesting/backtest.py`). Si no simula
  capital compartido, las estrategias de rotación entre activos quedan fuera de M2 o piden un ticket propio.
- Con 5 estrategias deterministas puede no haber ninguna que pase. Ese resultado es válido y está previsto en
  el criterio de salida.
