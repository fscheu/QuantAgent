# Reconciliación de PLAN-CONTINUACION y Beads

**Ticket:** `QuantAgent-pjwb` · **Corte:** 2026-10-23 · **Estado:** inventario sincronizado para revisión en PR.

Este documento acompaña a [PLAN-CONTINUACION.md](../../PLAN-CONTINUACION.md). La cola de L20/L21 y su aceptación están en ese plan y en el plan específico de L21. Beads conserva el estado de ejecución; este inventario registra a dónde pertenece cada issue todavía no cerrado y evita que tickets etiquetados históricamente entren a la cola activa por accidente.

## Reglas de reconciliación

- L20 y L21 se originan en `origin/main`; L20 corresponde al PR #416 y L21 al PR #412. No se fusionan aquí sus ramas de código.
- Cuando un ticket aparece cerrado en el PR de un lote, se sincroniza el estado en Beads. Un issue padre permanece abierto mientras tenga decisión pendiente o dependencia bloqueante real.
- Un ticket sin fila en L20/L21 queda explícitamente fuera de la cola, con la condición para retomarlo. Prioridad y etiquetas históricas no implican que esté aprobado para esos lotes.
- La puesta en marcha de dos semanas en Alpaca paper es una actividad operativa posterior al merge, fuera de los tickets de código y con autorización de Fede.

## Tickets cerrados al sincronizar evidencia

| Ticket | Evidencia |
|---|---|
| `QuantAgent-vbnh` | Registro limpio; 112 filas, cero snapshots sintéticos al 2026-10-23; guardas en `tests/conftest.py` y `verdict.py`; artefacto `docs/06_implementation/QuantAgent-vbnh-IM-limpieza-intentos.md`. |
| `QuantAgent-e4jc.1`, `.7`, `.8.1`, `.8.2`, `.8.3`, `.9` | Entregas cerradas en rama L20; PR #439 integró el último fragmento D-2 el 2026-10-23. D-2 padre `.8` también queda cerrado. |
| `QuantAgent-e4jc.13.3` | PR #462 mergeado a `lote/paper-alpaca` el 2026-10-23; cierre de sub-ticket según el flujo de loop. |
| `QuantAgent-e4jc.13` | Particiones `.13.1`–`.13.3` cerradas; Beads conserva el padre como cerrado. |
| `QuantAgent-cq4u`, `lwji`, `oc0h`, `v2ll`, `whq0.3` | Cierres sincronizados con las razones ya registradas en Beads (PR #423; #362/#389/#391; decisión R de L18 con pruebas #382; #408; revisión PM de L19). |

`QuantAgent-0gh7` sigue abierto porque Beads tiene una dependencia `blocks` hacia `QuantAgent-beo`, que permanece como decisión de Fede. Sus hijos `.1`–`.5` están cerrados. No se fuerza el cierre del padre en esta reconciliación.

## Inventario de issues no cerrados

El estado debajo es el estado de Beads al momento del corte. Para la cola activa, la tabla del plan define el orden; los estados cambian durante las entregas.

| Issue | Estado Beads | Título | Ubicación o disposición |
|---|---|---|---|
| `QuantAgent-0gh7` | open | Sizing por riesgo por trade usando el stop_loss de la señal | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-4148` | open | Auto-merge por GitHub Action de PRs de tickets hacia ramas lote/* | Automatización implementada; verificar dry-run requerido antes de cerrar. |
| `QuantAgent-55x` | open | Engine: el reloj por velas del dato no rige en corridas online ni en el replay del Streamlit | L21 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-577` | open | Capa macro: script que trae los indicadores del tablero semanal desde fuentes directas | Script macro; futuro después de paper, sin lote aprobado. |
| `QuantAgent-5z0` | open | Routine nocturna del loop AI: crear, correr sobre el primer ticket y ajustar el prompt | Rutina nocturna histórica; reemplazada por loop vigente, confirmar cierre como supersedida. |
| `QuantAgent-6ie` | open | Investigación: OpenBB como capa de datos, análisis macro y espacio compartido humano-agente con gráficos | Investigación OpenBB; decisión/recomendación, fuera del alcance actual. |
| `QuantAgent-7ic9` | open | Loop: evitar falso aviso sin entrega y registrar fallos de consulta del PR | Mantenimiento del loop; fuera de L20/L21, planificar después del lote activo. |
| `QuantAgent-9vn` | open | Loop por lotes: el PM revisa cada entrega e integra en lote/<nombre>; Fede mergea un PR por lote | Proceso por lotes operativo; verificar criterio de aceptación y reconciliar cierre. |
| `QuantAgent-av1p` | open | L21: paper trading con Alpaca (M4), lote lote/paper-alpaca | L21 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-av1p.4` | open | L21: conectar el rebalanceador al ciclo paper de las carteras | L21 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-b2zg` | open | Radar de papers: barrido Crossref y filtro LLM de candidatos al generador | Radar de papers; explícitamente después de paper y cuando haga falta más volumen. |
| `QuantAgent-b41` | open | Digest operativo diario en texto plano para el piloto de paper trading | L21 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-beo` | open | Riesgo: la regla de salida está duplicada entre backtest (should_exit) y paper (scheduler) | L18; diagnóstico entregado, decisión de Fede pendiente. |
| `QuantAgent-col` | open | Revisar: velas intradía quedan en hora de Nueva York sin zona y el calendario de mercado las interpreta como UTC | Fuera de cola: comprobar que `gjmf` cubra el caso Yahoo/SPY 4h y cerrar o crear fix si falla. |
| `QuantAgent-d4cq` | open | Plan M3: fábrica de estrategias (embudo para evaluar cientos de hipótesis) y decisión de motor | Epic M3; umbrella del trabajo de fábrica, no ejecutar como ticket. |
| `QuantAgent-d4cq.76.2` | open | Loop: reclamo por ticket con CLAIM y RELEASE (C11.2) | Loop-wip L12; bloqueado por partición previa, retener fuera de la cola activa. |
| `QuantAgent-dhlu` | open | Loop: Codex como agente implementador (LOOP_AGENT=codex) | Selector Codex; conservar abierto hasta validar entrega completa/criterios. |
| `QuantAgent-e4jc` | open | R01: investigación de edge, generador, sizing y paper liviano (lote L20) | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.10` | open | G-1: lección escrita para el generador en el catálogo | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.11` | open | G-2: refutador de otra familia, modelo del generador y búsqueda web | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.12` | open | G-3 (opcional): cupo por tema y marca 'sin prueba ciega en 2019–2022' | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.14` | open | P-2 (R13): protocolo de la reserva 2023+ | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.2` | open | F1-b (R09): pesos objetivo por fecha, clase rebalanceo-fin-de-mes y llenado al cierre en el motor | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.3` | open | F1-c (R10): F1 por las etapas 0 a 2 (2007–2018) como réplica | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.4` | open | F2-a (R11): celdas cortas en la cartera y en el control contra backtesting.py | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.5` | open | F2-b: cortos y cuenta margin en paper | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-e4jc.6` | open | F2-c (R12): F2 por las etapas 0 a 2 | L20 (fila en PLAN-CONTINUACION.md) |
| `QuantAgent-fbiv` | open | Tests: compare_control deja el tamaño de posición global en 100 % | Bug de aislamiento; fuera de lote, candidato de mantenimiento corto. |
| `QuantAgent-hak` | open | Loop AI: PR template, convención de labels y docs/workflows/ai-dev-loop.md | Artefactos fase 0; verificar criterio restante y reconciliar cierre. |
| `QuantAgent-hyt` | open | Reevaluar y reactivar la validación funcional post-deploy de QA | QA post-deploy; fuera de lote, requiere evaluación/manual QA. |
| `QuantAgent-iuf` | open | Decisión de datos para M2: ETFs diarios, 2 activos en 4h, rango, fuente y ubicación del snapshot | Decisión de datos M2 documentada como propuesta; confirmar estado/decisión, sin implementación activa. |
| `QuantAgent-ke38` | open | Mapa de lotes: GitHub Action diaria + Pages | Action/Pages implementados en #449; verificar aceptación restante y cerrar. |
| `QuantAgent-kkj.10` | open | Persistir y seedear catálogo base de configuración para QA/DEV limpio | Configuración UI/seed QA; fuera de L20/L21, diferido. |
| `QuantAgent-kkj.11` | in_progress | Configurar routing multi-provider por rol para estrategias costo-eficientes | Routing multi-provider; fuera de L20/L21, en progreso independiente. |
| `QuantAgent-p54` | open | Spike: correr una entrega del loop en Claude Code en la nube | Spike Claude cloud; experimento fuera de alcance del loop actual. |
| `QuantAgent-pjwb` | in_progress | Reconciliar plan vigente y Beads entre main y lotes L20/L21 | Esta reconciliación; cerrar al integrar el PR a main. |
| `QuantAgent-u0w` | open | Implementar APScheduler para ejecucion background de backtests | APScheduler para UI/backtests; fuera del alcance M4 paper. |
| `QuantAgent-wwi` | open | Skill Hermes 'revisión macro semanal': instalar borrador, primera corrida, cron dominical | Skill macro Hermes; tarea externa de escritorio, fuera del loop. |
| `QuantAgent-y89w` | open | Adaptador: cierre final tras entrada larga rechazada puede abrir un corto | Descartado del loop; conservar diagnóstico abierto, sin cola activa. |
| `QuantAgent-yxl1` | open | Decisión: capital por estrategia según su eficacia medida (regla de tamaño por eficacia) | Decisión de sizing por eficacia; esperar cartera finalista y evidencia. |
| `QuantAgent-zui` | open | CI: correr lint+tests en pull_request hacia main (sin deploy) | CI fase 0; verificar criterio restante y reconciliar cierre. |

## Próxima actualización

Al integrar este PR en `main`, cerrar `QuantAgent-pjwb`. Si cambian estados de los PR #416 o #412 antes del merge, actualizar la fila afectada y el inventario Beads del PR. No cerrar los issues de validación manual hasta que exista la evidencia indicada en sus criterios.
