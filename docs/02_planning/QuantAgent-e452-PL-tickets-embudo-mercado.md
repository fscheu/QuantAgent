# QuantAgent-e452 · Revisión del diseño v0.4.2 y tickets de Embudo, Ficha de hipótesis y Mercado

- **Fecha:** 2026-11-04. **Estado:** plan aprobado por Fede el 2026-11-04 (decisiones D1–D5 aceptadas, ver §4); PR #806 ya está en `main`. Los tickets se cargan en Beads desde la VM (ver §5).
- **Fuente:** paquete del designer `quantagent-web-design-v0.4.2`, ya copiado a `docs/03_design/web-design-system/` (archivo nuevo `EMBUDO-MERCADO.md`; tocados `COMPONENTES`, `PANTALLAS`, `DECISIONES`, `EVENTOS`, `RENDIMIENTO`, `README`, `api-types.ts`, `config.ts`, `help-texts.es.json`, `tokens.css`).
- **Epic:** `QuantAgent-e452`. Reemplaza el alcance de WEB-14 (`e452.14`) y FE-08 (`e452.22`), que pasan a ser paraguas de los tickets de abajo.
- **Lote sugerido:** `lote/m-embudo-mercado`. IDs provisorios `EM-xx` hasta que Beads asigne los reales.

## 1. Revisión del diseño

El paquete es coherente con lo que Fede confirmó el 04/11 (ciclo `YYYY-MM`, 11 sectores del motor M5, "En cartera" en sectores y ETFs, evento `review_rule_reached`, regla de dato macro viejo, comandos `quantagent registro …`, universo real de 29 ETFs). No traspasé ninguna decisión suya al designer. Hallazgos:

**Menores (documentación del paquete)**
1. `README.md`, bullet de v0.4.1: dice "ranking sobre los 9 sectores"; lo corrige el bullet de v0.4.2 justo arriba. Queda como historial, no hay que tocarlo.
2. `EMBUDO-MERCADO.md` §4: los prerrequisitos 7 y 9 están desordenados (1–6, 8, 10, 11, 9, 7). Cosmético.
3. `tokens.css` solo suma clases utilitarias; la copia de `web/src/styles/tokens.css` hay que sincronizarla a mano (mismo patrón que FE-01b).

**Brechas entre el contrato nuevo y lo que existe (esto es lo que cambia el esfuerzo)**

| # | Brecha | Evidencia | Consecuencia |
|---|---|---|---|
| G1 | `lote/m-ciclos` (PR #806, 68 commits) no está en `main`; sin él no existen `registro_db`, `registro_vista` ni las tablas `hypotheses`, `hypothesis_events`, `hypothesis_attempts`, `cycles`. | `git ls-tree origin/main quantagent/fabrica/` | Todo el backend de Embudo espera ese merge. Mercado y el frontend no. |
| G2 | La API web lee `~/web-ops/maestro.db` y las bases paper; el registro vive en la base de cada entorno (hoy el Postgres de investigación, `DATABASE_URL`) y se carga con `quantagent registro importar`. | memoria del proyecto, `apps/api` | Hay que decidir dónde lee la API el registro (ver decisión D1). |
| G3 | `Measure` (valor contra umbral, numérico) no existe: el registro guarda `reason` como texto libre. `causas_de_motivo` solo devuelve causas. | `registro_vista.py`, `HypothesisEvent.reason` | Hay que escribir la medida estructurada en `data` al registrar cada etapa (toca `fabrica`). Los eventos históricos quedan con `measure: null`, caso ya diseñado. |
| G4 | La tabla `hypotheses` solo tiene `id` y `created_at`: no hay `name` ni `idea`. | `models.py` | `name = id`; `idea` sale de la justificación del YAML en `hipotesis/` (o columnas nuevas). Hoy `idea` y `last_metric` salen `null` y el diseño lo tolera. |
| G5 | `embudo()`, `ciclos()` y `hipotesis()` devuelven la forma vieja (`reached/here/dropped`, `drop_reasons` con texto). El contrato nuevo pide `passed/cut/error/alarm`, `held`, `promoted`, hitos, `sample_too_small`, `cause_code`. | `registro_vista.py` vs `api-types.ts` | Reescribir esas vistas contra el contrato; no es solo exponerlas. |
| G6 | Las fases posteriores (sombra, paper, candidata) no tienen eventos en el registro; solo hay E0–E3. | `ETAPAS = ("0","1","2","3")` | Recomiendo v1: `paper` derivado de `accounts` (una cuenta por estrategia, PERF-06) y sombra/candidata en cero ("todavía ninguna"). Ver D3. |
| G7 | El recibo del ciclo se guarda como JSON en `cycles.receipt` sin hash propio. | `cerrar_ciclo`, `Cycle` | Calcular y guardar el hash al cerrar; relacionado con RH-05 (formato del recibo, pendiente de Fede). |
| G8 | Los tipos `cycle_closed` y `review_rule_reached` y sus plantillas ya existen (#806), pero nadie los produce; la tabla `events` es por cuenta y el registro ya no comparte base con ella. | `apps/api/events.py`, `apps/api/feed.py` | La API los deriva de los ciclos cerrados del registro, sin escribir en `events` (EM-09). |
| G9 | No hay almacén de precios diarios para la web. Los snapshots son congelados mensuales y `etf-1d-2026-10-amplio` no incluye XLC ni XLRE. | `UNIVERSO_AMPLIO`, `m5-priors.yaml` | Tabla de barras diarias (31 símbolos: 29 + XLC + XLRE), relleno histórico y job diario post-cierre. Es lo más grande de Mercado. |
| G10 | Régimen semanal, candidatos y estado del motor existen como CSV en la carpeta de datos de M5 (`m5-regimen-semanal.csv`), no en DB. | `digest_m5_semanal.py` | La API los lee de ahí (ruta configurable) y devuelve `null` si el motor todavía no calculó. |
| G11 | FRED: existe `macro_vivo.py` y el calendario `calendar_fred.py`, pero no una serie T10Y2Y/USREC lista para servir con recesiones y "dato viejo". | `apps/api/calendar_fred.py` | Ingesta de las dos series y la regla de dato viejo por calendario. |
| G12 | Frontend: la app no usa librería de rutas (coincidencia por regex en `App.tsx`), solo hay Recharts, `CumulativeChart` recibe un `PerformanceReport` y no hay `lightweight-charts`. | `web/src/App.tsx`, `package.json` | Rutas nuevas a mano; `CumulativeChart` genérico para Mercado; dependencia nueva para las velas (D2). |

## 2. Tickets

Granularidad pensada para entregas chicas (límite de ~100 líneas por entrega de Codex). Cada ticket lleva criterio de aceptación verificable.

### A. Backend de Embudo (esperan el merge de PR #806 salvo EM-02)

| ID | Título | Depende de | Aceptación |
|---|---|---|---|
| EM-01 | El registro de hipótesis en su propio Postgres, leído por la web | PR #806 en `main`, D1 | (1) Base propia para el registro (por ejemplo `quantagent_registro`), distinta de la `quantagent` de desarrollo (12 tablas, Alembic en versión vieja, la usan los tests locales). (2) La web entra con un usuario de SOLO LECTURA; la conexión llega al servicio `quantagent-api` por variable de entorno (sin credenciales en el repo); la API abre `hypotheses`, `hypothesis_events`, `hypothesis_attempts` y `cycles` desde esa conexión, con configuración aparte de la base maestra. (3) Backup antes de depender de él: en la VM no hay `pg_dump`, así que el ticket instala el cliente de Postgres o deja un respaldo programado (dump con timer) y prueba una restauración en una base temporal. (4) Orden operativo: backup -> base propia -> `alembic upgrade head` -> `registro importar --dry-run` (se le muestra a Fede) -> OK de Fede -> import real (55 etapas / 202 intentos). La migración y el import real los coordina el hilo del registro (BNqL), no este lote: el ticket cubre el código de lectura de la API, el usuario de solo lectura, el backup y la guía de operación. Tests: la API lee el registro desde una base de prueba con la conexión de solo lectura y responde 503 con mensaje (no 500) si falta la variable de entorno o la base no responde; el usuario de solo lectura no puede escribir. |
| EM-02 | Medida estructurada y causa en cada etapa | PR #806 | Los escritores de etapa guardan en `data` de los eventos de la base del registro `{measure:{metric,value,threshold,comparator,unit}, cause_code}`; `causas_de_motivo` devuelve código y etiqueta; test que falla si se pierde el valor contra el umbral y test de que `measure` y `cause_code` llegan hasta la API. Toca `fabrica` (ver D4). |
| RH-06b | Adaptar `registro_vista` al contrato v0.4 (embudo, ciclos, hipótesis) | EM-02 | Adapta `embudo()`, `ciclos()` y `hipotesis()` de `quantagent/fabrica/registro_vista.py` a `api-types.ts` v0.4: `Funnel` con `period`, `later`, `promoted`, `sample_too_small`; `StageCounts` con `passed/cut/error/alarm`; `DropReason` con `verdict`, `cause_code`, `cause_label`, `example`; `CycleSummary` con `status`, `milestones`, `receipt_id`; suma `review_rule` y `attempts`. ERROR y ALARMA son detenidas (held), no caídas; lo que falte va `null`. Lee el registro desde Postgres. Test que compara las claves contra `docs/03_design/web-design-system/api-types.ts` v0.4 leído del archivo (reemplaza la forma fija v0.3 de `tests/test_registro_vista.py`). Sin endpoints ni frontend. |
| EM-03 | Nombre e idea de cada hipótesis | EM-01 | `name` e `idea` salen del YAML de `hipotesis/` (primera línea de `justificacion`); sin YAML, `idea: null`. Test con hipótesis con y sin YAML. |
| EM-04 | `GET /api/funnel` con embudo y motivos | EM-01, EM-02, RH-06b | Devuelve `Funnel` de `api-types.ts`: 4 etapas con `passed/cut/error/alarm`, `later`, `promoted`, `sample_too_small`, `drop_reasons` con `cause_code`; `ciclo` (`all`, `YYYY-MM`) y `periodo`; "Todos" excluye historial; con ciclo concreto el período se ignora. Tests con registro sintético de cada veredicto. Solo expone por HTTP lo que `registro_vista` (RH-06b) ya devuelve. |
| EM-05 | Ciclos, regla de revisión e historial | EM-01, RH-06b | `GET /api/funnel/cycles`, `/cycles/current`, `/review`, `/history`: `CycleSummary` con hitos (`calendario_ciclos`), `ReviewRule` (racha de ciclos cerrados), `HistorySummary` (55 eventos, 202 intentos por etiqueta, sin mapear a E0–E3). El ciclo actual y la regla ignoran `periodo`. Solo expone por HTTP lo que `registro_vista` (RH-06b) ya devuelve. Los hitos del calendario (días 1, 8 y 25) esperan la decisión de Fede (D5); no lanzar la parte de hitos hasta tenerla. |
| EM-06 | `GET /api/hypotheses` | EM-02, EM-03, EM-04, RH-06b | `Hypothesis[]` con `status` derivado (`active/promoted/discarded/held`), `verdict`, intentos válidos/inválidos, `discard_measure`; filtros `estado/etapa/ciclo/periodo`; orden etapa avanzada primero. Solo expone por HTTP lo que `registro_vista` (RH-06b) ya devuelve. |
| EM-07 | `GET /api/hypotheses/:id` (ficha) | EM-06 | `HypothesisDetail`: línea de tiempo por etapa, intentos con `invalid`, backtests vinculados (`hypothesis_id`), `used_reserve`, `paper_account` desde `accounts` (D3); `404` claro si no existe. |
| EM-08 | Recibo del ciclo | EM-01, EM-29, RH-05 opción B (recibo enriquecido y versionado; lo implementa el hilo del registro) | `GET /api/funnel/cycles/:id/receipt` devuelve `CycleReceipt` de `api-types.ts` v0.4 mapeando desde el recibo versionado (campo de versión incluido); recibo inmutable con hash calculado al cerrar (test de inmutabilidad: el hash no cambia); ciclo abierto devuelve `partial: true` con los snapshots usados hasta ahora; un recibo viejo SIN versión se devuelve degradado/parcial con los campos que no existan en `null`, no con error (test con un recibo sin versión). No lanzar hasta tener el cambio del recibo y los días del calendario (D5). |
| EM-09 | Eventos de feed `cycle_closed` y `review_rule_reached` | EM-01, EM-05, RH-06b | Ya hecho en #806: los tipos están en `EVENT_TYPES` y las plantillas y enlaces en `apps/api/feed.py`. Decidido con el hilo del registro (2026-11-04): la API DERIVA los dos items de los ciclos cerrados del registro; no se escribe nada en la tabla `events` de `maestro.db`. `cycle_closed`: uno por ciclo cerrado, con la hora de cierre y `entered/dropped/promoted` del recibo. `review_rule_reached`: uno por cierre en que la racha de ciclos cerrados sin promovida llega a >= 6, calculada recorriendo los cierres en orden. Misma ventana de tiempo del feed; `build_feed` los construye desde el registro (conexión de solo lectura de EM-01) en lugar de `events`; `account_id = null`; no cuentan para `HomeStatus`. Tests: ciclos cerrados con y sin promovidas, racha que cruza 6 una sola vez por cierre, ventana de tiempo. |
| EM-29 | Recibo del ciclo versión 2 (RH-05b, `QuantAgent-e452.56`) | PR #806 | `cerrar_ciclo` escribe `version: 2`; por etapa arrived/passed/cut/error/alarm (error y alarm son detenidas, no caídas); por hipótesis estado final, etapa donde terminó y motivo; los recibos v1 siguen legibles; tests de cierre con y sin promovida, con error y alarma, y de inmutabilidad. Decisión de Fede del 2026-11-05: lo implementa este lote. No depende del calendario. |

### B. Backend de Mercado (independiente de PR #806; puede empezar ya)

| ID | Título | Depende de | Aceptación |
|---|---|---|---|
| EM-10 | Almacén de barras diarias de mercado | — | Tabla de barras (31 símbolos: 29 + XLC + XLRE) con relleno histórico desde snapshot y Alpaca; comando de actualización; calcula `session_date`, `stale` y `missing_sessions` con el calendario de NYSE. Sin descargas en vivo desde la API. |
| EM-11 | Job diario post-cierre y timer | EM-10 | Timer en la VM (después del cierre, antes del job paper de las 05:00 ART) con log y alarma de dato viejo existente; idempotente. |
| EM-12 | Régimen de SPY | EM-10 | `Regime`: usa `clasificar_regimen`; `since`, `sessions`, `ma200`, `vol20`, mediana 1A, tira de 12 meses; `null` explícito si faltan sesiones. |
| EM-13 | Estado del motor y sectores | EM-10 | `MotorState` y `SectorRow` (11 + SPY, EEM, EWZ como referencia) leyendo los artefactos semanales de M5 (ruta configurable); `enabled/blocked/blocked_reason`, `rank`, `momentum_score`, `state_description`; `null` si el motor no corrió. |
| EM-14 | Retornos, `spark_30d` y "En cartera" | EM-10 | `EtfRow` de los 29 con retornos 1d/1s/1m/3m/YTD (`null` con historia insuficiente y `history_sessions`), y `held_in` desde posiciones reales de todas las cuentas paper (helper compartido con `SectorRow`). |
| EM-15 | Macro: T10Y2Y y USREC | clave FRED en la VM | Ingesta de las dos series, `MacroSeries` activa para T10Y2Y, resto `coming_soon`; `stale` por calendario FRED + 2 días hábiles (1,5 × frecuencia sin calendario); `RecessionPeriod[]`. |
| EM-16 | `GET /api/market` y series bajo demanda | EM-12..15 | Agregado de una llamada, más `/api/market/compare?symbols=&range=` y `/api/market/etfs/:symbol/candles`; esquema Pydantic = `MarketPage`; secciones independientes (si una falla, las demás responden). |

### C. Frontend

| ID | Título | Depende de | Aceptación |
|---|---|---|---|
| EM-17 | Base: tipos, config, textos, rutas y mocks | — | Copias sincronizadas de `tokens.css`, `config.ts`, `help-texts.es.json` y tipos; rutas `/embudo`, `/embudo/ciclo/:id`, `/hipotesis/:id`, `/mercado`, `/mercado/etf/:symbol`; `AppNav` con Embudo y Mercado activos; handlers MSW con los números del canvas (`?mock=1`); test de contrato contra el esquema de la API. |
| EM-18 | Componentes de Embudo | EM-17 | `StageVerdictBadge`, `ReviewCounter`, `CycleMilestones`, `CycleFilter`, `HistoryPanel`, `FunnelChart`, `DropReasons`, `CycleSummaryCard` y `HypothesisCard` extendidos; stories claro/oscuro; tests de cada estado `null` del contrato (sin `idea`, sin `last_metric`). |
| EM-19 | Pantalla Embudo | EM-18 | Con mocks y luego con la API (EM-04..06); filtros en la URL, período deshabilitado con ciclo elegido, ERROR/ALARMA aparte, estados cargando/vacío/error por sección. |
| EM-20 | Recibo del ciclo y ficha de hipótesis | EM-18 | `CycleReceipt` (pantalla en teléfono, panel en escritorio), `StageTimeline`, `AttemptList`; enlaces desde tarjetas y desde Rendimiento del backtest. |
| EM-21 | Componentes base de Mercado | EM-17 | `MarketAsOfLine`, `RegimeCard` + `RegimeStrip`, `MotorStateCard`, `MacroIndicatorCard` (activa y fantasma); stories y tests. |
| EM-22 | Sectores y 29 ETFs | EM-21 | `SectorTable` (sobre `DataTable`) + `SectorHeatmap` con umbrales de `marketConfig`; `EtfTable` con casillas (máx. 5) y `CompareChart` (`CumulativeChart` genérico, sin depender de `PerformanceReport`). |
| EM-23 | Macro y detalle de ETF | EM-21, D2 | `MacroLineChart` con sombreado de recesiones y nota cuando no hay; `EtfDetail` con velas (`lightweight-charts`), panel lateral de 560 px en escritorio. |
| EM-24 | Pantalla Mercado | EM-21..23 | Integra bloques con mocks y luego con `/api/market`; falla por sección; `Notice stale` y cifras atenuadas. |
| EM-25 | Inicio: ciclo actual y evento de revisión | EM-18, EM-09 | `CycleSummaryCard` compacta en Inicio (hoy fantasma) y `FeedItem` de `review_rule_reached`. |

### D. Transversales

| ID | Título | Depende de | Aceptación |
|---|---|---|---|
| EM-26 | E2E y accesibilidad | EM-19, EM-24 | Playwright: rutas nuevas, cada estado vacío/error, teclado y 200 % de zoom; el job de CI de la web (ubuntu-24.04) pasa. |
| EM-27 | Manual de usuario | EM-19, EM-24 | Secciones Embudo, Ficha y Mercado en `docs/manual` (inglés), glosario con los términos nuevos; los tests de cobertura del glosario pasan. Cumple la regla de revisión del manual al cerrar el lote. |
| EM-28 | Deploy y migración en producción | EM-01, EM-10, EM-27 | Backup verificado del Postgres del registro (y de `maestro.db` para lo de Mercado), variable de entorno de solo lectura configurada en el servicio `quantagent-api`, tag `paper-*` y redeploy (mismo procedimiento que jtvz); verificación con la API local y Playwright visual. Sin sincronización entre bases. |

## 3. Orden y paralelismo (máximo 3 sesiones)

1. **Ahora (#806 ya está en `main`):** EM-17 → EM-18/EM-21 (frontend con mocks), EM-10 → EM-12/13/14, EM-15.
2. **Cuando #806 entre a `main`:** EM-01, EM-02, EM-03 → EM-04..09 → EM-19/20/25.
3. **Cierre:** EM-16 y EM-24, EM-26, EM-27, EM-28.

Camino crítico: EM-10 (precios) y EM-01/02 (registro y medida estructurada). Estimación gruesa: unos 28 tickets chicos; Mercado y Embudo pueden avanzar en paralelo porque solo comparten EM-17.

## 4. Decisiones (aceptadas por Fede el 2026-11-04)

| # | Decisión | Resuelto |
|---|---|---|
| D1 | ¿Dónde lee la API el registro de hipótesis? | **Aceptada (actualizada 2026-11-04):** el registro vive directamente en su propio Postgres y la web lo lee con un usuario de solo lectura; no hay copia a `maestro.db` (EM-01). La web sigue sin poder escribir. |
| D2 | Nueva dependencia `lightweight-charts` (el design system la pide para velas). | **Aceptada:** aceptarla; solo se carga en la pantalla de detalle de ETF. |
| D3 | Fases posteriores (sombra, paper, candidata): no hay eventos en el registro. | **Aceptada:** v1: solo `paper`, derivado de `accounts`; sombra y candidata en "todavía ninguna". |
| D4 | EM-02 toca `quantagent/fabrica`, que M-web había dejado fuera. | **Aceptada:** permitirlo: sin medida estructurada, el motivo de caída solo puede mostrarse como texto. |
| D5 | Formato del recibo (RH-05) y días del calendario (1, 8, 25): siguen pendientes. | **Aceptada como orden de trabajo:** cerrarlos antes de EM-08 y de los hitos de EM-05. Los valores siguen pendientes de Fede. |

## 5. Cómo cargar

Una sesión en la VM crea los tickets con `bd create` desde esta lista, bajo el epic `QuantAgent-e452`, y deja WEB-14 (`e452.14`) y FE-08 (`e452.22`) como paraguas desbloqueados con dependencia de los nuevos. Registrar los IDs reales en el README de `docs/02_planning/`. No ejecuté `bd` desde la nube: la base de Beads de la VM está desfasada del JSONL y los tickets se crean allí.
