# OpenBB como plataforma de datos y espacio compartido

Ticket: `QuantAgent-6ie`. Fecha: 2026-10-06. Estado: **propuesta, falta decisión de Fede**.

## Resumen

1. **Qué es hoy:** OpenBB Inc. cerró. El 2026-08-25 anunció que dejaba de operar y liberaba todo su código. Desde el 2026-09-29 todo está bajo licencia Apache 2.0.
2. **¿Lo vendieron?** No hay venta confirmada. Pero lo que recuerda Fede es casi cierto: BrightQuery, un proveedor de datos de empresas, financia la fundación que sigue el proyecto (OpenBQ), contrató a un tercio del equipo, y el sitio pago ahora anuncia "BQ Workspace — Coming Soon".
3. **Datos de mercado: no adoptar.** La versión 5 sacó Yahoo y los proveedores pagos del paquete. No agrega nada sobre `yfinance` para el backtest.
4. **Nota macro: adoptar en parte.** Su servidor MCP le da al agente unas 1.000 consultas a fuentes oficiales (Fed, BCE, FRED, CBOE), con números citables. No cubre Argentina.
5. **Espacio compartido: no adoptar todavía.** El Workspace se liberó hace 6 días. Tiene un solo commit y está pensado para escritorio. Los gráficos técnicos (TradingView) no vienen incluidos, y el agente solo puede editar tableros si Fede tiene el navegador abierto. Mientras tanto: páginas HTML que genera y publica el agente. OpenBB se revisa en enero de 2027.

## A. Estado actual

| Pregunta | Respuesta | Fuente |
|---|---|---|
| ¿Qué pasó con la empresa? | Cierre anunciado el 2026-08-25. El fundador dijo que no encontraron un modelo de negocio sostenible | [1], [2] |
| ¿Hubo venta? | No verificado. No hay comunicado de compra. BrightQuery financia la fundación OpenBQ, cofundada por su CEO y el fundador de OpenBB | [3], [4] |
| ¿Quién es el dueño hoy? | El código es de cualquiera (Apache 2.0). El copyright sigue a nombre de OpenBB Inc. Quien lo mantiene: OpenBQ | [3], [5] |
| ¿Qué pasa con el Workspace web pago? | `pro.openbb.co` muestra "BQ Workspace — Coming Soon" | [6] |
| ¿FINOS (Linux Foundation) lo adopta? | Solo propuesto: hay una propuesta abierta ("Open Workspace"). No está aceptado | [7] |

### Productos (verificado en el código, 2026-10-06)
| Pieza | Estado | Código abierto | Licencia |
|---|---|---|---|
| Terminal original (de consola) | Discontinuada antes del cierre. Hoy existe `openbb-cli` | Sí | Apache 2.0 |
| ODP (Open Data Platform), librería de Python `openbb` | Activa. v5.0.0 del 2026-09-29 | Sí | Apache 2.0 (era AGPL hasta v4) |
| Servidor MCP `openbb-mcp-server` | Activo. v2.0.1 del 2026-09-28 | Sí | Apache 2.0 |
| Workspace (web) + su MCP para agentes | Liberado el 2026-09-30, en un solo commit | Sí, sin TradingView | Apache 2.0 |
| Copilot (agente propio) | Liberado, pero necesita el backend de IA de OpenBB, que ya no existe | Sí | Apache 2.0 |
| Proveedor Yahoo `openbb-yfinance` 2.0 | Salió del repo oficial. Vive en el repo personal del mantenedor | Sí | **AGPL** |
| Proveedores FMP, Tiingo, Intrinio, Polygon | Quedaron en la v4. No funcionan con la v5 | Sí | AGPL |

**Licencia y repo público:** Apache 2.0 no obliga a nada más que mantener el aviso de copyright. Sin problema. Lo único con riesgo es `openbb-yfinance`, que es AGPL: QuantAgent tendría que publicarse con licencia compatible si lo distribuye. No hace falta usarlo.

**Salud del proyecto** (repo `OpenBB-finance/OpenBB`): 35 commits en los últimos 6 meses, 29 de una sola persona. Último commit: 2026-10-02. Sin commits del 2026-07-20 al 2026-09-29; después salió la v5 de golpe. Riesgo: una sola persona sostiene el código y no hay empresa detrás. A favor: la v5 es nueva y tiene tests completos en el núcleo.

**Precios:** todo lo que queda es gratis. Antes había un plan gratuito del Workspace en la nube (Copilot con 20 consultas por día) y planes pagos por usuario. Ese servicio ya no funciona: la dirección muestra "BQ Workspace — Coming Soon" [6], [8].

## B. Las tres necesidades

**B1. Datos de mercado**

| Pregunta | Respuesta |
|---|---|
| Proveedores en la v5 | 19 fuentes, casi todas oficiales: FRED, Fed, BCE, BLS, EIA, IMF, OECD, CBOE, Nasdaq, TMX (Toronto), SEC, CFTC, Deribit, Tesoro de EE.UU., USDA |
| Gratis sin clave | CBOE, Nasdaq, TMX, Fed, BCE, IMF, OECD, Tesoro |
| Gratis con clave propia | FRED, BLS, EIA (registro gratis) |
| Precios de acciones y ETFs | CBOE, Nasdaq, TMX. Yahoo solo con el paquete AGPL aparte |
| ¿Qué agrega sobre `yfinance`? | Para precios diarios de ETFs, nada. Es una capa más entre QuantAgent y la misma fuente |
| ¿Snapshots reproducibles con hash? | No. ODP baja datos. No guarda versiones ni calcula hashes. Eso lo sigue haciendo V02 |
| ¿Más de 730 días en intradía? | No. El límite es de Yahoo. Los proveedores pagos que daban más historia (FMP, Polygon, Intrinio) salieron de la v5 |

**B2. Nota macro semanal** (lo que pide el borrador del 2026-09-17)

| Dato | ¿Lo cubre? | Fuente en ODP | Clave |
|---|---|---|---|
| Fed funds, 10Y EE.UU. | Sí | Fed (curva de tasas), FRED | Fed no; FRED sí |
| Expectativas de tasas (FedWatch) | No. Solo las proyecciones de la Fed | FRED | Sí |
| Tasas BCE, 10Y Alemania | Tasas BCE sí. 10Y Alemania: no verificado | BCE | No |
| BCRA (tasa, reservas) | No | — | — |
| DXY, EUR/USD, USD/BRL | Monedas sí. DXY: no verificado | TMX | No |
| USD/ARS oficial, CCL, brecha | No | — | — |
| S&P 500, Nasdaq, VIX | Sí | CBOE, Nasdaq | No |
| Stoxx 600, Nikkei, MSCI EM, Merval | No verificado | — | — |
| Brent, oro, cobre, soja | Petróleo sí (EIA, FRED). Soja en parte (USDA). Oro y cobre: no verificado | EIA, FRED, USDA | EIA y FRED sí |
| Spreads high yield | Sí | FRED | Sí |
| Calendario económico | Sí | Nasdaq, FRED, BCE, BLS | Nasdaq no |
| Noticias | Solo de bancos centrales y gobierno | `openbb-news` | No |

**B3. Espacio compartido entre Fede y agentes** (código del Workspace, 2026-09-30)

| Pregunta | Respuesta |
|---|---|
| ¿Qué ofrece? | Tableros con widgets: gráficos, tablas, notas y HTML. Cada widget lee de un backend propio o de ODP |
| ¿Se autoaloja sin escritorio? | Sí, en teoría: una imagen Docker ("Lite") con todo adentro y SQLite. Que la imagen pública exista: no verificado. La guía de instalación pide credenciales que daba OpenBB |
| ¿Funciona desde el celular? | Hay pantallas para móvil en el código (menú y Copilot). Uso real: no verificado |
| ¿Un agente puede crear tableros y gráficos? | Sí. Tiene un MCP propio con herramientas para crear tableros y agregar gráficos, tablas, notas y HTML |
| Condición | Ese MCP manda las órdenes **al navegador abierto**. Si Fede no tiene la pestaña abierta, el agente no puede dibujar nada |
| Gráficos técnicos | Las velas avanzadas usan TradingView, que es software pago y no viene incluido. Sin él, ese widget dice "no disponible" |
| ¿Hace falta cuenta en la nube? | No para la versión Lite. Copilot sí: el de OpenBB ya no existe. Hay que conectar un agente propio |

## C. Prueba práctica (2026-10-06, contenedor temporal, Python 3.13)

| Paso | Resultado |
|---|---|
| `pip install openbb openbb-mcp-server` | OK en 111 s. El entorno pesa **1,1 GB** |
| Primer `import` | 44 s (arma el índice de comandos la primera vez) |
| SPY diario (CBOE, Nasdaq, TMX) | Falló: la red de la prueba no llega a `cdn.cboe.com`, `api.nasdaq.com` ni `www.tmx.com` (`curl` tampoco). Para repetirlo en la VM: `python -c "from openbb import obb; print(obb.cboe.equity.historical(symbol='SPY', start_date='2024-01-01').to_dataframe().tail())"` |
| Serie 10Y de FRED | Pidió clave: `Missing credential 'fred_api_key'`. FRED no funciona sin clave. En la VM, con la clave: `obb.user.credentials.fred_api_key = '<clave>'; obb.fred.economy.fred_series(symbol='DGS10')` |
| Servidor MCP (`openbb-mcp`) | Arrancó sin configurar nada, solo en la máquina local. Listó **1.056 herramientas** |
| Llamar una herramienta por MCP | Funciona. Devolvió el mismo error de clave de FRED, bien explicado |
| Ojo | La v5 no instala `obb.equity` ni `obb.economy` por defecto: los comandos van por fuente (`obb.cboe…`) |
| Workspace Lite | No probado: el registro de Docker está bloqueado en este entorno |

## D. Alternativas para el espacio compartido

| Opción | Autoalojable | Celular | El agente escribe | Mantenimiento |
|---|---|---|---|---|
| OpenBB Workspace Lite | Sí (Docker) | No verificado | Sí, con el navegador abierto | Alto: código nuevo, un solo mantenedor, sin TradingView |
| Streamlit (`apps/streamlit/`, ya existe) | Sí | Aceptable | Indirecto: escribe datos en la base y la app los muestra | Medio: hay que programar cada vista |
| Cuadernos reactivos (marimo) | Sí | Pobre | Sí: el agente edita el cuaderno (es un `.py`) | Medio |
| Grafana | Sí | Bueno | Sí, por API. Pensado para series de tiempo, no para análisis | Medio. Débil para velas y texto |
| Páginas HTML generadas por el agente y publicadas | Sí (o artefacto) | Bueno | Sí, de forma directa | Bajo: no hay servidor que mantener |

**Lectura:** el criterio que más pesa para Fede es "verlo desde el celular lo mismo que el agente". Las páginas HTML lo cumplen hoy y no suman servidores. No son interactivas en vivo: cada análisis es una foto con fecha. Para la nota macro eso alcanza.

## E. Recomendación (cada fila es una decisión para Fede; la columna "Decisión" es lo recomendado)

| Necesidad | Decisión | Motivo |
|---|---|---|
| 1. Datos de mercado | **No adoptar** | La v5 sacó Yahoo y los proveedores pagos. No da snapshots ni más historia intradía. Suma 1,1 GB |
| 2. Nota macro (`QuantAgent-wwi`) | **Adoptar en parte** | El MCP da números de fuentes oficiales con fecha. La nota ya pide eso. Argentina y FedWatch siguen por web |
| 3. Espacio compartido | **No adoptar todavía** | El código se liberó hace 6 días, la gobernanza no está clara y el agente depende del navegador abierto. Usar páginas HTML. Revisar en enero de 2027 |

**Filas del plan de M2 que cambian:** ninguna obligatoria.
- V02 (`data snapshot`): sigue con `yfinance`. Se agrega una línea al documento: "OpenBB evaluado en `QuantAgent-6ie`, descartado".
- V32 (fuente para 4h): OpenBB queda descartado como alternativa a Yahoo. La comparación queda entre Yahoo y una fuente paga.
- V28 (`report explain`): si Fede elige páginas HTML, el informe sale también en HTML además de markdown. Esto se decide en V27.

**Experimento más chico (necesidad 2, una semana):**
1. Sacar una clave gratis de FRED.
2. Correr `openbb-mcp --allowed-categories fred,federal_reserve,ecb,cboe,nasdaq` en la VM, solo en `127.0.0.1`.
3. Conectarlo al skill de la nota macro y generar una nota con MCP y otra solo con búsqueda web.
4. Comparar: ¿cuántos números de la tabla vienen con fuente y fecha exacta en cada una? Si la versión con MCP no gana en al menos 5 filas, se descarta.

**Experimento para la necesidad 3, si Fede igual quiere probar (2 horas):** levantar Workspace Lite con Docker en la VM, detrás de Tailscale, abrirlo desde el celular y pedirle a un agente un tablero con SPY y VIX. Si no levanta en 2 horas, se descarta.

## Qué no pude verificar

- Si BrightQuery compró activos (solo hay financiamiento y contrataciones). Si FINOS acepta el Workspace. Issues abiertos (API de GitHub bloqueada). Si la imagen Docker `openbb/lite` es pública.
- Uso real del Workspace en el celular. Bajada real de SPY y datos macro (red bloqueada). Cobertura de DXY, 10Y Alemania, Stoxx 600, Nikkei, MSCI EM, Merval, oro y cobre. Fecha de cierre de la terminal original.

## Qué NO se hizo

- No se editó `docs/04_decisions/README.md` (otro PR en curso toca ese índice). Propuesta de fila para después: `QuantAgent-6ie-DC-openbb-plataforma.md — OpenBB: no para datos, en parte para macro, todavía no como espacio compartido`. No se tocó código, dependencias, `.beads/`, `PLAN-CONTINUACION.md` ni `milestone-tracker.md`. No se instaló nada en la VM: la prueba fue en un contenedor temporal.

## Fuentes (consultadas el 2026-10-06)
1. Post de OpenBB en X, 2026-08-25: https://x.com/openbb_finance/status/2092275555408269796
2. Post de Didier Lopes en X, 2026-08-25: https://x.com/didier_lopes/status/2092274517498347648 y https://didierlopes.com/blog/openbb-is-shutting-down-and-going-open-source/
3. Post de Didier Lopes en X, 2026-10-01 ("It's official… Apache license"): https://x.com/didier_lopes/status/2105751125764768059
4. OpenBQ ("funded by BrightQuery"): https://openbq.org/
5. Repo ODP, `LICENSE` cambiado a Apache 2.0 en el commit "V5 (#7489)" del 2026-09-29: https://github.com/OpenBB-finance/OpenBB
6. `pro.openbb.co/app` muestra "BQ Workspace — Coming Soon" (resultado de búsqueda, 2026-10-06)
7. Propuesta FINOS: https://github.com/finos/community/issues/445
8. Página de precios anterior: https://openbb.co/pricing
9. Repo Workspace, commit único del 2026-09-30 (`lite/README.md`, `NOTICE`, `backend-api/backend/workspace_mcp/`): https://github.com/OpenBB-finance/workspace
10. PyPI: `openbb` 5.0.0 (2026-09-29), `openbb-mcp-server` 2.0.1 (2026-09-28), `openbb-yfinance` 2.0.0 (AGPL, repo `deeleeramone/openbb-danglewood`): https://pypi.org/project/openbb/

Nota de método: los sitios `openbb.co`, `didierlopes.com` y X estaban bloqueados para lectura directa. Las fechas de los posts salen del identificador del post. Su contenido sale de los resúmenes del buscador. El código, la licencia y los commits se leyeron directo de GitHub y PyPI.
