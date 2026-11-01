# M5-55a: refresco de fundamentales para paper H3

`refrescar_fundamentales(hoy, composicion, ...)` recibe la fecha del llamador y
solicita sólo los miembros de la última fecha de la composición. Crea
`fundamentales-sp500-AAAA-MM-DD` en una carpeta temporal dentro del directorio
de snapshots; publica mediante rename después de verificar hashes y cobertura.
La CLI usa la fecha local de la VM. El resultado devuelve `snapshot_vigente`
(o `None`), `nuevo` y `motivo`. No cambia la configuración del screener.

En la VM, desde el repo principal:

```bash
QUANTAGENT_SNAPSHOT_DIR="$PWD/local-data" python -m quantagent.cli data fundamentales refrescar --composicion sp500-2026-10
```

La cobertura se evalúa en la última sesión NYSE hasta `hoy`, con
`reporte_cobertura_fundamentales(..., elegibilidad=True)`. Se exige al menos
`UMBRAL_COBERTURA_FUNDAMENTALES = 60.0` por ciento de elegibles en cada sector:
el corte de [cobertura del screener](QuantAgent-fnpr.4.10-IM-cobertura-screener.md).
Un miembro requiere datos en al menos tres de las cuatro métricas fundamentales:
ROIC y margen operativo (ROE y margen pretax en Financials), aceleración de
ingresos y crecimiento de EPS. Se conserva la antigüedad máxima de 200 días.
La prueba no incluye precios ni retornos relativos; no reemplaza la elegibilidad
completa del screener. Un sector desconocido no cuenta como elegible.

Si ya existe el nombre del día, se valida y reutiliza sin descargar ni escribir.
Si la descarga, los hashes o la cobertura fallan, se elimina únicamente el
temporal creado por esta ejecución. Los snapshots existentes quedan intactos.
El fallback busca nombres diarios anteriores, de más reciente a más antiguo,
validando hashes y cobertura en la fecha de cada snapshot. Descarta nombres
futuros, parciales y sufijos de backups. Si ninguno sirve, intenta el congelado
`fundamentales-sp500-2026-10`, validando sus hashes (su cobertura histórica no se
reinterpreta como cobertura de datos vivos). La CLI imprime el vigente y el
motivo; devuelve 1 sólo cuando no encuentra ninguno válido.

Limitación conocida: `sp500-2026-10` llega hasta **2026-08-18** y su fuente
`fja05680/sp500` está fijada al SHA
`a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`. Este refresco usa esos últimos
miembros disponibles; actualizar la composición es un paso aparte y queda
fuera de este cambio. No se descargaron datos reales ni se modificaron los
snapshots del backtest.

Las pruebas de `tests/test_fundamentales_refresco.py` usan composición local y
el parámetro `downloader`; sólo sustituyen el mapa SEC de ticker a CIK para
evitar red. Cubren creación, universo, idempotencia, rechazo por descarga o
cobertura, selección del previo y códigos de salida de la CLI.

Validación en el worktree (entorno Python existente; sin instalar dependencias):

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fnpr661-mpl NUMBA_CACHE_DIR=/tmp/fnpr661-numba /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_fundamentales_refresco.py tests/test_fundamentales.py tests/test_sp500.py tests/test_snapshot*.py tests/test_cli_help.py tests/test_cli_reference.py -q
PATH="/home/azureuser/repos/projects/QuantAgent/.venv/bin:$PATH" PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fnpr661-mpl NUMBA_CACHE_DIR=/tmp/fnpr661-numba bash scripts/loop/smoke.sh
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/fnpr661-mpl NUMBA_CACHE_DIR=/tmp/fnpr661-numba /home/azureuser/repos/projects/QuantAgent/.venv/bin/python scripts/gen_cli_reference.py --output-dir docs/manual/reference/cli
```

Resultados: pytest **43 passed, 0 failed, 1 warning** (29,16 s); smoke **exit 0**,
9 trades, PnL 561,18; referencia regenerada y comprobada por sus tests.
`PYTHONPATH` evita que el generador importe el repo principal instalado en el
entorno en lugar de este worktree.

Handoff: el commit solicitado quedó bloqueado por el sandbox. `git add` falla
al crear `/home/azureuser/repos/projects/QuantAgent/.git/worktrees/QuantAgent-fnpr.6.6.1/index.lock`
con `Read-only file system`; no se llegó a ejecutar `git commit`. Los cambios
quedan sin stage en la rama `loop/QuantAgent-fnpr.6.6.1`, pendientes de commit
desde una sesión con escritura sobre el directorio administrativo de Git.
Mensaje requerido: `feat(QuantAgent-fnpr.6.6.1): refresco de fundamentales con fallback al snapshot previo`.

## Corrección 2026-10-31

QuantAgent-fnpr.6.6.1b corrige M5-55a a partir de las mediciones provistas de
la VM con SEC real: descargar los 503 `companyfacts` del S&P 500 llevó unos
6 minutos; AEE y MAR (2 de 503) fallaron con `http.client.IncompleteRead`.
Algunos pedidos demoraron entre 10 y 40 s. Dos corridas reales de
`quantagent data fundamentales refrescar` quedaron colgadas más de 25 minutos
sin escribir nada. Estas mediciones no se repitieron en esta entrega sin red.

`descargar_facts_cik` ahora reintenta también `IncompleteRead` y
`ConnectionError`: conserva los tres intentos, las esperas de 2 y 4 s y la
salida inmediata ante `HTTPError`. `FACTS_TIMEOUT_TOTAL = 90` segundos limita
cada intento completo, incluyendo apertura, lectura y parseo. La descarga corre
en un hilo daemon y el llamador espera un `Future` con ese límite; no usa un
executor que espere al hilo al salir. El timeout de socket de 30 s se conserva.
Un intento vencido puede terminar su lectura después: el hilo no se cancela y
sólo escribe su propio Future, nunca el snapshot. Tres intentos agotados toman
como máximo aproximadamente 276 s (3 × 90 + 2 + 4), más planificación del sistema.

Un fallido aislado ya no impide publicar: se rechaza cuando los fallidos superan
`MAX_FALLIDOS_PCT = 2.0` por ciento de los tickers pedidos, o cuando no pasa la
cobertura sectorial existente. El 2 % exacto se acepta. La caída total sigue
rechazándose y el snapshot previo queda intacto. El resultado aceptado incluye
`fallidos` y explica la aceptación parcial en `motivo`; al reutilizar el snapshot
del día también devuelve sus fallidos. La CLI informa cada 50 tickers procesados
por stderr mediante el callback opcional `progreso(n, total)`, sin cambiar la ayuda.

Los tests agregados usan snapshots locales y un opener falso, sin conexiones:
aceptación de 1/50 fallidos, rechazo de 2/50 conservando el previo, reintentos de
lectura incompleta y conexión, límite total en apertura y lectura bloqueadas,
`HTTPError` sin reintento y progreso por stderr. La referencia CLI conserva su
contenido porque no cambió la ayuda.

Validación de la corrección (comandos desde este worktree):

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.pytest_cache/mpl" NUMBA_CACHE_DIR="$PWD/.pytest_cache/numba" /home/azureuser/repos/projects/QuantAgent/.venv/bin/python scripts/gen_cli_reference.py --output-dir docs/manual/reference/cli > .pytest_cache/cli-reference.log
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/.pytest_cache/mpl" NUMBA_CACHE_DIR="$PWD/.pytest_cache/numba" /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_fundamentales_refresco.py tests/test_fundamentales.py tests/test_sp500.py tests/test_cli_help.py tests/test_cli_reference.py --basetemp=.pytest_cache/tmp -q
git diff --check
```

Resultados: generador **exit 0**, sin diferencias en la referencia; pytest
**42 passed, 0 failed, 1 warning** en 36,23 s; `git diff --check` sin errores.
Las cachés de Numba y Matplotlib se dirigen al worktree: sin esa configuración,
la primera corrida se frenó en colección porque Numba no podía escribir su caché
junto al entorno compartido. Una corrida intermedia dio 41 passed y 1 failed
por cambiar innecesariamente el mensaje esperado del caso de caída total;
se restauró esa expectativa antes de la validación final.

Handoff de QuantAgent-fnpr.6.6.1b: cambios sin stage en
`loop/QuantAgent-fnpr.6.6.1b`, sin commit, push, PR ni uso de Beads o red,
según la instrucción de esta entrega. Pendiente revisión por la otra sesión.

## Corrección 2026-11-01

QuantAgent-fnpr.6.6.1c reemplaza el corte absoluto del 60 % por una comparación
contra el snapshot anterior válido que usaría el fallback. El 60 % de la
[medición del screener](QuantAgent-fnpr.4.10-IM-cobertura-screener.md) marcaba
sector-años con cobertura baja; no era una compuerta de publicación. Según la
medición provista de SEC real del 2026-10-31, Utilities tenía 45 % y Real Estate
57 %, prácticamente iguales al congelado de octubre. No se repitió esa descarga.

Se rechaza si algún sector cae más de `MAX_CAIDA_COBERTURA_PP = 10.0` puntos
porcentuales respecto del vigente, o queda debajo de
`MIN_COBERTURA_SECTOR = 20.0` por ciento. Ambos snapshots se miden con
`reporte_cobertura_fundamentales(..., elegibilidad=True)` en la misma última
sesión NYSE hasta `hoy`, incluso si el anterior es el congelado de octubre.
Sin anterior válido sólo rige el piso; se aceptan exactamente 10 puntos de caída
y exactamente 20 % de cobertura. Se retira el parámetro `umbral_cobertura`.

El fallback mantiene el orden de diarios anteriores y después octubre: los
diarios validan hashes y el nuevo piso en su propia fecha; octubre conserva su
validación de hashes. El resultado agrega `cobertura_por_sector`, con `nueva` y
`vigente` por sector (`None` sin anterior), y `motivo` incluye esa comparación al
aceptar o rechazar por cobertura. Se conservan el límite de fallidos del 2 %, la
publicación mediante rename y la reutilización sin pisar el snapshot del día.

Los tests usan composición y snapshots locales, sin red ni mocks del cálculo de
cobertura. Cubren 60 % → 45 % rechazado sin modificar el previo, 45 % estable
aceptado, caída exacta de 10 puntos, piso y ausencia de anterior; se prueban tanto
un previo diario como el congelado de octubre. Financials permanece al 100 %
para verificar que un solo sector alcanza para rechazar.

Validación desde este worktree:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$PWD/.cache/mpl NUMBA_CACHE_DIR=$PWD/.cache/numba /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_fundamentales_refresco.py tests/test_fundamentales.py tests/test_cli_help.py tests/test_cli_reference.py --basetemp=.cache/pytest-tmp -q
PATH=/home/azureuser/repos/projects/QuantAgent/.venv/bin:$PATH TMPDIR=$PWD/.cache PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$PWD/.cache/mpl NUMBA_CACHE_DIR=$PWD/.cache/numba bash scripts/loop/smoke.sh
git diff --check
```

Handoff de QuantAgent-fnpr.6.6.1c: cambios sin stage, sin commit, push, PR ni uso
de Beads o red, según la instrucción de esta entrega. La otra sesión realiza la
revisión e integración. Se elimina `.cache` al terminar.

Resultados: pytest **51 passed, 0 failed, 1 warning** en 129,18 s; smoke
**exit 0**, 9 trades y PnL 561,18; `git diff --check` sin errores. La ayuda y la
referencia CLI no cambiaron. Tamaño: **173 líneas agregadas + borradas en
3 archivos**, sin cambios en fixtures.
