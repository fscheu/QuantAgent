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
