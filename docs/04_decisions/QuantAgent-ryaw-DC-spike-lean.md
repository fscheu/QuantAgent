# QuantAgent-ryaw: spike de LEAN compilado desde fuente

Fecha: 2026-10-13. Resultado negativo por bloqueo de Python.NET; pendiente de revisión del PM.
Contexto: [decisión del pivot, §§7–10](./2026-10-12-DC-quantconnect-y-pivot-lean.md).
Comandos y estado de los scripts: [README del spike](../../scripts/lean_spike/README.md).

| Criterio binario | Resultado | Evidencia |
|---|---|---|
| (a) SMA 5/5 y RSI 187/187, o cada diferencia de RSI explicada | **No** | Los controles tienen 5 y 187 trades; LEAN falla antes de inicializar el algoritmo. No hay trades válidos de LEAN para comparar ni clasificar. |
| (b) Completa el backtest sin cuenta de QuantConnect | **No, no verificado** | El Launcher arranca sin cuenta, login ni token, pero ningún backtest completa. En los tres arranques, `strace` registra cero llamadas AF_INET/AF_INET6. |
| (c) Disco y tiempos, incluida una corrida completa | **No, parcial** | Instalación y build medidos; arranques fallidos medidos. Falta el tiempo de una corrida completa. |

No se demostró que LEAN necesite una cuenta: el bloqueo observado es de inicialización de Python. Esta ejecución no aporta evidencia suficiente para aprobar P1 ni para recomendar P2 por un problema de credenciales. Conforme a la puerta del spike, P1 queda sin aprobar; el PM decide el siguiente paso. Se respetó el límite de tres intentos, sin probar variantes adicionales.

## Alcance y controles

Se usó el entorno `/home/azureuser/repos/projects/QuantAgent/.venv`, con `PYTHONPATH` apuntando a este worktree. Se verificó que `quantagent.__file__` apunta a `/home/azureuser/repos/projects/QuantAgent-spike-lean/quantagent/__init__.py`. La base SQLite está en W/try.db y se creó con `quantagent.database.init_db()`.

SPY, snapshot `etf-1d-2026-10`, 2007-01-03..2018-12-31, 100% del capital, perfil `etf` vigente: slippage 0.0005 por lado y comisión 0. El escritor exportó 3020 velas. Los únicos datos de precios preparados para LEAN son los del snapshot; de su carpeta Data se copiaron solamente market-hours y symbol-properties. No se leyó un tramo posterior a 2018-12-31 para los controles ni la exportación.

Los cuatro controles están en W/control/. El motor propio conserva su llenado al cierre; el adaptador usa `--fill next-open`. Por eso tener el mismo conteo no significa tener las mismas fechas.

| Estrategia | Control | Trades del control | Coincidencias LEAN por entrada, salida y lado | Desvío relativo del PnL total de LEAN |
|---|---|---:|---|---|
| sma-cross | bt_adapter next-open | 5 | No evaluable de 5 | No evaluable |
| sma-cross | motor propio | 5 | No evaluable de 5 | No evaluable |
| rsi | bt_adapter next-open | 187 | No evaluable de 187 | No evaluable |
| rsi | motor propio | 187 | No evaluable de 187 | No evaluable |

Como comprobación independiente, se ejecutó `compare_trades.py` entre ambos controles. SMA: 0/5 coincidencias por ambas fechas y lado; RSI: 0/187, con 27 entradas del mismo día y lado. El desvío relativo del PnL total del motor contra el adaptador es +3.9734% para SMA y +41.4781% para RSI. Son diferencias entre controles, no resultados de LEAN. En el adaptador hay 5 largos para SMA y 82 largos/105 cortos para RSI.

Clasificación de RSI: **no evaluable**. No hay trades candidatos de LEAN; no se atribuyeron diferencias a redondeo, activación del stop desde la vela siguiente ni otras causas. Los 187 trades del control siguen sin comparación. Cero trades generados por un error de inicialización no constituye una estrategia con cero operaciones.

## Instalación y mediciones

Todo se descargó o compiló dentro de W = `/home/azureuser/spike-lean-work`. No se usaron git, sudo, Docker, LEAN CLI ni instalaciones de sistema. La prohibición de git impidió realizar un clon superficial literal: se descargó el archivo de fuente de GitHub, fijado a la revisión `80e7843f645673bcbeaab963049f76f20f6785e1`, y se extrajo en W/Lean. La revisión está también en W/lean-revision.txt.

No hay global.json en esa revisión. Launcher/QuantConnect.Lean.Launcher.csproj pide `net10.0`; `dotnet-install.sh --channel 10.0` instaló SDK 10.0.401. Build Release con `-m:2` y `DOTNET_PROCESSOR_COUNT=2`: 0 errores y 4446 advertencias upstream; no se modificó el código de LEAN. Incluye las advertencias de NuGet sobre dependencias vulnerables; no se investigaron ni corrigieron en este spike.

Mediciones de tamaño con `du -sh` (MiB, redondeados), tiempos de pared con `/usr/bin/time`:

| Componente / paso | Disco | Tiempo |
|---|---:|---:|
| Fuente extraída antes del build | 290 MiB | 14.89 s, consulta de revisión + descarga + extracción |
| Archivo comprimido de fuente conservado | 211 MiB | Incluido arriba |
| SDK y runtimes W/dotnet | 624 MiB | 11.51 s, dotnet-install; descarga previa del instalador fuera de esa medición |
| Build Release, incremento del árbol Lean | Aproximadamente 400 MiB; árbol final 690 MiB | 172.02 s de pared; MSBuild informó 171.03 s |
| Launcher/bin/Release, incluido en el árbol final | 89 MiB | Incluido en el build |
| Caché de paquetes W/nuget, aparte del árbol Lean | 581 MiB | Restauración incluida en el build |
| Entorno Python W/py311 final | 118 MiB | 3.86 s, creación + instalación inicial; ajuste posterior de numpy fuera de esa medición |
| Primer arranque fallido de SMA | Resultados y trazas en W/sma-attempt-1-results | 11.55 s |
| Segundo arranque fallido de SMA | Resultados y trazas en W/sma-attempt-2-results | 8.59 s |
| Tercer arranque fallido de SMA | Resultados y trazas en W/results/SmaCrossSpike | 8.12 s |
| Backtest completo | No disponible | No disponible |

Los tiempos de arranque incluyen `strace`, no son tiempos de ejecución de una estrategia. Se ejecutó `df -h /` antes de las descargas/build y después de cada paso grande: 50G inicialmente, luego 49G y 48G. Nunca bajó de 30G. Final: **48G libres**, 50,607,628,288 bytes en la última medición previa a documentar.

Python base disponible: 3.11.14 de uv. Se creó W/py311 con pandas 2.3.3, numpy y pythonnet. LEAN carga su ensamblado `Python.Runtime` 2.0.66, provisto por el paquete NuGet `QuantConnect.pythonnet`; el paquete Python pythonnet 3.2.1 también está instalado, pero no demuestra compatibilidad con ese ensamblado. En el tercer intento se fijó numpy 1.26.4, versión de DockerfileLeanFoundation de la revisión descargada. Las dependencias exactas iniciales están en W/python-install.log y el ajuste en W/python-pin.log. `PYTHONNET_PYDLL` apunta a la libpython3.11.so del Python base disponible; no se descargó otro Python.

## Bloqueo y límite de intentos

1. Runner local con `python-venv` y `PYTHONNET_PYDLL`: falla en `PandasConverter`. Además aparece `Could not find platform dependent libraries <exec_prefix>`.
2. Se fija `PYTHONHOME` al `sys.base_prefix` del Python 3.11: desaparece ese aviso, persiste el mismo fallo.
3. Se fija numpy 1.26.4 según la fuente y se apunta `PYTHONPATH` a site-packages de W/py311: persiste el fallo. Se termina el spike sin un cuarto intento ni ejecutar RSI, que depende del mismo inicializador.

Error literal recortado del tercer intento:

```text
Loader.TryCreatePythonAlgorithm(): System.Exception: AlgorithmPythonWrapper():
The type initializer for 'QuantConnect.Python.PandasConverter' threw an exception.
in AlgorithmPythonWrapper.cs:line 123
```

Al finalizar los tres intentos también ocurre:

```text
Unhandled exception. System.InvalidOperationException:
GIL must always be released, and it must be released from the same thread that acquired it.
   at Python.Runtime.Py.GILState.Finalize()
   at System.GC.RunFinalizers()
```

El proceso sale con 134. LEAN escribe un JSON igualmente: `state.Status = RuntimeError`, `state.OrderCount = 0`, `totalPerformance.closedTrades` vacío. El log no expone la excepción interna de la importación de pandas; no se identificó su causa raíz y no se afirma que numpy o pythonnet sean por sí solos el problema.

## Cambios del spike y validación

Se agregó `run_lean_local.sh`, que genera config en W, activa el entorno Python para el Launcher y registra tiempo/red. Se completaron los handlers locales en config.json. Las entradas y salidas por regla de lean_algos.py solicitan MarketOnOpen para el supuesto next-open; no se agregó un modelo de llenado. Este cambio no llegó a ejecutarse dentro del algoritmo por el bloqueo.

El parser original exportaba un CSV vacío del JSON fallido. Se corrigió para rechazar `state.RuntimeError` con salida 1; el CSV vacío fue eliminado de W. Se comprobó esa corrección con la salida real de LEAN. No se validó la conversión de fills, los tags ni la zona horaria contra trades reales.

Validación realizada: los cuatro controles; comparaciones entre controles; build Release; tres arranques instrumentados; rechazo del parser sobre el JSON fallido; sintaxis de Python mediante ast.parse, JSON mediante json.loads y shell mediante `bash -n scripts/lean_spike/run_lean_local.sh`. No se tocaron motor, CLI ni tests.

Queda sin verificar: lectura de precios decimales/factor files/map files por LEAN; ejecución de SMA y RSI; restricciones cash/margen y leverage 1; tamaño entero y rechazos por buying power; activación intrabar de stops y OCO; paridad por fechas/lado; PnL; ausencia de solicitudes de red durante un backtest completo. En los arranques fallidos, las trazas de red no contienen AF_INET ni AF_INET6: no hubo solicitudes IP observadas. Eso no prueba el comportamiento posterior a la inicialización.

## Handoff al PM

`bd show QuantAgent-ryaw` falló con `sqlite3: unable to open database file`; no se modificó .beads porque está fuera del alcance autorizado. El PM debe registrar el resultado negativo y decidir el estado del issue. No hubo commits ni integración. Propuesta de commit del PM: `docs(spike): record source-built LEAN Python initialization blocker`, con los seis archivos del spike/reporte listados en el README. El runner queda como reproducción del bloqueo, no como comando validado para completar backtests. La revisión e integración quedan en el gate humano.
