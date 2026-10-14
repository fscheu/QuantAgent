# QuantAgent-ryaw: spike de LEAN compilado desde fuente

Fecha: 2026-10-13. Segunda corrida: backtests completos, paridad insuficiente para aprobar P1.
Contexto: [decisión del pivot, §§7–10](./2026-10-12-DC-quantconnect-y-pivot-lean.md).
Reproducción: [README del spike](../../scripts/lean_spike/README.md).

| Criterio binario | Resultado | Evidencia |
|---|---|---|
| (a) SMA 5/5 y RSI 187/187, o cada diferencia de RSI explicada | **No** | SMA 3/5; dos entradas rechazadas por poder de compra. RSI 140/187; los 47 desajustes están clasificados individualmente. |
| (b) Completa el backtest sin cuenta de QuantConnect | **Sí, con limitación de cierre** | Ambos JSON tienen Status Completed y RuntimeError vacío, hasta 2018-12-31, sin cuenta/token/login. Cero llamadas AF_INET/AF_INET6 en strace de cada ejecución completa. El proceso aborta después de guardar los resultados, en Shutdown de Python (salida 134). |
| (c) Disco y tiempos, incluida una corrida completa | **Sí** | Se conservan instalación/build; SMA 17.74 s y RSI 24.03 s con strace y cierre incluido. Al finalizar, / tiene 49G libres; nunca bajó de 30G. |

El bloqueo de inicialización quedó resuelto. Los resultados permiten medir el comportamiento por defecto de LEAN, pero SMA no pasa la puerta 5/5. No se recomienda aprobar P1 con estos datos. La excepción de cierre del proceso sigue pendiente; Completed describe el backtest, no una salida limpia del Launcher.

## Alcance y resultados

W = `/home/azureuser/spike-lean-work`. Se reutilizaron SDK, build Release, Python, datos y controles de la primera corrida. No se usaron git, sudo, Docker, LEAN CLI ni cuentas. No hubo rebuild ni descargas. Los scripts del proyecto corrieron con `/home/azureuser/repos/projects/QuantAgent/.venv`, PYTHONPATH en este worktree y SQLite W/try.db; LEAN usa W/py311.

SPY, snapshot `etf-1d-2026-10`, 2007-01-03..2018-12-31, 100% del capital, perfil etf: slippage 0.0005 por lado y comisión 0. No se leyó ningún precio posterior al límite. Se reutilizaron 3020 velas; LEAN informa 3021 puntos procesados (no equivale a 3021 velas de precios). SMA: cash; RSI: margen y leverage 1. BrokerageName.DEFAULT, EquityFillModel por defecto, sin modelo de llenado propio. Alpaca, usado inicialmente por el port, rechazaba MarketOnOpen al cierre por su ventana horaria 07:00–09:28; se reemplazó por el brokerage predeterminado.

Las coincidencias exigen misma sesión de Nueva York de entrada y salida, y mismo lado. No exigen cantidades ni precios iguales. El adaptador next-open es el control principal; el motor llena al cierre.

| Estrategia | Control | Trades LEAN | Coincidencias N de M | Desvío relativo del PnL total |
|---|---|---:|---:|---:|
| sma-cross | bt_adapter next-open | 3 | 3 de 5 | -15.1746% |
| sma-cross | motor propio | 3 | 0 de 5 | -18.4163% |
| rsi | bt_adapter next-open | 176 | 140 de 187 | +69.0487% |
| rsi | motor propio | 176 | 12 de 187 | +47.1116% |

Desvío = (PnL LEAN − PnL control) / abs(PnL control). Los porcentajes positivos de RSI indican una pérdida menor, no rentabilidad positiva. Las cantidades difieren en los 3 trades coincidentes de SMA y los 140 de RSI. Además del truncamiento a acciones enteras, cambian el capital disponible y los precios de ejecución: el fill model de LEAN redondea los niveles a centavos, no aplica slippage a LimitFill y calcula el slippage de stops con el precio del security. La paridad temporal no implica paridad de PnL.

Comparaciones guardadas en W/compare_{adapter,motor}_{sma-cross,rsi}.log; CSV de LEAN en W/lean_{sma-cross,rsi}.csv. No se regeneraron controles.

## Diferencias por causa contra next-open

Se cuenta cada trade del control que no coincide una sola vez, por causa principal. El desglose individual con fechas, lado y evidencia está en W/differences_sma-cross.csv y W/differences_rsi.csv; el script de auditoría y su resumen están en W/classify_differences.py y W/classification-summary.txt. No son un modelo alternativo de llenado.

| Causa principal | SMA | RSI |
|---|---:|---:|
| Redondeo a acciones enteras: señal menor que una acción | 0 | 1 |
| Stop/take activo desde la vela siguiente a la entrada, incluidos efectos en reentradas | 0 | 41 |
| Entrada rechazada por poder de compra | 2 | 2 |
| Otra: stop redondeado a centavos, incluida reentrada desplazada | 0 | 2 |
| Otra: posición abierta al terminar, sin cierre forzado | 0 | 1 |
| **Total de trades del control sin coincidencia** | **2** | **47** |

SMA: entradas de 2009-06-19 y 2015-12-10 rechazadas (órdenes 1 y 6). El port dimensiona al cierre de señal; el gap de apertura supera el efectivo. El adaptador recorta la compra al efectivo disponible usando la apertura de ejecución. No se introdujo ese recorte en el fill model de LEAN.

RSI: 32 trades comparten entrada pero difieren en salida; faltan 15 entradas del control y sobran 4 entradas candidatas. De los 41 atribuidos a activación diferida, 31 comparten entrada y 10 son reentradas alteradas por la trayectoria previa. EquityFillModel.GetBestEffortTradeBar descarta barras cuyo EndTime <= order.Time. Como las órdenes de protección se envían al recibir el fill diario de entrada, no actúan sobre esa misma barra; el adaptador sí puede cerrar en ella. Las cuatro entradas extra corresponden a esas trayectorias (tres) y al stop redondeado (una).

La señal del 2014-06-04 da menos de una acción: el port la omite y entra después, desplazando el trade del control del 2014-06-05. Para el corto del 2014-02-25, el máximo del 2014-03-06 alcanza el stop original pero no el redondeado de la orden 377: salida al día siguiente y reentrada desplazada del 2014-03-07 al 2014-03-10. Esos son los dos trades de la categoría centavos.

Las órdenes de entrada 531 y 552 son Invalid por buying power, con el cierre MarketOnOpen de la posición anterior todavía pendiente: faltan los trades del control del 2018-10-12 y 2018-12-26. El del 2018-12-27 sí entra (orden 553), pero queda abierto con protecciones 554/555: no aparece en closedTrades, a diferencia de finalize_trades del adaptador. Esta categoría separa una posición abierta de una entrada ausente.

## Python, selección del algoritmo y datos aceptados

Primer intento de esta segunda corrida: `uv pip uninstall --python W/py311/bin/python pythonnet clr-loader` eliminó pythonnet 3.2.1 y clr-loader 0.3.1. No se reemplazó el ensamblado Python.Runtime 2.0.66 de QuantConnect.pythonnet. PandasConverter inicializó inmediatamente; no hizo falta un segundo arreglo de Python ni especular con otra versión de numpy. Se conservaron Python 3.11.14, pandas 2.3.3, numpy 1.26.4, PYTHONHOME al base_prefix, PYTHONNET_PYDLL a libpython3.11 y PYTHONPATH a site-packages. El clr provisto por LEAN dejó de quedar eclipsado por el módulo de pip.

El siguiente fallo literal fue `'SpikeBase' object has no attribute 'evaluar_entrada'`. AlgorithmPythonWrapper enumera e instancia todas las clases QCAlgorithm definidas en el módulo, sin seleccionar por algorithm-type-name. El runner ahora exporta LEAN_SPIKE_ALGORITHM y el módulo elimina de sus atributos la clase base y la estrategia no seleccionada antes de que el cargador lo inspeccione.

El formato original se leyó sin cambiar datos: zip equity/usa/daily/spy.zip con spy.csv sin encabezado, fecha YYYYMMDD 00:00 y OHLC multiplicados por 10000 con cuatro decimales; volumen entero. Factor file con factor de precio/split 1 y map file sin cambio de ticker, ambos con extremos 19980102 y 20501231. Esas fechas son metadatos de identidad, no precios leídos después del límite. Normalización RAW. No hizo falta --enteros. write_lean_data.py solo actualiza su descripción de validación.

El parser pasó de symbol a symbols[0] y usa orderIds para obtener los tags de entrada/salida: los orders reales no traen lastFillTime. Los fills se exportan con Z (UTC); se convierten a sesión America/New_York y luego a medianoche, como los controles. direction 0 es buy y 1 sell. compare_trades.py funcionó sin cambios.

## Instalación y mediciones conservadas

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
| SMA completo, segunda corrida, con strace | W/results/SmaCrossSpike | 17.74 s de pared; algoritmo 2.13 s |
| RSI completo, segunda corrida, con strace | W/results/RsiSpike | 24.03 s de pared; algoritmo 6.07 s |

Los tiempos de pared finales incluyen strace y el aborto de cierre; no son una medición sin instrumentación. Hubo un breve solapamiento entre el cierre del Launcher SMA y el arranque del Launcher RSI; los comandos permanecieron en primer plano. Las mediciones no aíslan rendimiento en los dos núcleos compartidos. No se repitieron instalación/build. La primera corrida terminó con 48G libres; durante esta segunda se observaron 48G, 52G y al final 49G (51,586,236,416 bytes disponibles). No se atribuye el cambio de espacio a este spike. W/py311 mide ahora 129 MiB y W/results 6.2 MiB; los tamaños anteriores se conservan como mediciones históricas.

## Bloqueo de la primera corrida (historia)

Los tres intentos anteriores fallaron antes del algoritmo en `The type initializer for 'QuantConnect.Python.PandasConverter' threw an exception.` Primero se configuró el venv/libpython; luego PYTHONHOME; finalmente numpy 1.26.4 y PYTHONPATH. Ninguno expuso la excepción interna. Sus tiempos y cero llamadas IP corresponden a arranques fallidos, no a backtests. La pista posterior del PM identificó el import clr de pip que intentaba crear su propio runtime mono. La desinstalación de ambos paquetes resolvió el inicializador en esta corrida.

## Costos de implementación y límites

Lo que costó trabajo fue separar inicialización, selección de clase, rechazos del brokerage y resultados reales: el JSON existía incluso tras fallos. Se corrigió el port y el parser con los logs y campos reales, y se investigaron las diferencias con órdenes, eventos, OHLC exportado y EquityFillModel. No se corrigió LEAN upstream ni se ajustó un fill model para forzar paridad.

Después de guardar Completed, ambos procesos abortan con el mismo error literal:

```text
PythonInitializer.Shutdown(): calling engine shutdown...
Unhandled exception. System.InvalidOperationException: GIL must always be released, and it must be released from the same thread that acquired it.
   at Python.Runtime.Py.GILState.Finalize()
   at System.GC.RunFinalizers()
```

PythonInitializer.Shutdown del fuente conserva un Py.GIL local al llamar PythonEngine.Shutdown. No se modificó ni recompiló ese código. El runner conserva salida 134; no oculta el aborto ni lo transforma en éxito.

No se verificaron: salida limpia del proceso; capacidad de producción/HFT; datos sin ajustar, splits/dividendos reales, cambios de ticker, otros símbolos, benchmarks, ejecución live, slippage intradiario real, intereses/costos de préstamo ni equivalencia exacta de precios/PnL. Falta alternative/interest-rate/usa/interest-rate.csv: RSI registra una solicitud local fallida y ambos logs advierten que no se cargaron tasas; no se descargaron datos adicionales. La evidencia de ausencia de red corresponde a estas dos ejecuciones concretas, desde el arranque hasta el aborto de cierre.

Validación: dos resultados Completed con RuntimeError vacío, conversión de closedTrades, cuatro compare_trades (salida 1 por diferencias), auditoría de los 2 y 47 desajustes, cero AF_INET/AF_INET6 en las trazas finales, sintaxis Python/JSON/shell y control de disco. CSV y logs quedan en W; no se publican precios ni filas de trades aquí.

## Handoff al PM

La integración sigue pendiente del gate humano. Beads no se modificó porque queda fuera del alcance autorizado; en la primera corrida bd no pudo abrir la base. El PM debe actualizar QuantAgent-ryaw con el resultado: inicialización resuelta, backtests completos, criterio (a) no cumplido y cierre del proceso pendiente. No hubo commits. Plan de commit: `fix(spike): run local LEAN and record measured trade differences`, con los seis archivos tocados en esta corrida; validación reproducible en el README. El reporte anterior se reescribe por instrucción explícita del PM/usuario, conservando su historia y mediciones.

## Decisión de Fede (2026-10-13)

Las diferencias contra el motor propio son diferencias de criterio de ejecución, no un bloqueo. El desarrollo sigue como estaba planeado, con el motor propio, y el pivot a LEAN se decide en M3. Antes de esa decisión falta verificar lo que este spike no probó: fracciones de acción y el plugin de Alpaca en papel.

Diagramas de componentes (arquitectura actual y propuesta con LEAN), detalle de las piezas que se reemplazarían y balance del PM: [QuantAgent-ryaw-arquitectura/index.html](./QuantAgent-ryaw-arquitectura/index.html). Abrir el archivo en un navegador; GitHub muestra el HTML como código.

## Criterios para la decisión de M3 (PM, 2026-10-14)

Además del resultado (sma-cross 3 de 5, rsi 140 de 187) y de los tiempos medidos, la decisión de M3 mira estas aristas. Las planteó Fede al ver el tamaño de la solución.

Tamaño medido en la VM: LEAN son unas 756 mil líneas de C# en 4.405 archivos (690 MB compilado), más el SDK de .NET (624 MB) y un entorno Python 3.11 propio (129 MB). El código Python de `quantagent/` son unas 12 mil líneas.

1. **Código que no sabemos qué hace.** La mayor parte no corre en un backtest diario (opciones, futuros, cripto, brokers). Se mitiga con un commit fijo, compilado propio, corrida sin red (verificada en las dos corridas de este spike) y licencia Apache 2.0. Riesgo menor.
2. **Diagnóstico.** Cuando un trade no coincide, la causa está en C#, que no es el lenguaje del proyecto ni de los agentes del loop. Cerrar la brecha medida aquí es el primer costo real.
3. **Puente con Python.** LEAN trae su propio pythonnet y fija la versión de Python. Python no inicializó hasta sacar el pythonnet de pip, y el proceso aborta al cerrar (GIL). Cada actualización de LEAN o de Python puede romperlo de nuevo.
4. **Actualizaciones.** Un cambio de LEAN en el modelo de llenado movería números de referencia sin tocar nuestro código. Hace falta decidir cada cuánto se actualiza el commit fijo y con qué control.
5. **Validación.** Los oráculos de M1 y el control con `backtesting.py` están en Python. Hay que decidir qué queda como referencia y si el motor propio sigue como control.
6. **Paper y live.** Fracciones de acción y el plugin de Alpaca en papel (pendientes desde la decisión del 2026-10-13). Es donde LEAN aporta más.
7. **Recursos de la VM.** El build y las corridas compiten con el loop y los tests en 2 núcleos.
8. **Salida.** El costo de dejar LEAN depende de que las estrategias sigan escritas contra nuestra interfaz de señales y no contra la API de LEAN. Condición del PM para cualquier pivot.

### Escenario: LEAN como segundo motor de control

Propuesta de Fede: mantener LEAN como hoy se usa `backtesting.py`, un motor aparte que corre las mismas estrategias y confirma los resultados propios, sin reemplazar el motor.

- **Fork.** Es posible (Apache 2.0). Se recomienda un fork sin cambios propios, usado solo para fijar el commit y compilar desde ahí. Un fork con parches propios obliga a mantenerlos contra cada actualización. Si hiciera falta un cambio de comportamiento, va como modelo de llenado propio dentro del algoritmo, no como parche al motor.
- **Lo que cuesta.** Un adaptador como `scripts/bt_adapter.py` (señales nuestras hacia órdenes de LEAN, escritura de datos y lectura de trades), que en buena parte ya existe en `scripts/lean_spike/`. Para que el control sirva, cada diferencia tiene que tener causa escrita, como pasó con `backtesting.py` en L5 y L6.
- **Lo que aporta.** Un segundo control con un modelo de ejecución distinto y probado en producción por terceros, sin acoplar las estrategias a LEAN. Si en M3 se decide el pivot, ese adaptador es el primer paso. Si no, queda como control.
- **Cuándo correrlo.** Por su costo en la VM, en el punto de control de cada lote y no en cada entrega.
