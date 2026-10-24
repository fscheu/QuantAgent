# QuantAgent-7v45: referencia transversal incompatible con T4

Detectado por PM al revisar `QuantAgent-av1p.4.8` (L21), 2026-10-24.

## Evidencia

- [T4](../../04_decisions/2026-10-17-DC-piernas-transversales.md#t4-pesos-q1-decidida-por-fede) decidió 1/σ **entre elegidos**, sumando 1/S por pierna; rechazó ponderar todo el universo y dejar efectivo.
- `fabrica/cartera.py:curva_cartera` y D1 (`pesos_objetivo` por defecto) no pasan selección. La curva agregada de etapas 1/2/3 usa ese camino; `filtrar_pierna` sí normaliza entre elegidos para su evaluación interna.
- [Reproductor sintético](../diagnosticos/QuantAgent-7v45_pesos.py): A/B/C, solo A elegida, +10 %. Desde 100000: curva actual/D1 × estado = **103333.33**; T4 = **110000.00**.
- Con cantidades de activos de la YAML real y volatilidades iguales: `0.5*9/29 + 0.5*5/17 = 30.223124%` invertido, frente a 100 % según T4. No es redondeo.
- No se detectó una contradicción equivalente para núcleo+curva. No se propone ponderar por riqueza acumulada de piernas: el motor combina retornos diarios.

## Reproducir

`scripts/loop/try.sh lote/paper-alpaca -- 'PYTHONPATH=. python docs/loop/diagnosticos/QuantAgent-7v45_pesos.py'`

Solo usa datos sintéticos y funciones reales; no ejecuta backtests ni lee snapshots/registro de intentos. La salida y asserts describen la discrepancia previa a la corrección.

## Bloqueo y alcance propuesto

`av1p.4.8` depende de `7v45`; completar las dos carteras exige resolver la referencia. `av1p.4` excluye cambiar pesos/estrategias y el PM debe escalar cambios en curvas de referencia (plan §3.7).

Recomendación: mantener T4 y corregir el flujo de selección de la curva agregada y la sombra/objetivo paper, con pruebas sintéticas y evidencia nueva del impacto; conservar resultados históricos. Partir la corrección antes de código si supera 100 líneas. No se reabre A/B ni se autoriza reserva 2023+.

Fede recibió la consulta de alcance. Hasta su respuesta, no se toman nuevas implementaciones; la entrega independiente `av1p.4.2` (PR #466) puede terminar su CI y revisión. No cambian pesos, backtest, cuentas ni operación remota.

## Adenda 2026-10-24: alcance resuelto por Fede

Fede autorizó: «si. inclui la corrección sin tocar datos hacia atras». Se respeta T4 para próximas ejecuciones y consultas, con pruebas sintéticas; no se leen, recalculan ni reescriben datos/resultados reales anteriores, metadatos, informes previos, candados o intentos, ni se abre reserva 2023+. El diagnóstico anterior queda conservado como hallazgo previo a la corrección; no se cuantificó cambio real de SR y el SR individual por activo no se modifica por esta normalización.

La [partición aprobada de siete hijos](../particiones/QuantAgent-7v45.md) transporta selección hasta curvas, objetivos/sombra y pesos del diagnóstico `benchmark_mismo_peso`, incluyendo informes y criterio consultados sobre carpetas futuras. Reutiliza `pesos`, conserva 1/N y tasas; corrige además el caso demostrado de benchmark con suma de pesos cero para conservar efectivo. No exige que todo SR sintético cambie: prueba curva y pesos T4 correctos.

La decisión de alcance deja de bloquear la selección de trabajo. El padre `QuantAgent-7v45` sigue abierto hasta integrar los siete hijos y continúa bloqueando `av1p.4.8`. Primero se integra la entrega independiente `av1p.4.2` (PR #466); después sigue la corrección, antes de retomar `.3` de D4. Esta autorización no habilita piloto, cuentas, secretos ni operación remota.
