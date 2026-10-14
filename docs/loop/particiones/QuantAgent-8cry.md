# Partición de QuantAgent-8cry (V31b)

Estimación previa: unas 130 líneas entre append/encabezado/metadatos (~30),
consulta y registro del comando (~30), y tests reales de persistencia y filtros
(~70). Supera 100 líneas. No se escribió código ni se modificaron tests.

| Hijo | Objetivo | Estimación con tests | Aceptación binaria | Revisión |
| --- | --- | --- | --- | --- |
| QuantAgent-8cry.1 | Registrar cada combinación terminada en modo append | 80 líneas | Dos sweeps reales 2x2 dejan 8 filas; prefijo intacto; métricas y params iguales al CSV; grilla inválida no crea ni modifica registro | probar |
| QuantAgent-8cry.2 | Consultar corridas y combinaciones distintas | 70 líneas | 8 corridas / 4 combinaciones; ambos filtros funcionan; consulta no cambia bytes; comando en help | probar |

Orden: .1 → .2, con dependencia en Beads y filas L7 en
[la cola](../../../PLAN-CONTINUACION.md). V32 mantiene su dependencia del padre;
aprobar la partición no habilita walkforward. El padre queda in_progress.

Ambos hijos limitan cambios a sweep.py y sus tests. El segundo incluye además
quantagent/cli/backtest.py exclusivamente para importar y registrar el comando:
hoy backtest_group registra sweep allí, no en sweep.py. Esta ampliación mínima
queda explícita para revisión del PM antes de implementar.

Se conservan las columnas del padre, JSON ordenado y repetición de corridas.
No se registran run ni compare, ni se toca el registro de reserva o el motor.
Los tests del registro ejecutan el motor con snapshot sintético, sin mockear
correr_par. Cada hijo corre `python -m pytest -q tests/test_backtest_sweep.py`;
la suite completa corre en CI según el último comentario del padre.

Probar la partición: `test -f docs/loop/particiones/QuantAgent-8cry.md && bd show QuantAgent-8cry.1 && bd show QuantAgent-8cry.2`.
