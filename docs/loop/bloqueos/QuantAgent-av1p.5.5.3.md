# QuantAgent-av1p.5.5.3: partición bloqueada por profundidad de Beads

La integración se estima en 130–150 líneas: conexión CLI (~35), formato de
cuenta/fills/pendientes (~30), pruebas nuevas y adaptación de casos existentes
con `--portfolio` (~65). Supera el objetivo de 100; no se escribió código.

Partición prevista, dos entregas de hasta 100 líneas cada una:
1. Formato en `quantagent/digest.py` y `tests/test_trading_digest.py` (~90).
   Aceptación: equity 1000, caja 50 (5 %, reserva objetivo 5 %), posición 950,
   P&L -30, ejecuciones sin ganancias inventadas y pendientes; formato anterior
   conservado sin datos nuevos; salida menor a 4000 caracteres. Revisión: probar.
2. Conexión en `quantagent/cli/reconcile.py` y tests de digest/reconcile (~95),
   dependiente del primero. Aceptación: SQLite sin ActivePosition/Trade muestra
   Position, fill diario y pendiente previo; API fallida da aviso y exit 1;
   sin `--portfolio` conserva el camino anterior; sombra/dividendos/rechazos
   conservados. Revisión: probar.

Se intentó `bd create --parent QuantAgent-av1p.5.5.3` con objetivo y aceptación
del primer hijo. Falló con:
`Error: maximum hierarchy depth (3) exceeded for parent QuantAgent-av1p.5.5.3`.
No se crearon hijos ni se modificó la cola; el ticket queda in_progress.

Decisión que destraba: autorizar los dos tickets como hermanos bajo
`QuantAgent-av1p.5.5`, con dependencia entre ellos, y sustituir la fila X19h3
en la cola. La alternativa es permitir un cuarto nivel en Beads.

Pruebas previstas para ambas entregas: `tests/test_trading_digest.py`,
`tests/test_reconcile_cli.py`, `tests/test_chequeos_paper.py` y smoke.

Verificación sobre código sin cambios: 36 passed, 11 failed por
`ModuleNotFoundError: No module named 'alpaca'` en test_reconcile_cli.
La `.venv` del checkout principal tampoco tiene ese módulo. El entorno debe
incluir la dependencia antes de verificar la implementación. Smoke: exit 0,
Trades: 9. Verificador independiente: PASS como bloqueo; funcionalidad pendiente.
