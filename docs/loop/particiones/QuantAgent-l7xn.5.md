# QuantAgent-l7xn.5 — partición QA-05

Estimación: 305 líneas, incluyendo tests. El contrato §3.2 exige partir
antes de implementar cuando se estiman más de 100 líneas.
Esta entrega solo cambia planificación y Beads.

La unit actual usa código e intérprete de paper-ops y maestro de web-ops.
`/health` solo devuelve status; Account.database_url puede apuntar a bases
externas: copiar maestro sin redirigir cuentas no aísla el preview.

Hay una decisión pendiente: el ticket pide copiar datos productivos y a la
vez prohíbe leer ambos directorios originales. QA-05a debe fijar si recibe
snapshots preparados fuera de ellos o se autoriza lectura única para backup,
sin escritura. Hasta esa decisión, QA-05b queda bloqueado.

| Hijo | Objetivo y aceptación binaria | Revisión | Líneas estimadas |
|---|---|---|---|
| QuantAgent-l7xn.5.1 | PM registra origen, rutas y permiso de lectura; sin acceso a producción | decidir | 25 |
| QuantAgent-l7xn.5.2 | Backup SQLite de maestro y carteras; referencias apuntan a copias; escribir copia no altera origen; sin secretos | probar | 90 |
| QuantAgent-l7xn.5.3 | Rama o PR produce worktree con npm ci/build y datos aislados; health identifica rama configurada | probar | 95 |
| QuantAgent-l7xn.5.4 | Unit quantagent-api-dev sirve en 127.0.0.1:8012; stop apaga unit y elimina worktree; sin timers ni claves | probar | 95 |

Orden obligatorio: .1 → .2 → .3 → .4. Cada estimación incluye sus tests;
si supera 100 al implementar, se vuelve a partir. Cada hijo mantiene smoke
y tests dirigidos verdes. .2 prueba SQLite real, sin reemplazar la lógica
de copia por mocks. .3 conserva health normal cuando no hay preview.

El criterio completo de QA-05 se comprueba al terminar .4: rama y PR,
curl health, consultas de cuentas contra copias, originales intactos,
stop con unit inactiva, puerto cerrado y worktree eliminado.
Esa prueba requiere ejecución del PM con permisos para systemd; este
implementador no modifica servicios ni directorios fuera del worktree.

QA-06 conserva su dependencia del padre y agrega dependencia directa de .4:
cerrar el padre al aprobar esta partición no habilita el tunnel antes de tiempo.
QA-07 conserva su alcance. No se modifica install.sh ni la unit productiva.

## Decisión QA-05a (PM, 2026-11-02)

Origen de los datos del preview: lectura única con `sqlite3 .backup` de
`/home/azureuser/web-ops/maestro.db` y de las bases que referencia
`Account.database_url` (hoy `/home/azureuser/paper-ops/local-data/*.db`).
`.backup` solo lee el origen: no escribe, no cambia el modo de la base y no
exige parar la API. Fede autorizó el 2026-11-02 el preview con copias de las
bases y sin tocar producción.

Reglas para QA-05b a QA-05d:

- El código nunca abre un original en modo escritura ni lo usa como base del
  preview. Solo corre `.backup` hacia un directorio propio del preview.
- Los tests de los implementadores usan bases SQLite creadas en un directorio
  temporal. El sandbox del implementador no lee ni `~/paper-ops` ni `~/web-ops`.
- La copia real la ejecuta el PM al probar la entrega, no el implementador.
- Las referencias `database_url` se reescriben solo en el maestro copiado.
- No se copian `.env` ni credenciales Alpaca.
