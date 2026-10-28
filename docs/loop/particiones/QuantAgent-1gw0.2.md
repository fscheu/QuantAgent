# QuantAgent-1gw0.2 — partición de I02 (L31)

El ticket reúne dos comandos nuevos, registro en el CLI raíz, selección por fechas,
reescritura compatible del CSV, backup y cambios al informe, con cinco criterios
que requieren pruebas. Estimación total: 250–360 líneas con pruebas; supera 100.
Se entrega solo esta partición, sin cambiar código ni aplicar comandos a datos reales.

| Orden | Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
| --- | --- | --- | --- | --- | --- |
| 1 | QuantAgent-1gw0.2.1 | Grupo raíz y dry-run | Selección por hipótesis/fecha UTC inclusiva; errores claros; mismos bytes y sin backup | ≤100 líneas | probar |
| 2 | QuantAgent-1gw0.2.2 | Aplicar invalidación | Backup idéntico; conserva filas/orden/valores salvo estado seleccionado; admite encabezado sin estado | ≤100 líneas | probar |
| 3 | QuantAgent-1gw0.2.3 | Marca de época limpia | JSON con desde_utc/motivo/creada_utc; CSV intacto; fecha inválida no escribe | ≤70 líneas | probar |
| 4 | QuantAgent-1gw0.2.4 | Época y conteos en informe | Fixture muestra época y conteos globales válidos/invalidados; sin marca/estado funciona; HTML escapado | ≤90 líneas | probar |

Dependencias en Beads: I02a → I02b → I02c → I02d. I01 ya está cerrado.
La cola de §4 reemplaza el padre por estos cuatro hijos. Las descripciones completas,
archivos y tests dirigidos están en Beads; cada estimación incluye pruebas.
El padre permanece en curso hasta completar los hijos. Esta partición no lo cierra.

No cambia el alcance original: no se tocan hipotesis.csv ni DSR, no se invalidan
corridas reales y no se elige la fecha de época. Fede confirma esas decisiones.
