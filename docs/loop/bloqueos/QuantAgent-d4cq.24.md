# QuantAgent-d4cq.24: escritura real fuera del aislamiento

E09 requiere ejecutar las dos hipótesis sobre los registros reales de `local-data`.
El comentario del ticket del 2026-10-16 16:36 autoriza abrir la prueba y exige
contar las filas nuevas de esos archivos, sin carpeta aislada para los registros.

Se revisaron el ticket, el entorno y las rutas de escritura, sin ejecutar el embudo.
`QUANTAGENT_SNAPSHOT_DIR` apunta a
`/home/azureuser/repos/projects/QuantAgent/local-data`.
`registrar_estado`, `registrar_intento` y `abrir` escriben bajo `snapshot_root()`.
`--carpeta` es obligatorio y solo ubica los resultados del motor; no reubica esos registros.

El aislamiento permite escribir en este worktree, `/tmp` y el directorio compartido
`.beads`, pero no en el `local-data` real. Tampoco permite solicitar una ampliación.
Además, el prompt prohíbe modificar archivos fuera de este worktree.
Por eso no se intentó una escritura que pudiera dejar una corrida parcial.
No se abrió la prueba ni se produjeron métricas o conteos de una corrida nueva.

Destraba: relanzar E09 en un entorno autorizado para escribir en ese `local-data`,
manteniendo el OK de Fede y usando `--carpeta` para los CSV del motor.
Una copia en `/tmp` no satisface el comentario vigente del ticket.

Comprobación de la ruta (solo lectura):
`printenv QUANTAGENT_SNAPSHOT_DIR`
Esperado: `/home/azureuser/repos/projects/QuantAgent/local-data`.
