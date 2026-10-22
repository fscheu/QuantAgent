# Bloqueo de QuantAgent-v2ll

No se puede producir la atribución S04 en este worktree: `local-data/` contiene únicamente `README.md`; no están las curvas guardadas de las cuatro carteras.

La descripción exige usar esas curvas, no abrir ventanas nuevas y comprobar que los md5 de `intentos.csv`, `hipotesis.csv` e `hipotesis-candado.csv` no cambien. Sin las curvas y los registros no se puede calcular ni verificar la tabla.

Para destrabar: ejecutar el ticket en la VM indicada por Beads, con las curvas y registros existentes en solo lectura, y registrar sus hashes antes y después. No generar curvas ni modificar `local-data`.
