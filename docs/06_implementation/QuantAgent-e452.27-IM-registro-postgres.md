# Operación del registro en Postgres

El orden de puesta en marcha es: generar un backup de la base origen; crear una base propia llamada `quantagent_registro`; apuntar `DATABASE_URL` de Alembic a esa base y ejecutar `alembic upgrade head`; correr `registro importar --dry-run` y mostrar el resultado a Fede; esperar su OK; recién entonces ejecutar el import real. La migración y el import real los coordina el hilo del registro y requieren OK de Fede por paso. Nada de esto se ejecutó como parte de esta entrega.

Para la web, crear el rol con `psql -v ro_password='...' -d quantagent_registro -f deploy/web/registro/crear_usuario_lectura.sql`; el SQL da CONNECT, USAGE y SELECT solo sobre las cuatro tablas del registro. Pasar `QUANTAGENT_REGISTRO_URL` al servicio `quantagent-api` mediante un override de systemd con `EnvironmentFile` guardado fuera del repositorio. Si falta la variable, la API responde 503.

El backup usa `QUANTAGENT_REGISTRO_ADMIN_URL`; el directorio y la retención se configuran con `QUANTAGENT_REGISTRO_BACKUP_DIR` y `QUANTAGENT_REGISTRO_BACKUP_KEEP`. El timer y el servicio son ejemplos para instalar a mano. Para restaurar y contar filas, ejecutar `probar_restauracion.sh`; requiere permisos para crear y borrar una base temporal.
