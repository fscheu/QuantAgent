# Operación del registro en Postgres

El orden de puesta en marcha es: generar un backup de la base origen; crear una base propia llamada `quantagent_registro`; apuntar `DATABASE_URL` de Alembic a esa base y ejecutar `alembic upgrade head`; correr `registro importar --dry-run` y mostrar el resultado a Fede; esperar su OK; recién entonces ejecutar el import real. La migración y el import real los coordina el hilo del registro y requieren OK de Fede por paso. Nada de esto se ejecutó como parte de esta entrega.

Para la web, crear el rol con `psql -v ro_password='...' -d quantagent_registro -f deploy/web/registro/crear_usuario_lectura.sql`; el SQL da CONNECT, USAGE y SELECT solo sobre las cuatro tablas del registro. Pasar `QUANTAGENT_REGISTRO_URL` al servicio `quantagent-api` mediante un override de systemd con `EnvironmentFile` guardado fuera del repositorio. Si falta la variable, la API responde 503.

## Postgres de la VM y backup

El Postgres de la VM corre en Docker: contenedor `quantagent-dev-db`, imagen `postgres:15` (servidor 15.16), puerto 5432 publicado en el host, política `unless-stopped`, datos en un volumen anónimo de Docker (`/var/lib/postgresql/data`). Contiene las bases `quantagent`, `quantagent_dev`, `quantagent_test` y `quantagent_loop_test`; no existe el usuario `postgres` del sistema ni `pg_dump` en el host. Como los datos están en un volumen anónimo, el backup es obligatorio antes de depender de la base.

El backup se hace dentro del contenedor, así no hace falta instalar `postgresql-client` y el cliente siempre coincide con la versión del servidor:

```
QUANTAGENT_REGISTRO_CONTAINER=quantagent-dev-db QUANTAGENT_REGISTRO_DB=quantagent \
  QUANTAGENT_REGISTRO_BACKUP_DIR=$HOME/backups/registro deploy/web/registro/backup_registro.sh
```

Para el primer backup (antes de crear `quantagent_registro`) apuntar `QUANTAGENT_REGISTRO_DB` a la base de origen (`quantagent`). El directorio y la retención se configuran con `QUANTAGENT_REGISTRO_BACKUP_DIR` y `QUANTAGENT_REGISTRO_BACKUP_KEEP`; el usuario con `QUANTAGENT_REGISTRO_PGUSER` (por defecto `postgres`). `probar_restauracion.sh` restaura el último dump en una base temporal dentro del contenedor, cuenta las filas de las cuatro tablas y borra la base temporal. Sin la variable del contenedor, ambos scripts caen al modo host (`QUANTAGENT_REGISTRO_ADMIN_URL` y `pg_dump` del host). El timer y el servicio son ejemplos para instalar a mano (agregar `QUANTAGENT_REGISTRO_CONTAINER` al `EnvironmentFile`).

## Volumen y puerto (actualizado 2026-11-04, tras recrear el contenedor)

- **Contenedor:** `quantagent-dev-db`, imagen `postgres:15` (servidor 15.16), `restart: unless-stopped`. Los datos están en el volumen con nombre `quantagent-pgdata` (montado en `/var/lib/postgresql/data`). El contenedor anterior quedó parado como `quantagent-dev-db-old` con su volumen anónimo: no se borra hasta que se confirme que no hace falta.
- **Puerto:** publicado solo en loopback (`127.0.0.1:5432`). Responde por `localhost` (la web, los comandos locales y los tests siguen funcionando con `DATABASE_URL=...@localhost:5432/...`) y ya no responde por la IP privada de la VM ni por las demás interfaces.
- **Verificación hecha en solo lectura:** las cuatro bases (`quantagent`, `quantagent_dev`, `quantagent_test`, `quantagent_loop_test`) existen con las mismas tablas y el mismo número de filas que el `pg_dumpall` del 2026-11-04 20:33 (12, 12, 11 y 18 tablas; 3.979, 630, 0 y 4.690 filas, sin diferencias por tabla).
- **Backup:** el procedimiento no cambia (`docker exec quantagent-dev-db pg_dump ...`, ver arriba). Un volumen con nombre sobrevive a `docker rm` del contenedor, pero no a `docker volume rm`/`docker compose down -v`, así que el backup sigue siendo obligatorio antes del import real.
- Pendiente para Fede: revisar el NSG de Azure por si el puerto 5432 estuviera abierto en otras reglas (ya no hace falta para esta base).
