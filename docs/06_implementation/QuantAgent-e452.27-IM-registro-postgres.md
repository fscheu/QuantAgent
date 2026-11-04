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

## Volumen y exposición (revisado 2026-11-04, sin cambios hechos)

- **Volumen:** el contenedor usa un volumen anónimo de Docker (driver local, bajo `/var/lib/docker/volumes`, disco raíz con espacio libre). Sobrevive a `docker restart`, pero se pierde si se borra el contenedor junto con sus volúmenes (`docker rm -v`, `docker compose down -v`) y es fácil de perder de vista. Si el registro vive ahí, conviene pasar a un volumen con nombre (o un bind a un directorio conocido) reusando los datos, después de un backup verificado.
- **Puerto 5432:** Docker lo publica en todas las interfaces (IPv4 e IPv6), `ufw` está inactivo y la cadena `DOCKER-USER` está vacía. Responde por la red privada y por Tailscale. La metadata de Azure no muestra una IP pública asociada a la interfaz, así que desde internet probablemente no llega, pero no se puede confirmar desde la VM: depende del NSG de Azure. Paso sugerido para Fede: publicar solo en loopback (`-p 127.0.0.1:5432:5432`), lo que obliga a recrear el contenedor reusando el volumen, y revisar el NSG. Hacerlo después del backup. Con el puerto solo en loopback, la web (en la misma VM) y los comandos locales siguen funcionando.
