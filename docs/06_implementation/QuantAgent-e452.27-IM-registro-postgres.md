# Operación del registro en Postgres

## Dónde vive el registro

- **Producción:** contenedor `quantagent-prod-db` (postgres:15), publicado solo en `127.0.0.1:5433`, con volumen con nombre, base `quantagent_registro`. Lo crea y lo opera el hilo del registro (BNqL), con un usuario de escritura (comandos de investigación) y un usuario de solo lectura (la web), y un `pg_dump` diario. Este lote no crea ni toca ese contenedor.
- **Desarrollo y tests:** `quantagent-dev-db` (puerto 5432 solo en loopback, volumen `quantagent-pgdata`, bases `quantagent`, `quantagent_dev`, `quantagent_test`, `quantagent_loop_test`). Es solo para desarrollo y tests: **el registro real NO vive ahí.**

## Cómo lee la web el registro

La API abre las tablas `hypotheses`, `hypothesis_events`, `hypothesis_attempts` y `cycles` con `QUANTAGENT_REGISTRO_URL`, con configuración aparte de la base maestra. Debe ser el usuario de solo lectura:

```
QUANTAGENT_REGISTRO_URL=postgresql://quantagent_registro_ro:<contraseña>@127.0.0.1:5433/quantagent_registro
```

La contraseña va en un archivo fuera del repo (`~/secrets/...`) cargado con `EnvironmentFile` en un override de systemd del servicio `quantagent-api`; nunca en el repo ni en la línea de comandos. Si la variable falta o la base no responde, la API contesta 503 con un mensaje (no 500). El rol de solo lectura se crea con `deploy/web/registro/crear_usuario_lectura.sql` (SELECT solo sobre esas cuatro tablas y `default_transaction_read_only`); si el hilo del registro ya lo creó con otro nombre, usar el suyo y no duplicar. El SQL es compatible con la base nueva: se ejecuta después de `alembic upgrade head` y no requiere nada del contenedor de desarrollo.

## Orden operativo (lo coordina el hilo del registro, con OK de Fede por paso)

Backup de la base de origen -> contenedor y base propios -> `alembic upgrade head` apuntando a `quantagent_registro` -> `registro importar --dry-run` (se le muestra a Fede) -> OK de Fede -> import real (55 etapas / 202 intentos) -> usuario de solo lectura -> `QUANTAGENT_REGISTRO_URL` en el servicio. Nada de esto lo ejecuta este lote.

## Backup y restauración de prueba

Los scripts de `deploy/web/registro/` usan `docker exec` sobre el contenedor de producción por defecto (`QUANTAGENT_REGISTRO_CONTAINER=quantagent-prod-db`, base `quantagent_registro`), así que no hace falta `postgresql-client` en el host y el cliente coincide con la versión del servidor:

```
QUANTAGENT_REGISTRO_BACKUP_DIR=$HOME/backups/registro deploy/web/registro/backup_registro.sh
QUANTAGENT_REGISTRO_BACKUP_DIR=$HOME/backups/registro deploy/web/registro/probar_restauracion.sh
```

Variables: `QUANTAGENT_REGISTRO_CONTAINER`, `QUANTAGENT_REGISTRO_DB`, `QUANTAGENT_REGISTRO_PGUSER` (usuario con permiso de dump; por defecto `postgres`, ajustar al que defina el hilo del registro), `QUANTAGENT_REGISTRO_BACKUP_DIR` y `QUANTAGENT_REGISTRO_BACKUP_KEEP`. `probar_restauracion.sh` restaura el último dump en una base temporal dentro del contenedor, cuenta las filas de las cuatro tablas y borra la base temporal. Con `QUANTAGENT_REGISTRO_CONTAINER=""` los scripts caen al modo host (`QUANTAGENT_REGISTRO_ADMIN_URL` y `pg_dump` del host). El `pg_dump` diario de producción lo define el hilo del registro; el service/timer `quantagent-registro-backup.*` de este directorio es solo un ejemplo y no debe instalarse si ya existe el suyo.

## Contenedor de desarrollo (referencia)

Verificado en solo lectura el 2026-11-04 tras recrearlo: `postgres:15` (15.16), `restart: unless-stopped`, volumen con nombre `quantagent-pgdata`, puerto solo en loopback; las cuatro bases coinciden con el `pg_dumpall` del 04/11 (mismas tablas y filas). El contenedor anterior quedó parado con su volumen anónimo. Pendiente para Fede: revisar el NSG de Azure por si hubiera reglas abiertas.
