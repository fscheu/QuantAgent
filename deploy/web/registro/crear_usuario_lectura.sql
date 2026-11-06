-- Usuario de SOLO LECTURA de la web sobre el registro de producción.
-- Se ejecuta una vez, con un usuario con permiso para crear roles, en el contenedor quantagent-prod-db, DESPUÉS de que
-- `alembic upgrade head` creó las tablas en la base quantagent_registro:
--   psql -v ro_password="$(cat ~/secrets/<archivo>)" -d quantagent_registro -f crear_usuario_lectura.sql
-- (la contraseña viene de un archivo en ~/secrets o de una variable de entorno; nunca va en el repo).
-- Si el hilo del registro ya creó este usuario, no ejecutar: este archivo documenta los permisos mínimos esperados.
CREATE ROLE quantagent_registro_ro LOGIN PASSWORD :'ro_password';
ALTER ROLE quantagent_registro_ro SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE quantagent_registro TO quantagent_registro_ro;
GRANT USAGE ON SCHEMA public TO quantagent_registro_ro;
GRANT SELECT ON TABLE public.hypotheses, public.hypothesis_events,
    public.hypothesis_attempts, public.cycles TO quantagent_registro_ro;
