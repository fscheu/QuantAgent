-- Ejecutar con psql -v ro_password='...' -d quantagent_registro.
CREATE ROLE quantagent_registro_ro LOGIN PASSWORD :'ro_password';
ALTER ROLE quantagent_registro_ro SET default_transaction_read_only = on;
GRANT CONNECT ON DATABASE quantagent_registro TO quantagent_registro_ro;
GRANT USAGE ON SCHEMA public TO quantagent_registro_ro;
GRANT SELECT ON TABLE public.hypotheses, public.hypothesis_events,
    public.hypothesis_attempts, public.cycles TO quantagent_registro_ro;
