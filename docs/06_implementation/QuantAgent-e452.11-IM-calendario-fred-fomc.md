# Calendario FRED y FOMC

`quantagent api sync-fred-calendar` sincroniza IPC, empleo y PCE desde FRED, y reemplaza los eventos manuales desde `calendario/eventos_manuales.yaml`. Requiere `FRED_API_KEY` y `QUANTAGENT_DATABASE_URL`; se puede programar semanalmente.

Para agregar recordatorios, sumá entradas con `date` ISO y `title` no vacío en `events` del YAML. Al sincronizar, se eliminan de la base los eventos `manual` que ya no figuran allí.

Las fechas FOMC cargadas cubren 2026. Hay que cargar las de 2027 cuando la Fed publique su calendario.
