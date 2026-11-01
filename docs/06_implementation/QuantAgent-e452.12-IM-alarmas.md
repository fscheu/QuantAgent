# Alarmas mínimas y estado de Inicio

`apps/api/alarms.py` evalúa drawdown, diferencia de pesos frente a la sombra y antigüedad de datos sin escribir en las bases. Los valores iniciales son drawdown 8 %, diferencia máxima por activo 5 puntos porcentuales y datos vencidos a los 30 minutos con mercado abierto o 24 horas cerrado; `alarm_config.py` permite reemplazarlos por variables `QUANTAGENT_ALARM_*`.

`quantagent api evaluate-alarms` compara el resultado con `events` y agrega únicamente aperturas o resoluciones. Las alarmas de cuenta usan `entity_id=NULL`, en línea con la clave del feed; la resolución lleva `payload.alarm_type`. Se asume que la clave de cartera guardada en el JSON de sombra es `account.id`, y que la fecha más reciente del archivo contiene los pesos objetivo.

La API web sigue siendo de solo lectura. El endpoint `/api/home/status` agrega las alarmas actuales de cuentas con posiciones; el evaluador no escribe y lee las DB de cuenta con `query_only`.
