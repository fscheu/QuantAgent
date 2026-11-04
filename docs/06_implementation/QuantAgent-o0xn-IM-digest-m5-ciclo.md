# QuantAgent-o0xn: estado del ciclo diario M5 en el digest

`trading m5-correr` persiste heartbeats `running`, `completed` o `error` en `m5.db`, con el modo en `stats`. El digest M5 avisa en días hábiles cuando no encuentra un ciclo diario completado; omite la alerta los fines de semana.

Validación: `/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_digest_m5.py tests/test_digest_m5_semanal.py tests/test_m5_54_puesta_en_marcha.py tests/test_pipeline_m5.py`.
