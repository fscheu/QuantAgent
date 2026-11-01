# WEB-13: curva, sectores y dividendos

La API web expone benchmark SPY reescalado y valor de cartera sombra en la curva de equity, dividendos recibidos por cuenta desde eventos de la DB maestra y exposición agrupada por sector con long, short y cash. Las lecturas de snapshots faltantes producen valores nulos o clasificación sin sector; todos los accesos siguen siendo de solo lectura.

Validación: `pytest tests/api`.
