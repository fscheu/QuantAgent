# Bloqueo QuantAgent-e4jc.3

El YAML F1 fue aprobado según la decisión 7 de R01 y movido a `hipotesis/`.
Se intentó una sola vez la corrida hasta etapa 2, con snapshots reales y los
directorios de caché indicados por el ticket:

```text
Error: Parámetro desconocido 'peso_acciones_referencia' para la estrategia 'rebalanceo-fin-de-mes'. Opciones válidas:
```

La ejecución se detuvo antes del backtest. Etapas 0–2 no evaluadas; no se
generó carpeta de intento. `intentos.csv` conserva SHA-256
`2e1b903ce87e7b23f6de45dfd37a844db34ca15f4af45ac4d8acb86de9cb30e8` y
`hipotesis-candado.csv` conserva SHA-256
`8cc215d8a57c73991d2c36513b0743e09752d4383b0f8304f21f99ae8910f7a5`.

Destraba: corregir la representación/admisión de los parámetros F1 para que
`rebalanceo-fin-de-mes` pueda ejecutar la configuración aprobada, y repetir
la corrida autorizada una vez. No se abrió 2019–2022, la reserva ni el candado;
no se pide autorización para la reserva sin veredicto de etapa 2.
