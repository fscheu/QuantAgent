# Partición de QuantAgent-e4jc.13

El ticket combina tres comportamientos operativos: comparar pesos de paper contra la sombra, auditar rechazos y dividendos, y publicar el resultado. Cada parte requiere pruebas sintéticas y puede verificarse por separado. No hay digest diario implementado en la base del lote, por lo que la última parte contempla un comando invocable.

Hijos creados:

- `QuantAgent-e4jc.13.1`: comparación pura de pesos con tolerancia de redondeo. Aceptación: el caso dentro de tolerancia pasa y una diferencia mayor falla con activo y motivo. Revisión: probar.
- `QuantAgent-e4jc.13.2`: reporte de rechazos y dividendos. Aceptación: un rechazo sin explicación falla; cada dividendo muestra símbolo, importe, moneda y tratamiento registrado, y si falta tratamiento falla. Revisión: probar.
- `QuantAgent-e4jc.13.3`: comando que publica las tres secciones e integración con digest si existe. Aceptación: comando reproducible muestra pesos, rechazos y dividendos y sale con error ante incumplimientos. Revisión: probar.
