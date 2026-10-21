# Partición de Q01

Estimación total: 180–230 líneas entre cálculo, CLI, tests e IM.
Se aplica PLAN-CONTINUACION.md §3.2 antes de escribir código.

| Hijo | Objetivo y aceptación binaria | Estimación | Revisión |
|---|---|---:|---|
| QuantAgent-ro3w.1 | Cálculo desde curvas; sintéticas pasan al superar ambos benchmarks y cortan con motivo si no superan etapa 3; test verifica cero aperturas de candado y cero escrituras de registros | 95 | leer |
| QuantAgent-ro3w.2 | CLI con tabla completa, N/DSR crudo y efectivo, recargo y motivos; fixture CLI verifica salida y ausencia de escrituras; informa curvas faltantes | 90 | probar |
| QuantAgent-ro3w.3 | Corrida autorizada e IM con tabla/veredicto reales o faltantes explícitos; hashes de registros/candado intactos y oráculos verdes | 35 | decidir |

Dependencias: .1 → .2 → .3. Cada estimación incluye sus tests y documentación.
El padre queda in_progress hasta integrar los hijos y resolver la decisión de Fede.

Riesgo identificado: `_curva_y_v30_cartera` llama `registrar_intento` cuando
recibe snapshot. El cálculo debe evitar ese camino y usar las curvas guardadas
o `curva_cartera`; no modificar el motor ni los umbrales.
Se conserva la autorización de Q01: leer únicamente curvas L14 y cierres de
los mismos activos/ventanas. Ninguna nueva corrida sobre 2019–2022.
