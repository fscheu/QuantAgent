# Partición de QuantAgent-g7kg

`filtrar_activo` recibe las velas de un único activo, pero `posiciones_objetivo` de F1 calcula pesos con los cierres sincronizados de SPY e IEF. Para evaluar correctamente los cambios por activo, hace falta conectar el cálculo multiactivo con la etapa 1 y cubrirlo con una prueba sintética; hacerlo en una entrega excede el objetivo de 100 líneas.

| Hijo | Objetivo | Criterio de aceptación | Tipo de revisión |
|---|---|---|---|
| QuantAgent-g7kg.1 | Extender el filtro para estrategias de pesos objetivo con el panel completo de activos y obtener métricas por activo. | Los cambios de peso generan órdenes contadas por activo; los resultados de estrategias de señal no cambian. | leer |
| QuantAgent-g7kg.2 | Integrar el filtro multiactivo en la etapa 1 y cargar/alinear los cierres de la pierna. | Etapa 1 ejecuta la hipótesis de pesos objetivo y suma las órdenes de todos sus activos. | leer |
| QuantAgent-g7kg.3 | Actualizar la prueba sintética de F1 y comprobar la ruta existente de señales. | El fixture cuenta 288 fechas y 576 órdenes; pruebas dirigidas pasan y oráculos no cambian. | probar |

La ejecución de F1, la reserva y el candado quedan fuera de alcance, según el ticket padre.
