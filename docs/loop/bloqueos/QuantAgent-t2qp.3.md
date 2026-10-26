# Bloqueo de QuantAgent-t2qp.3

El YAML oficial en `catalogo/propuestas/tendencia-larga-corta.yaml` usa el esquema de propuesta
(`idea`, `ventana`, `universo`) y no el esquema de ejecución de `cargar_hipotesis`
(`justificacion`, `parametros`, `activos`). Además, `tsmom-largo-corto` necesita el nombre de
un snapshot macro mediante `parametros.snapshot`; el `--snapshot` de `hipotesis correr` carga
los precios de los ETFs y no puede cumplir esa función.

Sin una regla de traducción se deben inventar al menos la procedencia de autor/justificación
y cómo se selecciona el snapshot macro. No hice cambios de código para evitar fijar datos o
semántica no definidos por el ticket.

Decisión que destraba: definir si el CLI debe aceptar y traducir el esquema de propuesta
(incluyendo el snapshot macro, por configuración o argumento) o si el YAML debe sumar los
campos de ejecución antes de correr. El YAML permanece fuera de `hipotesis/`.
