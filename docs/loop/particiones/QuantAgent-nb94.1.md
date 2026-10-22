# Partición de QuantAgent-nb94.1

La validación de campos nuevos, las pruebas de listas cerradas, los tres YAML existentes y sus fixtures de escritura superan el objetivo de 100 líneas.

## Hijos

| Hijo | Objetivo | Aceptación binaria | Revisión |
|---|---|---|---|
| QuantAgent-nb94.1.1 | Exigir en `validar_propuesta` los campos nuevos de §5.2 y rechazar valores inválidos en rol, tema y fuente de retorno. | Una propuesta completa pasa; campos requeridos ausentes y valores fuera de lista se rechazan; pruebas dirigidas verdes. | leer |
| QuantAgent-nb94.1.2 | Completar las tres propuestas de L19 y adaptar los fixtures de escritura. | Las tres propuestas completas pasan validación; la escritura acepta una completa y no escribe una incompleta. | leer |

El segundo hijo depende del primero. El padre nb94.1 queda como agrupador; S06.2 depende de que cierre el padre.
