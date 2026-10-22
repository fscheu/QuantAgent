# Partición de QuantAgent-nb94

El chequeo S06 supera el objetivo de 100 líneas porque combina contrato de campos, verificación HTTP contra Crossref, integración con la escritura de propuestas, tres propuestas existentes y pruebas sin red.

## Hijos

| Hijo | Objetivo | Aceptación binaria | Revisión |
|---|---|---|---|
| QuantAgent-nb94.1 | Exigir y validar los campos nuevos de propuesta y actualizar las tres propuestas de L17. | Las tres propuestas cargan y pasan validación local; faltantes y valores fuera de lista se rechazan. | leer |
| QuantAgent-nb94.2 | Verificar DOI y título con Crossref; integrar el resultado en `escribir_propuestas`. | DOI coincidente pasa; DOI ajeno se rechaza; error de red queda «sin verificar» y no escribe propuestas; tests pasan sin red. | probar |

El padre queda como agrupador. S08 depende del chequeo completo del padre.
