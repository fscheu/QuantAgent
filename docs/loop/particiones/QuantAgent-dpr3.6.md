# Partición de MAN-06

La implementación completa supera el límite de 10 archivos porque incluye las páginas CLI generadas. Se divide en cuatro entregas que mantienen cada diff por debajo de 100 líneas y 10 archivos.

| Hijo | Objetivo | Aceptación binaria | Revisión |
|---|---|---|---|
| `QuantAgent-dpr3.6.1` | Glosario y test desde roadmap §0/§5, plan M2 §14 y análisis del tablero §2/§13; máximo 2 archivos y 100 líneas | Cada término extraído tiene encabezado, definición y ejemplo; el test detecta omisiones | probar |
| `QuantAgent-dpr3.6.2` | Enlazar `criteria.md` y `decisions.md` al inicio; 2 archivos | Ambas páginas enlazan antes del contenido y las rutas relativas resuelven | leer |
| `QuantAgent-dpr3.6.3` | Enlazar las páginas CLI de primer nivel y mantenerlas sincronizadas con el generador; máximo 10 archivos | Cada página de primer nivel tiene enlace válido y coincide con el generador | probar |
| `QuantAgent-dpr3.6.4` | Enlazar las páginas CLI anidadas y mantenerlas sincronizadas con el generador; máximo 10 archivos | Cada página anidada tiene enlace válido y coincide con el generador | probar |

El padre `QuantAgent-dpr3.6` queda abierto hasta cerrar los cuatro hijos.
