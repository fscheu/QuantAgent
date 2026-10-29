# Partición de QuantAgent-dpr3.3

La referencia completa cubre el árbol Click, sus pruebas, las páginas de comandos y el índice. El volumen
estimado supera las 100 líneas, así que se divide en entregas chicas:

| Hijo | Objetivo | Criterio binario | Revisión |
|---|---|---|---|
| `QuantAgent-dpr3.3.1` | Renderizador Click determinista en memoria | Renderiza todos los grupos y opciones sin dependencias nuevas | leer |
| `QuantAgent-dpr3.3.2` | Test del renderizador | Detecta diferencias de comandos/opciones, orden inestable y help faltante | probar |
| `QuantAgent-dpr3.3.3` | Páginas de backtest e hipótesis | Esas páginas coinciden con el renderizador | probar |
| `QuantAgent-dpr3.3.4` | Páginas de trading, data e intentos | Esas páginas coinciden con el renderizador | probar |
| `QuantAgent-dpr3.3.5` | Páginas de walkforward, sweep, verdict y profile | Esas páginas coinciden con el renderizador | probar |
| `QuantAgent-dpr3.3.6` | Índice de referencia CLI | README enlaza las páginas con rutas relativas válidas | leer |

Los hijos implementan el alcance original de MAN-03 en ese orden; esta partición no agrega páginas explicativas.
