# Decisiones abiertas

Cada fila tiene un valor provisorio para no bloquear el desarrollo. Cuando se decida, completar la última columna y actualizar el archivo indicado.

| # | Duda | Valor provisorio | Dónde vive | Se decide cuando… | Decisión |
|---|---|---|---|---|---|
| 1 | Nombres y función de las etapas 0–3 | "Etapa 0…3" | `config.ts`, `help-texts.es.json` | Se cierre el diseño del embudo | — |
| 2 | Motivos de descarte (salvo los 3 dados) | Inventados, de ejemplo | Backend | El motor del embudo los emita | — |
| 3 | Métrica que muestra cada hipótesis | Sharpe deflactado | Backend (`last_metric`) | Se definan las métricas por etapa | — |
| 4 | Umbrales del stop | cerca < 2 %, atención < 5 %, escala 15 % | `config.ts` | Después de ver stops reales | — |
| 5 | Distancia en % o en ATR | % | `config.ts` | Idem | — |
| 6 | Posiciones cortas | No se contemplan (solo largas) | `stopDistance()` | Si el sistema opera en corto | — |
| 7 | Orden por defecto de posiciones | Distancia al stop ascendente | Pantalla Cuenta | 2 semanas de uso | — |
| 8 | Rechazadas siempre arriba | Sí, también con "Todas" | Pantalla Cuenta | 2 semanas de uso | — |
| 9 | Ciclo por defecto en Embudo | Todos | `config.ts` | Prueba con datos reales | — |
| 10 | Hipótesis en escritorio: tarjetas o tabla | Tarjetas | Pantalla Embudo | Más de 30 activas | — |
| 11 | Alarmas: catálogo y severidades | drawdown (crítica), sombra (advertencia), datos viejos (advertencia) | Backend + EVENTOS.md | Se implementen las alarmas | — |
| 12 | Umbral de "dato viejo" | 30 min con mercado abierto | `config.ts` | Se conozca la frecuencia real del backend | — |
| 13 | Frecuencia de refresco | 60 s abierto / 15 min cerrado | `config.ts` | Idem | — |
| 14 | Dividendos: retención por pago o total del mes | Ambos (total + lista) | Pantalla Cuenta | Se implemente | — |
| 15 | Ventana del feed | Última visita o 24 h | `config.ts` | No urgente | — |
| 16 | Publicaciones macro a seguir en FRED | Inflación, empleo | Backend (lista blanca) | Se implemente el calendario | — |
| 17 | Biblioteca de gráficos (no velas) | Recharts | `package.json` | Primer gráfico | — |
| 18 | Secciones "próximamente": datos de ejemplo o fantasma | Fantasma punteado en producción | Flag | Primera entrega | — |

## Decisiones tomadas (registro)

| Fecha | Decisión |
|---|---|
| 29/10/2026 | Ganancia/pérdida en azul/naranja con signo y flecha; nunca solo color. |
| 29/10/2026 | Velas verde/rojo (convención de mercado) con respaldo hueco/lleno; solo en velas. |
| 29/10/2026 | Horas siempre con zona: órdenes y mercado en ET; actualización de datos en ART; eventos en ambas. |
| 29/10/2026 | Formato es-AR (USD 98.412,37), menos tipográfico U+2212. |
| 29/10/2026 | Sin total general en Inicio (no mezclar paper con live). |
| 29/10/2026 | Geist (texto) + Geist Mono tabular (cifras). |
| 29/10/2026 | El código es la fuente de verdad de los tokens; el canvas, de la estructura de pantallas. |
