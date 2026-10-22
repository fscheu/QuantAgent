# QuantAgent-e4jc.8 — partición de D-2

Estimación previa: 170 líneas (40 de regla, 50 de empalme, 80 de inventario
y receta), sin contar el manifiesto excluido del límite. Supera el objetivo
de 100 líneas de PLAN-CONTINUACION.md §3.2; no se implementó ni descargó nada.
Además, el criterio original exige un commit de la regla anterior a la lista:
una entrega única no permite demostrar esa precedencia.

| Hijo | Objetivo | Aceptación binaria | Líneas | Revisión |
|---|---|---|---|---|
| QuantAgent-e4jc.8.1 | Regla con corte explícito en 2007, sin seleccionar candidatos | Regla integrada en commit propio, condiciones verificables sin retornos futuros, exclusión PTP y tratamiento de desaparecidos; ninguna lista | 40 | leer |
| QuantAgent-e4jc.8.2 | Documento de empalme por clase con fuentes gratuitas | Enlaza commit de la regla; define fuentes, unidades, fórmula, fecha del enlace, validación en tramo común y límites; declara clases sin proxy defendible | 50 | leer |
| QuantAgent-e4jc.8.3 | Aplicar la regla y congelar ETFs en un snapshot nuevo | Lista con evidencia de admisión/rechazo posterior al commit de la regla; manifiesto versionado y `snapshot verify --name <nombre-nuevo>` exit 0 sobre datos reales | 80 | probar |

Orden obligatorio: .8.1 → .8.2 → .8.3, con dependencias en Beads.
La regla se redacta ahora con corte histórico en 2007; no se la presenta
como un documento que existía en 2007. No se eligen ETFs por sus retornos.
El empalme es documentación: no implementa precios sintéticos ni confunde
tasas o FX spot con retornos totales de un ETF.

Los tres hijos actualizan `local-data/README.md`; .8.3 también versiona el
manifiesto en `tests/fixtures/snapshots/`. Datos locales ignorados por git.
No se evalúa 2019–2022 ni se abre la reserva 2023+.
El padre conserva los criterios originales y queda `in_progress` hasta
integrar los tres hijos; X12 depende del último hijo (X14c) también en Beads.
La cola de §4 reemplaza X14 por X14a–c, todos en L20.
