# QuantAgent-d4cq.14: partición del CLI de hipótesis

Estimación previa: 190 líneas de código y tests, más 35 de evidencia.
El límite objetivo por entrega es 100; esta entrega no implementa código.

## Entregas propuestas

| Hijo | Objetivo | Estimación | Revisión |
|---|---|---:|---|
| QuantAgent-d4cq.14.1 | Registro CLI, carga recortada a 2007–2018, admisión e intento de rechazo | 85 | leer |
| QuantAgent-d4cq.14.2 | Filtro por activo, tabla, intentos etapa 1 y PASA/CORTA | 95 | probar |
| QuantAgent-d4cq.14.3 | YAML reproducible y medición RSI de dos combinaciones en doce ETFs | 45 | probar |

La estimación de cada hijo incluye sus tests. Cadena: .1 → .2 → .3.
El primero conserva las dependencias H02, H03 y H05 del padre, ya cerradas.
La cola L9 reemplaza H06 por sus tres hijos; H07 sigue esperando al padre.
El padre queda in_progress hasta completar y revisar los tres hijos.

## Aceptación binaria por hijo

- .1: comando visible; rechazo real sale 1, imprime motivos y registra etapa 0;
  un snapshot con filas 2019 demuestra que admisión solo recibe 2007–2018.
- .2: tabla e intentos etapa 1 con hipótesis; dos activos dan PASA;
  nueve de doce bajo 0,10 dan CORTA; ocho pasan y 0,10 no cuenta como bajo.
  El filtro nunca recibe fechas desde 2019.
- .3: corrida real de doce ETFs termina, tabla y veredicto reproducibles,
  tiempo total e incremento de doce intentos etapa 1 documentados.

El criterio original «dos activos: caso que corta» es incompatible con
«más de ocho de doce». Se propone probar el corte con doce activos;
se conserva el umbral decidido por Fede, sin normalizarlo por universo.

Tests de cada hijo: `python -m pytest -q tests/test_cli_hipotesis.py tests/test_admision.py tests/test_fabrica_filtro.py`.
El archivo CLI todavía no existe: no corresponde ejecutarlo en esta partición.
Cada implementación deberá pasar esos tests y `scripts/loop/smoke.sh`.
No se cambiaron motores, estrategias, candado ni umbrales.
