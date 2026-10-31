# M5-36: H1 y H3, etapa 3 (2019–2022)

**H1 CORTA; H3 CORTA. M5-55: NO sumar acciones al paper.** H3 tiene solo 2/4 años positivos,
aunque mejora el Sharpe de H1. El sesgo tampoco queda certificado como bajo: la cota de peso medio
afectado es 55,71%. La falta de precios impide identificar el conteo exacto del top 3 contrafactual.

## Resultados netos (1.008 sesiones, 208 semanas)

| Métrica | H1 | H3 |
|---|---:|---:|
| Sharpe a tasa 0 | 0,071614 | 0,236142 |
| CAGR | −0,2909% | 2,7208% |
| MaxDD | 37,0196% | 34,6243% |
| Rotación anual (Σ\|Δw\|) | 20,5676 | 29,7032 |
| Costos acumulados (capital inicial 100.000) | 3.818,71 | 5.819,48 |
| Costo anual aproximado (rotación × tarifa) | 1,0284% | 1,4852% |
| Años con Sharpe positivo | 2/4 | 2/4 |
| Operaciones (cambios de peso) | 593 | 1.739 |
| Benchmark de etapa 3 (Sharpe) | 0,600181 | 0,071614 |
| DSR informativo tras registrar ambos intentos | 0,260586 | 0,376646 |
| Veredicto | CORTA | CORTA |

**ΔSharpe H3−H1 = +0,164528 ± 0,212723**, un error estándar bootstrap pareado; no es un intervalo al 95%.
H1 corta por Sharpe inferior al benchmark, 2/4 años positivos y maxDD > 35%.
H3 supera a H1 y respeta el límite de drawdown, pero corta por 2/4 años positivos.
La recomendación NO para M5-55 se sostiene por ese corte, independientemente del sesgo no identificado.

## Tablas de exposición a faltantes

Descartes = acción-semanas antes del ranking, no reemplazos confirmados del top 3.
Puestos directos y peso son cotas superiores; sus límites inferiores son cero. Ver método abajo.

| Año | Descartes | Puestos directos ≤ | Semanas afectadas | % semanas | % peso ≤ |
|---|---:|---:|---:|---:|---:|
| 2019 | 396 | 267 | 48/52 | 92,31 | 62,42 |
| 2020 | 495 | 282 | 42/52 | 80,77 | 60,98 |
| 2021 | 479 | 362 | 52/52 | 100,00 | 79,16 |
| 2022 | 102 | 84 | 18/52 | 34,62 | 20,30 |
| Total | 1.472 | 995 | 160/208 | 76,92 | 55,71 |

Por año y sector: cada celda muestra **descartes; % semanas; % peso máximo medio**.
Los porcentajes de peso por sector suman el porcentaje total; los de semanas no se suman.

| Sector | 2019 | 2020 | 2021 | 2022 | Total |
|---|---:|---:|---:|---:|---:|
| Communication Services | 0; 0,0; 0,0 | 112; 26,9; 9,4 | 56; 13,5; 4,4 | 0; 0,0; 0,0 | 168; 10,1; 3,4 |
| Consumer Discretionary | 43; 21,2; 6,5 | 72; 46,2; 15,9 | 24; 23,1; 7,5 | 0; 0,0; 0,0 | 139; 22,6; 7,5 |
| Consumer Staples | 50; 48,1; 17,2 | 4; 3,8; 1,6 | 4; 3,8; 1,4 | 24; 23,1; 10,7 | 82; 19,7; 7,7 |
| Energy | 0; 0,0; 0,0 | 0; 0,0; 0,0 | 52; 25,0; 5,4 | 68; 32,7; 7,8 | 120; 14,4; 3,3 |
| Financials | 24; 11,5; 3,5 | 0; 0,0; 0,0 | 69; 44,2; 14,5 | 3; 1,9; 0,6 | 96; 14,4; 4,6 |
| Health Care | 25; 9,6; 3,2 | 30; 11,5; 4,0 | 64; 28,8; 10,7 | 4; 1,9; 0,7 | 123; 13,0; 4,7 |
| Industrials | 12; 3,8; 1,2 | 39; 25,0; 7,7 | 33; 21,2; 8,1 | 0; 0,0; 0,0 | 84; 12,5; 4,2 |
| Information Technology | 151; 34,6; 9,1 | 208; 50,0; 14,4 | 95; 28,8; 8,9 | 0; 0,0; 0,0 | 454; 28,4; 8,1 |
| Materials | 0; 0,0; 0,0 | 26; 25,0; 7,4 | 10; 9,6; 3,2 | 0; 0,0; 0,0 | 36; 8,7; 2,6 |
| Real Estate | 87; 55,8; 19,1 | 4; 1,9; 0,6 | 72; 46,2; 15,1 | 3; 1,9; 0,5 | 166; 26,4; 8,8 |
| Utilities | 4; 7,7; 2,7 | 0; 0,0; 0,0 | 0; 0,0; 0,0 | 0; 0,0; 0,0 | 4; 1,9; 0,7 |

## Método y límites

Ventana fija 2019-01-01..2022-12-31; última sesión 2022-12-30. No se evalúa la reserva 2023+.
Se reutilizan `h3.pesos_h1(desde, hasta)`, `pesos_h3`, `curva` y `metricas`, sin cambiar parámetros de estrategia.
Precios de acciones: `acciones-sp500-2026-10c`; calentamiento hasta 2018 con `acciones-sp500-2026-10b`.
ETFs: `etf-m5-1d-2026-10`; fundamentales y composición: `fundamentales-sp500-2026-10` y `sp500-2026-10`.
Paneles fundamentales calculados con `paneles_en_fechas`, que reutiliza las vistas por presentación vigente (f+1 día).
El cache nuevo `/tmp/ai-loop/h3-etapa3-2019-2022.pkl` guarda período y snapshot; rechaza un cache distinto.
El cache de 2011–2018 se conserva. Los retornos y las señales se truncan en 2022.
Llenado a la apertura siguiente, costo `etf` de 0,05% por unidad de rotación, idéntico para ambas curvas.
Se exige precio de apertura y cierre en toda sesión con peso positivo; no se imputan fills faltantes.

Criterio: `evaluar_etapa3_cartera` (Sharpe > 0 y mayor al benchmark en etapas 2 y 3,
al menos 3 de 4 años positivos, maxDD ≤ 35%, al menos 2 operaciones, constantes V30 vigentes). DSR informativo.
H1 usa 1/N mensual de los 14 ETFs; H3 usa H1, conforme a sus YAML.
Los Sharpes de etapa 2 son 0,61 y 0,65, publicados en [M5-33](QuantAgent-fnpr.4.4-IM-h3.md).
El benchmark 1/N de 2011–2018 se recalcula: 0,599431. No se aplica el corte DSR ≥ 0,95 antiguo.
Permanece la limitación de M5-33: no hay control independiente de esta curva contra otro motor.

## Sesgo: qué se puede medir

Sin el precio faltante no se conoce la fuerza relativa de una acción: tampoco su score, el z-score del
sector ni el cuartil. Por eso **no es identificable el número exacto de reemplazos del top 3 contrafactual**.
No se sustituye ese número por el total de miembros sin precio ni por cero.

El conteo semanal incluye miembros de los sectores elegidos por H1 que quedan excluidos porque les
falta momentum de precio, pero pasarían el filtro de fundamentales (como máximo uno faltante).
Se respeta que un único NaN total no excluye. Las semanas en efectivo tienen peso afectado cero.
Se informa el número de descartes antes del ranking y la cota superior de puestos directos afectados:
`min(3, descartes)` por sector-semana. Los reemplazos efectivos están entre cero y esa cota;
los cambios indirectos del ranking no quedan acotados por ese conteo de puestos.

La cota conservadora de peso incluye **todo el peso del sector** cuando hay alguno de esos descartes:
los z-scores y el cuartil podrían cambiar todas las posiciones. Es peso objetivo semanal medio sobre
todas las semanas, incluidas las de efectivo, no pérdida de retorno ni proporción de patrimonio perdido.
Cada porcentaje de semanas usa las semanas de su año, o todas las de 2019–2022 para un sector/total.
H1 no tiene este sesgo de selección de acciones; la comparación favorece a H3.
Las cotas no certifican que el sesgo sea bajo y no constituyen una corrección del Sharpe observado.
También incluyen historia insuficiente (por ejemplo, altas recientes); no todo descarte es supervivencia.

## Reproducción y validación

```bash
PYTHONPATH=.:scripts/m5 python scripts/m5/etapa3.py --sin-registro
python -m pytest -q tests/test_m5_etapa3.py tests/test_m5_h4.py
python -m ruff check --no-fix scripts/m5/h3.py scripts/m5/etapa3.py tests/test_m5_etapa3.py
scripts/loop/smoke.sh
```

Sin `--sin-registro` se abre el candado de prueba y se registra una sola fila por hipótesis en etapa 3;
repetir el comando conserva los registros. La reserva permanece cerrada.
Los archivos `.sesgo.csv` (semana × sector) y `.equity.csv` quedan junto al cache fuera de Git.
El primero permite reconstruir las tablas por año y sector. No se necesita red con el cache construido.
Registro real 2026-10-31 14:18 UTC: `intentos.csv` tiene una fila de etapa 3 para `m5-h1` y una para `m5-h3`;
ambas con estado CORTA en `hipotesis.csv`. `hipotesis-candado.csv` tiene solo sus dos aperturas de prueba.
Validación: 9 tests pasaron, lint sin correcciones OK, smoke exit 0 (9 trades).
Snapshot 10c verificado: 611 símbolos y 1.155.273 filas, hashes OK.
