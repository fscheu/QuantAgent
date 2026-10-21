# QuantAgent-ro3w: la cartera núcleo + curva contra el criterio de paper (L18, 2026-10-20)

Corrió el PM con `scripts/criterio_paper_desde_curvas.py` (Q01a y Q01b, mergeados) sobre las curvas guardadas de la corrida única de L14 (`local-data/curvas-l14-curva/{etapa2,etapa3}`) y los cierres del snapshot `etf-1d-2026-10-amplio` de los mismos activos. **No se corrió el motor, no se abrió ningún candado y no se escribió en ningún registro**: los md5 de `intentos.csv`, `hipotesis.csv` e `hipotesis-candado.csv` son los mismos antes y después. No se leyó ninguna ventana distinta de las ya abiertas para esta cartera.

```bash
QUANTAGENT_SNAPSHOT_DIR=local-data python scripts/criterio_paper_desde_curvas.py \
  hipotesis/cartera-nucleo-curva.yaml local-data/curvas-l14-curva/etapa2 local-data/curvas-l14-curva/etapa3 etf-1d-2026-10-amplio
```

| Cartera | Sharpe etapa 3 | Benchmark etapa 3 | Sharpe etapa 2 | Benchmark etapa 2 | Años positivos | Drawdown | Veredicto |
|---|---:|---:|---:|---:|---:|---:|---|
| núcleo + curva | 0,38 | 0,61 | 0,82 | 0,41 | 3 de 4 | 6,7 % | **CORTA** |

| Cartera | DSR con N efectivo (96) | DSR con N crudo (111) | Recargo de paper si pasara |
|---|---:|---:|---|
| núcleo + curva | 0,41 | 0,40 | 50 % del tamaño, 12 semanas |

## Qué dice

- **No pasa el criterio de paper.** Cumple todo menos una condición: en 2019–2022 su Sharpe (0,38) no supera al de comprar y mantener los mismos activos con pesos iguales (0,61). Sharpe positivo, 3 de 4 años y drawdown (6,7 % contra 0,35) están bien; la etapa 2 sí supera al benchmark (0,82 contra 0,41).
- Es el mismo motivo que cortó a las otras tres carteras (`QuantAgent-n96l`). La capa curva mejora al núcleo (0,25 a 0,38) pero no alcanza a la referencia.
- Consistencia con L14: Sharpe 0,38 y 0,82 coinciden con 0,3756 y 0,8151 de las corridas registradas.
- Validación del script: sobre las curvas de `cartera-tendencia-momentum` reproduce exactamente el IM de n96l (0,76/0,25, benchmark 0,41/0,61, 3 de 4, 5,5 %).

## Consecuencia para M4

Ninguna cartera cumple hoy el criterio de paper, así que M4 no tiene cartera de entrada por esta vía. La más cercana sigue siendo `cartera-transversal` (0,56 contra 0,61), no la curva.

## Qué no se hizo

- Ninguna corrida del motor ni lectura nueva de 2019–2022. No se cambió el criterio ni los registros.
