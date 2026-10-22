# QuantAgent-cq4u — Punto de control L19 (S09)

Lo corrió el PM en la VM el 2026-10-22, solo lectura: curvas guardadas de etapa 2 (2007–2018), snapshot `etf-1d-2026-10-amplio`, sin motor, sin ventanas nuevas y sin LLM. Las curvas de transversal, tendencia-momentum y estacionalidad salen de la corrida I08 (`/tmp/claude-1000/i08/<id>/equity`); las de núcleo-curva, de `local-data/curvas-l14-curva/etapa2`. El loop no pudo hacerlo tres veces porque trabaja en un checkout sin `local-data`.

## Las cuatro filas (M1 a M3)

M1 es el Sharpe en exceso de DGS3MO; M2 compara contra el benchmark con los mismos pesos; M3 es el error estándar del ΔSharpe contra 1/N.

| Cartera | M1 cartera | M1 1/N | Brecha M1 | Sharpe (tasa 0) benchmark con pesos | Sharpe (tasa 0) 1/N | M2 cartera menos pesos | M3 error estándar | Brecha / error estándar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transversal | 0,3610 | 0,3550 | 0,0060 | 0,4704 | 0,4114 | 0,1105 | 0,1378 | 0,04 |
| tendencia-momentum | 0,5223 | 0,3550 | 0,1673 | 0,6253 | 0,4114 | 0,1354 | 0,2232 | 0,75 |
| estacionalidad | 0,3859 | 0,3538 | 0,0321 | 0,4159 | 0,3919 | 0,0318 | 0,2117 | 0,15 |
| núcleo-curva | 0,4993 | 0,3550 | 0,1443 | 0,8616 | 0,4114 | -0,0465 | 0,2494 | 0,58 |

Las brechas coinciden con las de la atribución de S04 (`QuantAgent-v2ll-IM-atribucion.md`).

## Conclusión

Ninguna brecha contra 1/N llega a una vez su error estándar (la mayor, tendencia-momentum, es 0,75): con 12 años de datos, el Sharpe en exceso de las cuatro carteras no se distingue de comprar y mantener con pesos iguales. Casi toda la ventaja de núcleo-curva está en los pesos; su residual de señal es negativo (-0,0465).

## Comprobaciones

- Pedido en seco (`armar_pedido()`, sin LLM): 6899 caracteres, sin las palabras Sharpe, DSR ni ganancia.
- `intentos.csv`, `hipotesis.csv` y `hipotesis-candado.csv` con el mismo MD5 antes y después: `2b0a48fc68f503f732ad6e948c8c158d`, `10d64050e9fde2fa8c080b241f533751`, `2157d34f28454b8328cdd28f7df19cd5`. Ninguna ventana nueva abierta.
- `NUMBA_DISABLE_JIT=1 pytest -m oraculo`: 35 passed, 1668 deselected (13 min).
- `cartera-nucleo-curva` no está en `hipotesis.csv`, así que `backtest hipotesis informe` no la reconoce; sus cifras salen del mismo cálculo directo sobre las curvas guardadas.
