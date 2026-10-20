# QuantAgent-d4cq.75: punto de control de la capa macro (L14)

Corrida del PM el 2026-10-19 de `backtest hipotesis comparar` sobre datos reales en `local-data` para las dos carteras candidatas de capa contra la cartera núcleo (`hipotesis/cartera-tendencia-momentum.yaml`), snapshot `etf-1d-2026-10-amplio` y `macro-2026-10`, margen 0,05 fijado por Fede. Solo período 2007–2018 (fuera de muestra); tramo de prueba 2019–2022 intacto.

## Verificación previa del snapshot macro

```bash
python -m quantagent.cli data macro verify --name macro-2026-10
```
Resultado: `OK 10 series, 125185 filas`. Pasa antes de comenzar las comparaciones.

## Comparación de carteras de capa (2007–2018)

| Cartera con capa | Período | Sharpe núcleo | Sharpe con capa | ΔSharpe | DSR núcleo | DSR con capa | ΔDSR | N con capa | Veredicto |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `cartera-nucleo-curva` | 2007–2018 | 0,7608 | 0,8151 | +0,0543 | 0,0000 | 0,0000 | +0,0000 | 107 | DESCARTA |
| `cartera-nucleo-rsi-regimen` | 2007–2018 | 0,7608 | 0,6969 | -0,0638 | 0,0000 | 0,0000 | +0,0000 | 109 | DESCARTA |

## Lectura

- Ambas capas se descartan al no superar el margen de ΔDSR > 0,05 fijado por Fede antes de correr.
- `cartera-nucleo-curva`: la pierna de bonos con pendiente positiva sube levemente el Sharpe (+0,0543), pero con N=107 el DSR deflactado es 0,0000 en ambos casos y ΔDSR es 0,0000. Veredicto: DESCARTA.
- `cartera-nucleo-rsi-regimen`: la pierna de RSI condicionada a volatilidad propia empeora el Sharpe (-0,0638); el DSR se mantiene en 0,0000. Veredicto: DESCARTA.
- Etapa 3 (2019–2022): no se abrió para ninguna de las carteras (requiere aceptación y autorización explícita de Fede). El candado de prueba en `hipotesis-candado.csv` no registra entradas nuevas.
- Registros en `local-data`: 4 filas agregadas a `intentos.csv` (etapa `comparar`, todas con `hasta: 2018-12-31`). N pasó de 105 a 109. Artefactos HTML en `/tmp/qa-comparar-macro/`.

## Verificación en la VM

```bash
python3 -c 'import csv, pathlib; p=pathlib.Path("/home/azureuser/repos/projects/QuantAgent/local-data"); i=list(csv.DictReader((p/"intentos.csv").open()))[105:]; c=(p/"hipotesis-candado.csv").read_text().splitlines(); assert len(i)==4 and all(x["etapa"]=="comparar" and x["hasta"]=="2018-12-31" for x in i); assert [x["hipotesis"] for x in i]==["cartera-tendencia-momentum","cartera-nucleo-curva","cartera-tendencia-momentum","cartera-nucleo-rsi-regimen"]; assert len(c)==4; print("Punto de control L14 OK: 2 capas descartadas (ΔDSR <= 0.05), N=109, prueba intacta")'
```

## Segunda medición (2026-10-20, registro limpio por E01 de L16)

Se repitió la comparación con L16 ya en main: `intentos.csv` sin las filas sintéticas retiradas por `QuantAgent-w12l` (99 filas, 95 distintas antes de esta corrida), mismo snapshot `etf-1d-2026-10-amplio`, mismo margen 0,05, solo 2007–2018. Prueba 2019–2022 sin abrir.

| Capa | ΔSharpe | DSR núcleo → con capa | ΔDSR | N crudo | Veredicto de la CLI |
|---|---:|---|---:|---:|---|
| `cartera-nucleo-curva` | +0,0543 | 0,7827 → 0,8330 | +0,0503 | 101 | ACEPTA (por 0,0003) |
| `cartera-nucleo-rsi-regimen` | -0,0638 | 0,8005 → 0,7341 | -0,0664 | 103 | DESCARTA |

Sensibilidad de la capa curva: la CLI mide ΔDSR contra el registro tal como está al correr, y la varianza de los Sharpe del registro cambia con cada intento nuevo. Recalculado sobre las mismas curvas con el registro final de la sesión (103 filas, 95 distintas): ΔDSR +0,0463 con N=101 y +0,0457 con N efectivo=95; ambos están por debajo del margen. El veredicto depende de qué filas hay en el registro, no de la capa.

Los DSR pasaron de 0,0000 (medición del 2026-10-19, registro con filas sintéticas) a cerca de 0,8: esa limpieza, no la capa, es lo que movió el nivel.

El ticket `QuantAgent-m9cx` (regla de aceptación con ΔSharpe y años positivos) sigue abierto y espera los umbrales de Fede.
