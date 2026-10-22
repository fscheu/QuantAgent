# QuantAgent-v2ll — Atribución de las cuatro carteras

Diagnóstico de etapa 2 (2007–2018), sin motor ni apertura de ventanas.
Se reconstruye la cartera con `curva_cartera`, igual que la validación guardada;
S02 aporta el benchmark mensual siempre invertido con pesos 1/σ agregados por activo.
S01 resta DGS3MO conocida al inicio de cada intervalo, incluidos fines de semana.
El universo de cada benchmark es la unión de activos de sus piernas.

Sean C la cartera, W el benchmark con sus pesos y E el benchmark 1/N;
S0 es Sharpe a tasa cero y Se es Sharpe en exceso. La identidad usada es:
pesos = S0(W) − S0(E); señal = S0(C) − S0(W);
tasa = [Se(C) − Se(E)] − [S0(C) − S0(E)].
Las tres partes suman la brecha en exceso, no la brecha a tasa cero.
Es una descomposición contable dependiente de este orden, no atribución causal.
«Señal» incluye efectivo, costos y diferencias de rebalanceo diario/mensual;
en transversal se conserva la combinación de celdas de la validación existente.

## Resultado

| Cartera | S0(C) | S0(W) | S0(E) | Pesos | Señal | Tasa | Brecha exceso |
|---|---:|---:|---:|---:|---:|---:|---:|
| transversal | 0.580856 | 0.470373 | 0.411390 | 0.058982 | 0.110484 | -0.163511 | 0.005955 |
| tendencia-momentum | 0.760771 | 0.625330 | 0.411390 | 0.213939 | 0.135442 | -0.182101 | 0.167280 |
| estacionalidad | 0.447659 | 0.415863 | 0.391918 | 0.023945 | 0.031795 | -0.023602 | 0.032139 |
| nucleo-curva | 0.815061 | 0.861561 | 0.411390 | 0.450170 | -0.046499 | -0.259361 | 0.144310 |

MD5 antes = después (ejecución 2026-10-22): `intentos.csv` = `2b0a48fc68f503f732ad6e948c8c158d`;
`hipotesis.csv` = `10d64050e9fde2fa8c080b241f533751`;
`hipotesis-candado.csv` = `2157d34f28454b8328cdd28f7df19cd5`.
La mayor parte de la ventaja bruta de núcleo-curva proviene de pesos;
su residual de señal es negativo. La tasa casi elimina la ventaja transversal.
Estos valores son diagnósticos, no cambian cortes, DSR ni decisiones de admisión.

## Reproducción de solo lectura

Ejecutar el único bloque Python de este archivo desde la raíz del checkout.
Las rutas son las indicadas por el PM; solo se leen curvas de etapa 2.
Los snapshots se validan mediante su lector habitual. Los asserts impiden
silenciar fechas incompletas, datos ausentes o cambios en los tres registros.

```python
import hashlib
from pathlib import Path
import pandas as pd
from quantagent.data.snapshot import read_snapshot
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.cartera import curva_cartera, pesos
from quantagent.fabrica.criterio import curva_benchmark, sharpes_en_exceso
from quantagent.backtesting.vbt_filter import equity_metrics

base = Path('/home/azureuser/repos/projects/QuantAgent/local-data')
registros = ['intentos.csv', 'hipotesis.csv', 'hipotesis-candado.csv']
antes = {n: hashlib.md5((base / n).read_bytes()).hexdigest() for n in registros}
print('| Cartera | S0(C) | S0(W) | S0(E) | Pesos | Señal | Tasa | Brecha exceso |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for nombre in ['transversal', 'tendencia-momentum', 'estacionalidad', 'nucleo-curva']:
    h = cargar_hipotesis(Path('hipotesis') / f'cartera-{nombre}.yaml')
    carpeta = (base / 'curvas-l14-curva/etapa2' if nombre == 'nucleo-curva'
               else Path('/tmp/claude-1000/i08')) / h.id / 'equity'
    curvas = {}
    for i, p in enumerate(h.piernas):
        for a in p.activos:
            archivo = carpeta / f'{p.clase}-{a}.csv'
            eq = pd.read_csv(archivo, index_col=0, parse_dates=True)['equity']
            assert eq.index.is_unique and eq.index.is_monotonic_increasing, archivo
            assert eq.index.min().year == 2007 and eq.index.max().year == 2018, archivo
            curvas[i, a] = eq
    fechas = next(iter(curvas.values())).index
    assert all(eq.index.equals(fechas) and eq.notna().all() for eq in curvas.values())
    mapa = {i: list(p.activos) for i, p in enumerate(h.piernas)}
    cierres = pd.DataFrame({a: read_snapshot('etf-1d-2026-10-amplio', a, base)['close']
                            .reindex(fechas) for _, a in curvas})
    assert cierres.notna().all().all()
    c = curva_cartera(cierres, mapa, {k: eq.pct_change().dropna() for k, eq in curvas.items()})
    w = pesos(cierres, mapa).T.groupby(level=1).sum().T
    b = curva_benchmark(cierres, pesos=w).reindex(c.index)
    e = curva_benchmark(cierres).reindex(c.index)
    sc, sw, se = [equity_metrics(x, 0.0)['sharpe'] for x in (c, b, e)]
    ec, ee = sharpes_en_exceso(c, e, base_dir=base)
    partes = [sw - se, sc - sw, (ec - ee) - (sc - se)]
    assert abs(sum(partes) - (ec - ee)) < 1e-12
    print('| ' + nombre + ' | ' + ' | '.join(f'{x:.6f}' for x in [sc, sw, se, *partes, ec-ee]) + ' |')
despues = {n: hashlib.md5((base / n).read_bytes()).hexdigest() for n in registros}
assert antes == despues
for n in registros:
    print(f'{n}: antes={antes[n]} después={despues[n]} IGUALES')
```
