# QuantAgent-yo4s · Umbrales del veredicto, llenado, tamaño y cortos (V30)

- **Issue:** QuantAgent-yo4s (Fila V30, Lote 4 / L7)
- **Tipo:** Decisión metodológica previa a abrir la prueba 2019–2022
- **Estado:** Propuesta para decisión de Fede (4 decisiones a marcar)
- **Base de datos:** Snapshot `etf-1d-2026-10`, tramo de ajuste 2007-01-03 a 2018-12-31, 4 estrategias x 12 ETFs (48 celdas), `--costos etf` (comisión 0%, slippage 0.05%), capital 100.000 con `--tamano-posicion 1.0`.

Comando base de la matriz y control:
```bash
python -m quantagent.cli backtest compare --snapshot etf-1d-2026-10 --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 --trades-dir $QUANTAGENT_SNAPSHOT_DIR/punto-de-control-lote3-2026-10-13/cmp-trades --out $QUANTAGENT_SNAPSHOT_DIR/punto-de-control-lote3-2026-10-13/final.csv
python scripts/compare_control.py --tabla $QUANTAGENT_SNAPSHOT_DIR/punto-de-control-lote3-2026-10-13/final.csv --trades-dir $QUANTAGENT_SNAPSHOT_DIR/punto-de-control-lote3-2026-10-13/cmp-trades --snapshot etf-1d-2026-10 --costos etf --tamano-posicion 1.0
```

---

## 1. Supuesto de llenado del número oficial

- **Opciones:**
  - **A:** Cierre de la vela de la señal (motor actual).
  - **B:** Apertura de la sesión siguiente (`--fill next-open`, adaptador).
- **Dato medido (2007–2018):**
  - Comando: `python -c "import os, pandas as pd, numpy as np; d=os.getenv('QUANTAGENT_SNAPSHOT_DIR','local-data'); df=pd.read_csv(f'{d}/punto-de-control-lote3-2026-10-13/final.csv'); [(print(s, 'cambio signo pnl:', (np.sign(sub.pnl)!=np.sign(sub.pnl_apertura_siguiente)).sum(), 'cambio signo exceso:', (np.sign(sub.pnl-sub.bh_pnl)!=np.sign(sub.pnl_apertura_siguiente-sub.bh_pnl)).sum(), 'mediana diff:', round((sub.pnl_apertura_siguiente-sub.pnl).median(),2))) for s, sub in df.groupby('estrategia')]"`
  - `rsi`: cambia signo de PnL en 2/12 ETFs; exceso en 0/12; cambio mediano de PnL: -4.154,82 USD.
  - `fifty-two-week-high`: cambia signo de PnL en 1/12 ETFs; exceso en 0/12; cambio mediano: -48,32 USD.
  - `sma-cross`: cambia signo de PnL en 0/12 ETFs; exceso en 0/12; cambio mediano: -3.556,86 USD.
  - `momentum-12m`: cambia signo de PnL en 1/12 ETFs; exceso en 1/12; cambio mediano: -1.072,54 USD.
  - Global: en 33 de 48 celdas (68,8%) el PnL empeora al pasar a la apertura siguiente (mediana global: -715,36 USD).
- **Implicancias de elegir B:**
  - *Implementación:* la apertura siguiente hoy solo existe en el adaptador de `backtesting.py` (`scripts/bt_adapter.py --fill next-open`), no en el motor de QuantAgent. Si se elige B, debe definirse si el número oficial saldría del adaptador externo o si hace falta un cambio en el motor propio (ticket aparte en M2 o M3).
  - *Control:* **0 de 48 celdas** tienen control cruzado en ese modo. El control actual de `compare_control.py` valida trade a trade motor vs. adaptador ambos con llenado al cierre. Para apertura siguiente no existe control motor/adaptador (y en 1 celda de 48, `rsi`/GLD, la cantidad de trades difiere: 186 en adaptador vs. 187 en motor por una orden cancelada por gap al alza al dimensionar al 100%).
- **Recomendación:** **B** (apertura siguiente) como número oficial porque refleja la ejecución factible de órdenes diarias (requiriendo ticket para el motor o adoptando el adaptador); o **A** con la regla de que el signo no cambie en B.
- **Elección de Fede:** `Fede elige: [ ]`

---

## 2. Umbrales de PASA / NO PASA por par (2019–2022)

- **Opciones de juegos de umbrales:**
  - **Juego 1 (Laxo · Viabilidad):** Trades mínimos, PnL > 0, Max Drawdown <= B&H Max Drawdown.
    - Pasan dentro del ajuste (12 años, Trades >= 3): **31 de 48 celdas (64,6%)** (rsi: 4, 52w: 7, sma-cross: 10, momentum-12m: 10).
  - **Juego 2 (Moderado · Robustez y riesgo relativo):** Trades mínimos, PnL > 0, Sharpe (tasa 0) >= 0.10, Max Drawdown <= 0.35.
    - Pasan dentro del ajuste (12 años, Trades >= 5): **20 de 48 celdas (41,7%)** (rsi: 2, 52w: 6, sma-cross: 7, momentum-12m: 5).
  - **Juego 3 (Estricto · Superar a Buy & Hold):** Trades mínimos, PnL > 0, Exceso de retorno anualizado sobre B&H >= 0.
    - Pasan dentro del ajuste (12 años, Trades >= 5): **8 de 48 celdas (16,7%)** (rsi: 0, 52w: 1, sma-cross: 4, momentum-12m: 3).
- **Calibración de trades (12 años de ajuste vs. 4 años de prueba):**
  - En 12 años (2007–2018), los trades por año dentro del ajuste fueron:
    - `fifty-two-week-high`: min 1 trade (0,08/año), mediana 5,5 (0,46/año), max 9 (0,75/año).
    - `momentum-12m`: min 2 trades (0,17/año), mediana 7,0 (0,58/año), max 12 (1,00/año).
    - `sma-cross`: min 5 trades (0,42/año), mediana 6,0 (0,50/año), max 10 (0,83/año).
    - `rsi`: min 10 trades (0,83/año), mediana 216,0 (18,00/año), max 294 (24,50/año).
    - Comando: `python -c "import os, pandas as pd; d=os.getenv('QUANTAGENT_SNAPSHOT_DIR','local-data'); df=pd.read_csv(f'{d}/punto-de-control-lote3-2026-10-13/final.csv'); [(print(s, 'min:', sub.trades.min(), f'({sub.trades.min()/12:.2f}/a)', 'mediana:', sub.trades.median(), f'({sub.trades.median()/12:.2f}/a)', 'max:', sub.trades.max(), f'({sub.trades.max()/12:.2f}/a)')) for s, sub in df.groupby('estrategia')]"`
  - En una prueba de 4 años (2019–2022), exigir `Trades >= 5` pide 1,25 trades/año. Como `sma-cross` (máx 0,83/a), `momentum-12m` (máx 1,00/a) y `52w` (máx 0,75/a) nunca alcanzaron esa tasa en ningún ETF del ajuste, `Trades >= 5` las deja afuera por construcción.
  - **Mínimo propuesto para la ventana de 4 años:**
    - *Opción escalada uniforme (1/3 de la ventana):* **Trades >= 1** para Juego 1 (tasa >= 0,25/a) y **Trades >= 2** para Juego 2 y 3 (tasa >= 0,42/a).
    - *Opción por estrategia:* **Trades >= 1** o **>= 2** para baja frecuencia (`52w`, `sma-cross`, `momentum-12m`) y **Trades >= 5** para alta frecuencia (`rsi`).
- **Métrica con baja exposición:** Cuando la estrategia está poco tiempo invertida (`fifty-two-week-high` invierte 1%–3% del capital medio; `rsi` arriesga ~15%), el PnL nominal y el Sharpe a tasa 2% penalizan artificialmente el efectivo (que rinde 0%). Se compara **Sharpe a tasa 0 (`sharpe_tasa_0`)** y **Max Drawdown relativo a B&H (`max_drawdown < bh_max_drawdown`)**, que mide preservación de capital (52w tiene menor caída que B&H en 12 de 12 ETFs).
- **Comando:** `python -c "import os, pandas as pd; d=os.getenv('QUANTAGENT_SNAPSHOT_DIR','local-data'); df=pd.read_csv(f'{d}/punto-de-control-lote3-2026-10-13/final.csv'); print('J1:', ((df.trades>=3)&(df.pnl>0)&(df.max_drawdown<=df.bh_max_drawdown)).sum(), 'J2:', ((df.trades>=5)&(df.pnl>0)&(df.sharpe_tasa_0>=0.10)&(df.max_drawdown<=0.35)).sum(), 'J3:', ((df.trades>=5)&(df.pnl>0)&(df.exceso_sobre_referencia>=0)).sum())"`
- **Recomendación:** **Juego 2** con mínimo escalado a 4 años (**Trades >= 2**, o mínimo por estrategia), evaluando alfa relativo contra B&H solo en estrategias de alta exposición (`sma-cross`, `momentum-12m`).
- **Elección de Fede:** `Fede elige: [ ]`

---

## 3. Tamaño mínimo de orden

- **Opciones:**
  - **A:** Ninguno (regla actual, permite fracciones de acción y montos pequeños).
  - **B:** Monto mínimo de orden (ej. 100 USD; órdenes menores se descartan).
  - **C:** Acciones enteras (mínimo 1 acción entera, sin fraccionamiento).
- **Dato medido (2007–2018):**
  - De 2.681 trades totales en las 48 celdas:
    - Menores a 1 acción: **8 trades (0,30%)**, todos en `rsi` (8 de 2.450 trades, 0,33%).
    - Menores a 100 USD: **18 trades (0,67%)**, todos en `rsi` (18 de 2.450 trades, 0,73%).
    - En `fifty-two-week-high`, `sma-cross` y `momentum-12m`: **0 trades** bajo esos umbrales.
  - Comando: `python -c "import os, glob, csv; d=os.getenv('QUANTAGENT_SNAPSHOT_DIR','local-data'); ts=[row for f in glob.glob(f'{d}/punto-de-control-lote3-2026-10-13/cmp-trades/*.csv') for row in csv.DictReader(open(f))]; print('total:', len(ts), '<1 acc:', sum(1 for r in ts if abs(float(r['qty']))<1.0), '<100 usd:', sum(1 for r in ts if abs(float(r['qty']))*float(r['entry_price'])<100.0))"`
- **Recomendación:** **A** para M2 (no altera código ni crea discrepancias antes de abrir 2019–2022, afectando a menos del 1% de trades). Para paper trading M3 se propone **B** (100 USD).
- **Elección de Fede:** `Fede elige: [ ]`

---

## 4. Costo de préstamo en posiciones cortas

- **Opciones:**
  - **A:** Ignorarlo en M2 (tasa 0%, solo comisión y slippage).
  - **B:** Descontar tasa fija anual de préstamo prorrateada por día de tenencia corta (ej. 1,0% anual).
- **Dato medido (2007–2018):**
  - Proporción de trades cortos:
    - `rsi`: **1.300 cortos de 2.450 trades (53,06%)**.
    - `fifty-two-week-high`: **0 cortos de 70 trades (0,00%)**.
    - `sma-cross`: **0 cortos de 83 trades (0,00%)**.
    - `momentum-12m`: **0 cortos de 78 trades (0,00%)**.
  - Total celdas: 1.300 cortos de 2.681 trades (48,49%).
  - Comando: `python -c "import os, glob, csv; d=os.getenv('QUANTAGENT_SNAPSHOT_DIR','local-data'); ts={}; [(ts.setdefault(os.path.basename(f).rsplit('-',1)[0], []).extend(list(csv.DictReader(open(f))))) for f in glob.glob(f'{d}/punto-de-control-lote3-2026-10-13/cmp-trades/*.csv')]; [(print(s, 'cortos:', sum(1 for r in rows if r['side'].lower() in ('sell','short')), 'de', len(rows), f'({sum(1 for r in rows if r[\"side\"].lower() in (\"sell\",\"short\"))/len(rows):.2%})')) for s, rows in sorted(ts.items())]"`
- **Recomendación:** **A** para M2: los 12 ETFs son activos altamente líquidos (fácil préstamo institucional, costo típico despreciable para horizontes de pocos días); no justificaría rehacer comparaciones previas.
- **Elección de Fede:** `Fede elige: [ ]`
