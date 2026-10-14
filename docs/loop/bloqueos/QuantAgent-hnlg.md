# QuantAgent-hnlg: igualdad de métricas al cierre no reproducida

Implementé opción A: frecuencia observada `(n-1)/años`, 365.25 días,
retornos simples, desviación muestral, equity en dólares y sin centinela.
En spy-2y-1d/fifty-two-week-high, control `coinciden 6 de 6`, pero
Sharpe adaptador -4.63 frente a -4.68 del motor. Llevar la liquidación
de la centinela al último cierre real tampoco corrigió la diferencia.
El verificador independiente reprodujo el fallo; los otros 14 tests pasaron.
No identifiqué todavía la causa precisa. Retiré código y tests nuevos.
El ticket queda in_progress; el motor y el adaptador no cambiaron.

En el intento VM rsi/SPY 2007–2018, costos etf, tamaño 1 y mínima 50:
control 187/187; Sharpe next-open -0.45; drawdown 0.177756.
Decisión: separar un diagnóstico de curvas, o aceptar verificar la fórmula
sobre una misma curva sin exigir igualdad con la curva propia del motor.

Reproducción autónoma (el comando Probar ejecuta este bloque):
```
import csv, io, os, runpy, sys, tempfile
from pathlib import Path
from click.testing import CliRunner
sys.path.insert(0, 'scripts')
import compare_control as c
t = runpy.run_path('tests/test_compare_control.py')
with tempfile.TemporaryDirectory() as d:
    os.environ['QUANTAGENT_SNAPSHOT_DIR'] = d
    os.environ['QUANTAGENT_PRUEBA_DESDE'] = os.environ['QUANTAGENT_RESERVA_DESDE'] = '2030-01-01'
    t['write_snapshot']('t', {'SPY': t['_daily_frame']()})
    result = CliRunner().invoke(t['backtest_group'], ['compare', '--strategies', 'fifty-two-week-high', '--symbols', 'SPY', '--snapshot', 't', '--out', d+'/cmp.csv', '--trades-dir', d+'/trades'])
    assert result.exit_code == 0, result.output
    c.settings.TRADING_BASE_POSITION_PCT = 1.0
    stats = c.bt_adapter.run_adapter(c.build_strategy('FiftyTwoWeekHighStrategy'), c.port.load_fixture(snapshot='t', symbol='SPY'), symbol='SPY', timeframe='1d', commission_pct=c.settings.TRADING_COMMISSION_PCT, slippage_pct=c.settings.TRADING_SLIPPAGE_PCT)
    row = next(csv.DictReader(Path(d+'/cmp.csv').open()))
    trades = list(csv.DictReader(io.StringIO(c.trades_to_csv(c.bt_adapter.trade_rows(stats, 'SPY')))))
    e = stats['_equity_curve']['Equity'].iloc[:stats['_strategy'].n_real_bars]
    r = e.pct_change().dropna()
    p = (len(e)-1)*365.25*86400/(e.index[-1]-e.index[0]).total_seconds()
    print(c.compare_trades_with_reasons(list(csv.DictReader(Path(d+'/trades/fifty-two-week-high-SPY.csv').open())), trades))
    print('Sharpe motor:', row['sharpe_tasa_0'], 'adaptador:', round(float(r.mean()/r.std()*p**0.5), 2))
```
