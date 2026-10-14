"""Tests de scripts/recalc_verdict.py (QuantAgent-0oqm), el recálculo del veredicto sin `quantagent`.

`backtest verdict` (QuantAgent-0z29) todavía no existe: el fixture es una salida escrita a mano
con sus columnas. Los veredictos y motivos esperados salen de aplicar la regla de V30 a mano.
"""

import ast
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "scripts" / "recalc_verdict.py"
SALIDA = RAIZ / "tests" / "fixtures" / "verdict-4-celdas.csv"


def _correr(path):
    return subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)


def test_salida_correcta_coincide_entera():
    # Filas: pasa en los dos llenados; falla solo a la apertura; falla al cierre (y a la apertura);
    # sin columnas de apertura. QQQ está justo en los límites al cierre (sharpe 0.10, drawdown 0.35).
    r = _correr(SALIDA)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "coinciden 4 de 4"


def test_veredicto_cambiado_a_mano_falla_y_nombra_la_fila(tmp_path):
    texto = SALIDA.read_text().replace(
        "rsi,QQQ,2007-01-01,2018-12-31,NO PASA,apertura: pnl <= 0; apertura: sharpe_tasa_0 < 0.10,",
        "rsi,QQQ,2007-01-01,2018-12-31,PASA,,",
    )
    alterada = tmp_path / "verdict.csv"
    alterada.write_text(texto)
    r = _correr(alterada)
    assert r.returncode == 1
    lineas = r.stdout.strip().splitlines()
    assert lineas[0] == "coinciden 3 de 4"
    assert lineas[1] == (
        "rsi QQQ: salida=PASA () recalculado=NO PASA "
        "(apertura: pnl <= 0; apertura: sharpe_tasa_0 < 0.10)"
    )


def test_no_importa_quantagent():
    arbol = ast.parse(SCRIPT.read_text())
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos.update(a.name for a in nodo.names)
        elif isinstance(nodo, ast.ImportFrom):
            modulos.add(nodo.module or "")
    assert modulos == {"csv", "sys"}
    assert not any(m.split(".")[0] == "quantagent" for m in modulos)
