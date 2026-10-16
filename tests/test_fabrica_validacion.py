import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.data.snapshot import write_snapshot
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica import validacion


@pytest.mark.parametrize("abrir_prueba,etapa,desde", [(False, "2", "2007-01-01"), (True, "3", "2019-01-01")])
def test_motor_dos_activos(tmp_path, monkeypatch, abrir_prueba, etapa, desde):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range(desde, periods=150)
    close = 100 + 10 * np.sin(np.arange(150) / 5)
    velas = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "adj_close": close, "volume": 1000.0}, index=fechas)
    write_snapshot("sintetico", {"SPY": velas, "TLT": velas}, base_dir=tmp_path)
    h = cargar_hipotesis(Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml")
    params = {"SPY": {"rsi_period": 7}, "TLT": {"rsi_period": 21}}
    real, recibidos = validacion.subprocess.run, {}

    def observar(cmd, **kwargs):
        activo = cmd[cmd.index("--symbols") + 1]
        recibidos[activo] = json.loads(Path(cmd[cmd.index("--params") + 1]).read_text())
        assert ("--abrir-prueba" in cmd) == abrir_prueba
        return real(cmd, **kwargs)

    monkeypatch.setattr(validacion.subprocess, "run", observar)
    carpeta = tmp_path / "salida"
    filas = validacion.correr_motor(h, "sintetico", params, desde, fechas[-1].date(), carpeta, etapa, abrir_prueba)
    assert recibidos == {a: {h.clase: p} for a, p in params.items()}
    assert [f["activo"] for f in filas] == ["SPY", "TLT"]
    assert filas[0]["pnl"] != filas[1]["pnl"]
    for activo in params:
        for sub in ("trades", "equity"):
            with (carpeta / sub / f"rsi-{activo}.csv").open() as f:
                assert len(list(csv.DictReader(f))) > 0
    with (tmp_path / "intentos.csv").open() as f:
        intentos = list(csv.DictReader(f))
    assert len(intentos) == 2
    for intento, fila in zip(intentos, filas):
        assert intento["hipotesis"] == h.id and intento["etapa"] == etapa
        assert json.loads(intento["params"]) == params[fila["activo"]]
        assert intento["pnl"] == fila["pnl"]
    with pytest.raises(RuntimeError, match="AUSENTE"):
        validacion.correr_motor(h, "sintetico", {"AUSENTE": {}}, desde, fechas[-1].date(), carpeta, etapa, abrir_prueba)
    with (tmp_path / "intentos.csv").open() as f:
        assert list(csv.DictReader(f)) == intentos
