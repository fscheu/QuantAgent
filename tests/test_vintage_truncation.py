"""Prueba de truncamiento de vintage (QuantAgent-d4cq.71). Snapshot sintético, sin red ni clave FRED."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.backtesting.signals import bar_signals, check_no_lookahead_macro
from quantagent.data.macro import crear_snapshot_macro, recortar_snapshot_macro, valor_conocido
from quantagent.data.snapshot import snapshot_root
from quantagent.fabrica.admision import admitir
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.strategy.registry import STRATEGY_REGISTRY, register_strategy

FIN = pd.Timestamp("9999-12-31")
H = cargar_hipotesis(Path(__file__).parent / "fixtures" / "hipotesis" / "rsi-valida.yaml")

# UNRATE: abril sale el 2020-05-01 con 4.0 y se revisa a 4.6 el 2020-05-15.
# DGS10 (diaria): el valor del jueves 05-07 aparece el viernes; el del viernes 05-08 recién el lunes 05-11.
FILAS = {
    "UNRATE": [("2020-03-01", 3.0, "2020-04-03", FIN), ("2020-04-01", 4.0, "2020-05-01", "2020-05-14"), ("2020-04-01", 4.6, "2020-05-15", FIN)],
    "DGS10": [("2020-05-07", 1.0, "2020-05-08", FIN), ("2020-05-08", 2.0, "2020-05-11", FIN)],
}


class ClienteSintetico:
    def observaciones(self, serie, desde, hasta, todas_las_versiones=True):
        return pd.DataFrame(
            [(pd.Timestamp(f), v, pd.Timestamp(a), pd.Timestamp(b)) for f, v, a, b in FILAS[serie]],
            columns=["fecha", "valor", "realtime_start", "realtime_end"],
        )


class MacroDePrueba(TradingStrategy):
    """LONG si el valor de `serie` supera `umbral`. `modo` "conocido" lee lo publicado en la fecha; "ultima" la última versión."""

    requiere_macro = True
    required_history_bars = 1

    def __init__(self, snapshot, serie="UNRATE", umbral=4.3, modo="conocido"):
        self.snapshot, self.serie, self.umbral, self.modo = snapshot, serie, umbral, modo

    def _valor(self, al):
        if self.modo == "conocido":
            conocido = valor_conocido(self.snapshot, self.serie, al)
            return None if conocido is None else conocido[1]
        df = pd.read_parquet(snapshot_root() / self.snapshot / f"{self.serie}.parquet")
        df = df[df["fecha"] <= al].sort_values(["fecha", "realtime_start"])
        return float(df["valor"].iloc[-1]) if len(df) else None

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        valor = self._valor(pd.Timestamp(kline_data[-1]["timestamp"]))
        if valor is not None and valor > self.umbral:
            return TradingSignal(decision="LONG", confidence=1.0)
        return None

    def should_reevaluate(self, position, current_price):
        return False


@pytest.fixture
def snap(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    crear_snapshot_macro("m", ["UNRATE", "DGS10"], ClienteSintetico(), base_dir=tmp_path)
    return tmp_path


def _velas(inicio, n):
    idx = pd.bdate_range(inicio, periods=n)
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"timestamp": idx, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)


def _chequear(snap, modo, serie, umbral, velas, cortes):
    est = MacroDePrueba("m", serie, umbral, modo)
    return check_no_lookahead_macro(lambda c: bar_signals(est, c, "SPY", "1d"), velas, cortes, "m")


def test_recorte_descarta_versiones_posteriores_y_acota_realtime_end(snap, tmp_path):
    destino = tmp_path / "recorte"
    manifiesto = recortar_snapshot_macro("m", "2020-05-08", destino)
    df = pd.read_parquet(destino / "m" / "UNRATE.parquet")
    assert df["valor"].tolist() == [3.0, 4.0]  # la revisión 4.6 (realtime_start 05-15) desaparece
    assert (df["realtime_end"] == pd.Timestamp("2020-05-08")).all()
    assert manifiesto["series"]["UNRATE"]["rows"] == 2
    assert manifiesto["series"]["DGS10"]["rows"] == 1  # el viernes todavía no se publicó el valor del viernes


def test_estrategia_con_valor_conocido_pasa_y_la_de_ultima_revision_falla(snap):
    velas, cortes = _velas("2020-04-01", 60), [pd.Timestamp("2020-05-08"), pd.Timestamp("2020-05-20")]
    assert _chequear(snap, "conocido", "UNRATE", 4.3, velas, cortes) == []
    difs = _chequear(snap, "ultima", "UNRATE", 4.3, velas, cortes)
    assert difs and difs[0].startswith("entry:") and "2020-05-08" in difs[0]


def test_demora_de_publicacion_de_serie_diaria_en_fin_de_semana(snap):
    sab, dom, lun = pd.Timestamp("2020-05-09"), pd.Timestamp("2020-05-10"), pd.Timestamp("2020-05-11")
    assert valor_conocido("m", "DGS10", sab)[1] == valor_conocido("m", "DGS10", dom)[1] == 1.0
    assert valor_conocido("m", "DGS10", lun)[1] == 2.0
    velas, cortes = _velas("2020-05-04", 10), [pd.Timestamp("2020-05-08"), lun]
    assert _chequear(snap, "conocido", "DGS10", 1.5, velas, cortes) == []
    difs = _chequear(snap, "ultima", "DGS10", 1.5, velas, cortes)  # asume el valor del viernes publicado el viernes
    assert difs and "2020-05-08" in difs[0]


def test_admitir_ejecuta_la_prueba_para_clases_con_requiere_macro(snap):
    register_strategy(MacroDePrueba, min_bars=1, params={p: {"type": str, "default": None, "description": ""} for p in ("snapshot", "serie", "modo")})
    try:
        velas = _velas("2020-04-01", 60)
        pierna = replace(H.piernas[0], clase="MacroDePrueba", parametros={"snapshot": ["m"], "serie": ["UNRATE"], "modo": ["conocido"]})
        assert admitir(replace(H, piernas=(pierna,)), velas) == (True, [])
        mala = replace(pierna, parametros={**pierna.parametros, "modo": ["ultima"]})
        ok, motivos = admitir(replace(H, piernas=(mala,)), velas)
        assert not ok and "prueba de truncamiento fallida" in motivos[0]
        sin_snapshot = replace(pierna, parametros={"serie": ["UNRATE"]})
        ok, motivos = admitir(replace(H, piernas=(sin_snapshot,)), velas)
        assert not ok and "snapshot" in motivos[0]
    finally:
        STRATEGY_REGISTRY.pop("MacroDePrueba")
