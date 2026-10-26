from datetime import date

import pandas as pd
import pandas_market_calendars as mcal

from quantagent.strategy import build_strategy
from quantagent.strategy.registry import STRATEGY_REGISTRY, cli_aliases
from quantagent.strategy.tsmom_largo_corto_strategy import TsmomLargoCortoStrategy


def _velas(retorno, fin=date(2023, 9, 29)):
    calendario = mcal.get_calendar("NYSE")
    sesiones = calendario.valid_days("2022-01-01", fin).tz_localize(None)[-253:]
    cierres = [100 + (retorno * 100 * i / 252) for i in range(253)]
    return [{"timestamp": fecha, "open": cierre, "high": cierre, "low": cierre,
             "close": cierre, "volume": 1} for fecha, cierre in zip(sesiones, cierres)]


def test_senal_mensual_compara_252_sesiones_con_dgs3mo(monkeypatch):
    fechas_consultadas = []

    def tasa_conocida(snapshot, serie, fecha):
        fechas_consultadas.append((snapshot, serie, pd.Timestamp(fecha)))
        return (pd.Timestamp(fecha), 3.0)

    monkeypatch.setattr("quantagent.strategy.tsmom_largo_corto_strategy.valor_conocido", tasa_conocida)
    estrategia = TsmomLargoCortoStrategy("macro-test")
    baja = estrategia.generate_signal(_velas(-0.20), "TEST", "1d", 80)
    sube = estrategia.generate_signal(_velas(0.20), "TEST", "1d", 120)

    assert baja.decision == "SHORT"
    assert sube.decision == "LONG"
    assert len(_velas(0.2)) == estrategia.required_history_bars == 253
    assert all(s == "DGS3MO" and snap == "macro-test" for snap, s, _ in fechas_consultadas)
    assert fechas_consultadas[0][2] == pd.Timestamp(_velas(-0.2)[0]["timestamp"])


def test_registro_sin_parametros_libres_y_alias_cli():
    registro = STRATEGY_REGISTRY["TsmomLargoCortoStrategy"]
    assert registro["params"].keys() == {"snapshot"}
    assert cli_aliases()["tsmom-largo-corto"] == "TsmomLargoCortoStrategy"
    assert isinstance(build_strategy("TsmomLargoCortoStrategy", snapshot="m"), TsmomLargoCortoStrategy)
