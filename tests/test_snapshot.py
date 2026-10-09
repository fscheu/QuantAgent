"""Snapshot Parquet con manifiesto (V01). No usa red ni el snapshot real."""

import json

import pandas as pd
import pytest

from quantagent.data.snapshot import SnapshotError, read_snapshot, verify_snapshot, write_snapshot


def _frame(offset: float = 0.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"], name="timestamp")
    return pd.DataFrame(
        {
            "open": [100.0 + offset, 101.0 + offset, 102.5 + offset],
            "high": [101.0 + offset, 102.0 + offset, 103.0 + offset],
            "low": [99.0 + offset, 100.5 + offset, 101.5 + offset],
            "close": [100.5 + offset, 101.5 + offset, 102.0 + offset],
            "adj_close": [100.1 + offset, 101.1 + offset, 101.6 + offset],
            "volume": [1000, 1200, 900],
        },
        index=idx,
    )


def test_escribe_y_relee_dataframe_identico(tmp_path):
    spy, tlt = _frame(), _frame(50.0)
    manifest = write_snapshot("t", {"SPY": spy, "TLT": tlt}, base_dir=tmp_path)

    pd.testing.assert_frame_equal(read_snapshot("t", "SPY", tmp_path, adjusted=False), spy)
    pd.testing.assert_frame_equal(read_snapshot("t", "TLT", tmp_path, adjusted=False), tlt)
    assert manifest["symbols"]["SPY"]["rows"] == 3
    assert (manifest["symbols"]["SPY"]["start"], manifest["symbols"]["SPY"]["end"]) == ("2020-01-02", "2020-01-06")
    assert verify_snapshot("t", tmp_path)["name"] == "t"


def test_un_byte_distinto_hace_fallar_verify_nombrando_el_archivo(tmp_path):
    write_snapshot("t", {"SPY": _frame(), "TLT": _frame(50.0)}, base_dir=tmp_path)
    path = tmp_path / "t" / "TLT.parquet"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))

    with pytest.raises(SnapshotError) as exc:
        verify_snapshot("t", tmp_path)
    assert "TLT.parquet" in str(exc.value)
    assert "SPY.parquet" not in str(exc.value)


def test_no_pisa_un_snapshot_existente(tmp_path):
    write_snapshot("t", {"SPY": _frame()}, base_dir=tmp_path)
    before = (tmp_path / "t" / "SPY.parquet").read_bytes()
    with pytest.raises(SnapshotError, match="ya existe"):
        write_snapshot("t", {"SPY": _frame(9.0)}, base_dir=tmp_path)
    assert (tmp_path / "t" / "SPY.parquet").read_bytes() == before


def test_verify_compara_contra_el_manifiesto_versionado(tmp_path):
    base, versioned = tmp_path / "data", tmp_path / "versioned"
    versioned.mkdir()
    manifest = write_snapshot("t", {"SPY": _frame()}, base_dir=base)
    (versioned / "t.json").write_text(json.dumps(manifest))
    verify_snapshot("t", base, versioned)  # coincide

    manifest["symbols"]["SPY"]["sha256"] = "0" * 64
    (versioned / "t.json").write_text(json.dumps(manifest))
    with pytest.raises(SnapshotError, match="SPY.parquet.*versionado"):
        verify_snapshot("t", base, versioned)


def _split_y_dividendo() -> pd.DataFrame:
    """Serie como la entrega Yahoo con `auto_adjust=False`: `close` y volumen ya ajustados por splits.

    Split 2:1 el 2020-01-08: el cierre no salta (51 -> 51) y el volumen está expresado en acciones post-split.
    Dividendo de 1.0 con ex-date el 2020-01-07: `adj_close` lo descuenta antes de esa fecha con el factor
    1 - 1.0 / 51 = 50/51 (51 es el cierre del 2020-01-06, el día previo al ex-date).
    """
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    close = pd.Series([50.0, 51.0, 51.0, 50.5, 51.0, 52.0], index=idx)
    adj_close = pd.Series([50.0 * 50 / 51, 50.0, 50.0, 50.5, 51.0, 52.0], index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": adj_close,
            "volume": [20, 40, 60, 80, 80, 90],
        },
        index=idx.rename("timestamp"),
    )


def test_ajuste_por_dividendo_sin_salto_en_el_split(tmp_path):
    raw = _split_y_dividendo()
    write_snapshot("t", {"SPY": raw}, base_dir=tmp_path)
    df = read_snapshot("t", "SPY", tmp_path)

    # Lo guardado no salta en la fecha del split; el dividendo es lo único que separa close de adj_close.
    assert df["close_sin_dividendos"].pct_change().abs().max() < 0.03  # un split mal ajustado daría -50%
    assert df["close_sin_dividendos"].tolist() == raw["close"].tolist()
    # Cierre ajustado calculado a mano: 51 * 50/51 = 50 antes del ex-date, igual después.
    assert df["close"].tolist() == pytest.approx([2500 / 51, 50.0, 50.0, 50.5, 51.0, 52.0])
    # Con el ajuste, el retorno del ex-date es +1% (50 -> 50.5); sin él sería -0.98% (51 -> 50.5).
    assert df["close"].pct_change().loc["2020-01-07"] == pytest.approx(0.01)
    assert df["close_sin_dividendos"].pct_change().loc["2020-01-07"] == pytest.approx(50.5 / 51 - 1)
    # Todo el OHLC se mueve con el mismo factor, calculado a mano en la primera fila (50/51).
    assert df["open"].iloc[0] == pytest.approx(49.5 * 50 / 51)
    assert df["high"].iloc[0] == pytest.approx(51.0 * 50 / 51)
    assert (df["low"] <= df["close"]).all() and (df["close"] <= df["high"]).all()
    # El volumen no se toca y adj_close se descarta.
    assert df["volume"].tolist() == raw["volume"].tolist()
    assert "adj_close" not in df.columns
