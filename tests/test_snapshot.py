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

    pd.testing.assert_frame_equal(read_snapshot("t", "SPY", tmp_path), spy)
    pd.testing.assert_frame_equal(read_snapshot("t", "TLT", tmp_path), tlt)
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
