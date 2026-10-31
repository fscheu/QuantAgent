"""Cuenta precios solo durante la pertenencia, incluyendo reclasificaciones. Sin red."""

import pandas as pd

from scripts.m5 import cobertura_precios as c


def test_pertenencia_parcial_y_cambio_sector(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(c, "load_manifest", lambda _: {
        "requested_start": "2023-03-16", "requested_end": "2023-03-20", "symbols": {"V": {}, "X": {}}})
    monkeypatch.setattr(c, "miembros", lambda _, d: ["V", "X", "Z"] if d.day < 20 else ["V", "Z"])
    monkeypatch.setattr(c, "sector", lambda _, t, d: "IT" if d.day < 17 else "Financials")
    monkeypatch.setattr(c, "read_snapshot", lambda _, t, **kw: pd.DataFrame(
        {"close": [10., 11.], "adj_close": [10., float("nan") if t == "V" else 11.]},
        index=pd.to_datetime(["2023-03-20", "2023-03-17"])))
    r = c.cobertura().set_index(["sector", "ticker"])
    assert r.loc[("IT", "V")].tolist() == [2023, 1, 0]
    assert r.loc[("Financials", "V")].tolist() == [2023, 2, 1]
    assert r.loc[("Financials", "X")].tolist() == [2023, 1, 1]
    assert r.loc[("Financials", "Z")].tolist() == [2023, 2, 0]
