"""QuantAgent-qr1n: retrospectiva por lote sin cifras de rendimiento."""
import yaml

from scripts.retrospectiva_lote import armar_retrospectiva, main


def test_retrospectiva_cuenta_causas_con_fichas_sinteticas(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    p_dir = tmp_path / "postmortem"
    p_dir.mkdir()
    for f in ({"id": "c1", "causas": ["pocos_trades", "rotacion_alta"]},
              {"id": "c2", "causas": ["pocos_trades"]},
              {"id": "c3", "causas": ["costos_comen_ganancia"]}):
        (p_dir / f"{f['id']}.yaml").write_text(yaml.safe_dump(f), encoding="utf-8")
    from quantagent.fabrica.registro_db import registrar_intento
    for hipotesis in ("c1", "c2", "sin_ficha"):
        registrar_intento("rsi", "SPY", "snap", "", 1.0, {}, {"desde": "2007-01-01", "hasta": "2018-12-31", "sharpe": 0.5,
                                                        "trades": 10, "pnl": 100.0}, hipotesis=hipotesis)
    out_file = tmp_path / "retro.md"
    texto, dest = armar_retrospectiva("L16", postmortem_dir=p_dir, out=out_file)
    assert dest == out_file
    assert "| pocos_trades | 2 | c1, c2 |" in texto and "| rotacion_alta | 1 | c1 |" in texto
    assert "| costos_comen_ganancia | 1 | c3 |" in texto and "- Intentos registrados de las carteras con ficha: 2" in texto
    assert "sin_ficha" not in texto  # cartera sin ficha no cuenta


def test_retrospectiva_sin_cifras_de_rendimiento(tmp_path):
    p_dir = tmp_path / "postmortem"
    p_dir.mkdir()
    (p_dir / "c1.yaml").write_text(yaml.safe_dump({"id": "c1", "causas": ["drawdown"]}), encoding="utf-8")
    out_file = tmp_path / "retro.md"
    main(["L16", "-d", str(p_dir), "-n", "5", "-o", str(out_file)])
    assert not any(p in out_file.read_text(encoding="utf-8").lower() for p in ("sharpe", "dsr"))
