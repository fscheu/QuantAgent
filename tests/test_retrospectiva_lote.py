"""QuantAgent-qr1n: retrospectiva por lote sin cifras de rendimiento."""
import csv
import yaml
from scripts.retrospectiva_lote import armar_retrospectiva, main


def test_retrospectiva_cuenta_causas_con_fichas_sinteticas(tmp_path):
    p_dir = tmp_path / "postmortem"
    p_dir.mkdir()
    for f in ({"id": "c1", "causas": ["pocos_trades", "rotacion_alta"]},
              {"id": "c2", "causas": ["pocos_trades"]},
              {"id": "c3", "causas": ["costos_comen_ganancia"]}):
        (p_dir / f"{f['id']}.yaml").write_text(yaml.safe_dump(f), encoding="utf-8")
    int_csv = tmp_path / "intentos.csv"
    with open(int_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["hipotesis", "activo", "sharpe", "dsr"])
        w.writerows([["c1", "S", "0.5", "0.9"], ["c2", "T", "0.2", "0.4"], ["otra", "S", "0.8", "0.99"]])
    out_file = tmp_path / "retro.md"
    texto, dest = armar_retrospectiva("L16", postmortem_dir=p_dir, intentos_path=int_csv, out=out_file)
    assert dest == out_file
    assert "| pocos_trades | 2 | c1, c2 |" in texto and "| rotacion_alta | 1 | c1 |" in texto
    assert "| costos_comen_ganancia | 1 | c3 |" in texto and "- Intentos sumados por el lote: 2" in texto


def test_retrospectiva_sin_cifras_de_rendimiento(tmp_path):
    p_dir = tmp_path / "postmortem"
    p_dir.mkdir()
    (p_dir / "c1.yaml").write_text(yaml.safe_dump({"id": "c1", "causas": ["drawdown"]}), encoding="utf-8")
    out_file = tmp_path / "retro.md"
    main(["L16", "-d", str(p_dir), "-n", "5", "-o", str(out_file)])
    assert not any(p in out_file.read_text(encoding="utf-8").lower() for p in ("sharpe", "dsr"))
