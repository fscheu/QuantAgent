from click.testing import CliRunner

from quantagent.cli.__main__ import cli
from quantagent.fabrica.registro_db import registrar_evento, registrar_intento


def test_cli_registro_lee_ciclos_eventos_y_recibo_desde_base(db_temporal):
    runner = CliRunner()
    opened = runner.invoke(cli, ["registro", "ciclo", "abrir", "2026-11", "H1"])
    assert opened.exit_code == 0, opened.output
    registrar_evento("H1", "etapa", "3", "PASA", ciclo="2026-11")
    registrar_intento("s", "SPY", "snap", None, None, {},
        {"desde": "2026-01-01", "hasta": "2026-01-02", "trades": 1, "pnl": 1, "sharpe": 1}, "H1")

    ver = runner.invoke(cli, ["registro", "ver", "--hipotesis", "H1"])
    assert ver.exit_code == 0 and "H1 | etapa | 3 | PASA" in ver.output
    calendario = runner.invoke(cli, ["registro", "calendario", "2026-11"])
    assert calendario.exit_code == 0 and "2026-11:" in calendario.output
    cerrado = runner.invoke(cli, ["registro", "ciclo", "cerrar", "2026-11"])
    assert cerrado.exit_code == 0 and "promovidas: 1" in cerrado.output
    informe = runner.invoke(cli, ["registro", "ciclo", "informe", "2026-11"])
    assert informe.exit_code == 0 and "1 promovieron" in informe.output
    corte = runner.invoke(cli, ["registro", "corte"])
    assert corte.exit_code == 0 and "consecutivos sin pasar etapa 3: 0" in corte.output
    assert "H1: 1" in informe.output


def test_cli_registro_recibo_se_recupera_sin_archivos_de_registro(db_temporal):
    from quantagent.fabrica.registro_db import abrir_ciclo, cerrar_ciclo, recibo_ciclo
    abrir_ciclo("2026-12", ["H2"])
    recibo = cerrar_ciclo("2026-12")
    assert recibo_ciclo("2026-12") == recibo
    assert not list(db_temporal.glob("hipotesis.csv"))
    assert not list(db_temporal.glob("registro-hipotesis*"))
