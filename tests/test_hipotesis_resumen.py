from datetime import datetime, timezone

from click.testing import CliRunner

from quantagent.cli.hipotesis import hipotesis_group
from quantagent.fabrica.estado import resumen_tanda
from quantagent.fabrica.registro_db import registrar_evento
from quantagent.models import HypothesisAttempt
from quantagent.database import SessionLocal


def _attempt(fecha, hipotesis, *, status=None, snapshot="real"):
    with SessionLocal() as db:
        db.add(HypothesisAttempt(ts=datetime.fromisoformat(fecha), hypothesis_id=hipotesis,
            snapshot=snapshot, strategy="s", asset="SPY", params={}, sharpe=0.5,
            trades=2, pnl=3, stage="0", status=status, extra={"desde": "", "hasta": ""}))
        db.commit()


def test_resumen_ultimos_fecha_causas_e_intentos(db_temporal):
    filas = [("2026-01-01T00:00:00+00:00", "vieja", "0", "CORTA", "sin historia"),
             ("2026-02-01T00:00:00+00:00", "h1", "0", "PASA", "aceptada"),
             ("2026-02-02T00:00:00+00:00", "h1", "3", "CORTA", "SPY: DSR < 0.95"),
             ("2026-02-02T00:00:00+00:00", "h2", "3", "CORTA", "XLK: DSR < 0.95"),
             ("2026-02-02T00:00:00+00:00", "h3", "2", "ALARMA", "filtro optimista")]
    for fecha, h, etapa, verdict, motivo in filas:
        registrar_evento(h, "etapa", etapa, verdict, motivo, fecha_utc=fecha)
        _attempt(fecha, h)
    runner = CliRunner()
    completo = runner.invoke(hipotesis_group, ["resumen"])
    assert completo.exit_code == 0
    assert "| 0 | CORTA | 1 |" in completo.output
    assert "| 3 | CORTA | 2 |" in completo.output
    assert "| 2 | ALARMA | 1 |" in completo.output
    assert "PASA" not in completo.output
    assert "DSR < 0.95: 2" in completo.output
    assert "Intentos: 5" in completo.output
    out = db_temporal / "resumen.md"
    filtrado = runner.invoke(hipotesis_group, ["resumen", "--desde", "2026-02-01", "--out", str(out)])
    assert filtrado.exit_code == 0
    assert "sin historia" not in filtrado.output
    assert "Intentos: 4" in filtrado.output
    assert out.read_text() == filtrado.output


def test_resumen_causa_en_varios_activos_cuenta_una_vez_por_hipotesis(db_temporal):
    for h, motivo in (("h1", "SPY: DSR < 0.95; XLK: DSR < 0.95; QQQ: DSR < 0.95"),
                      ("h2", "EFA: DSR < 0.95; IWM: DSR < 0.95")):
        registrar_evento(h, "etapa", "3", "CORTA", motivo, fecha_utc="2026-02-01T00:00:00+00:00")
        _attempt("2026-02-01T00:00:00+00:00", h)
    result = CliRunner().invoke(hipotesis_group, ["resumen"])
    assert result.exit_code == 0
    assert "- DSR < 0.95: 2" in result.output


def test_resumen_reconstruye_desde_tablas_sql(db_temporal):
    registrar_evento("H1", "etapa", "0", "PASA", "csv", fecha_utc="2026-02-01T00:00:00+00:00")
    _attempt("2026-02-01T00:30:00+00:00", "H1")
    _attempt("2026-02-01T01:00:00+00:00", "H1", status="invalida_bug", snapshot="s1")
    _attempt("2026-02-01T02:00:00+00:00", "H1", snapshot="sintetico")
    summary = resumen_tanda()
    assert "| 0 | PASA | 1 |" in summary
    assert "Intentos: 1" in summary
    assert resumen_tanda() == summary
    assert resumen_tanda(desde=datetime(2026, 2, 1, tzinfo=timezone.utc)) == summary
