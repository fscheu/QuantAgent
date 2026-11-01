import pandas as pd
import pytest

from quantagent.models import Order, TradeIntentRecord
from quantagent.trading.narrativa_m5 import NO_DISPONIBLE, nota_semanal
from quantagent.trading.pipeline_m5 import correr_diario, correr_semanal
from tests.test_pipeline_m5 import BrokerMemoria, SiempreSenala, mercado, DatosSinteticos  # noqa: F401


def _explota(_):
    raise RuntimeError("sin LLM")


def test_nota_semanal_indica_sectores_entrantes_y_salientes():
    prompts = []
    actual = pd.Series({"fecha": "2026-10-30", "habilitados": "A,C", "estado": "risk-on",
                        "z_inflacion": 1.2, "score_A": 0.8, "score_C": 0.7, "score_B": 0.9})
    anterior = pd.Series({"habilitados": "A,B"})
    texto = nota_semanal(actual, anterior, lambda prompt: prompts.append(prompt) or "nota")
    assert texto == "nota"
    assert "Entraron: C" in prompts[0]
    assert "Salieron: B" in prompts[0]
    assert "z_inflacion" in prompts[0]


def test_nota_semanal_sin_llm_o_con_error_es_no_disponible():
    fila = pd.Series({"habilitados": "A"})
    assert nota_semanal(fila, None, None) == NO_DISPONIBLE
    assert nota_semanal(fila, None, _explota) == NO_DISPONIBLE


def _correr(as_of, data, db, tmp_path, narrador):
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "regimen.csv"
    narr = tmp_path / "narrativa.csv"
    regimen = correr_semanal(as_of, data_provider=data, regimen_path=path,
                             narrador=narrador, narrativa_path=narr)
    broker = BrokerMemoria()
    intents = correr_diario(as_of, data_provider=data, db=db, broker=broker, regimen_path=path,
                            estrategias=(("test", SiempreSenala()),), narrador=narrador,
                            narrativa_path=narr)
    ordenes = [(o.symbol, o.side, o.order_type, o.quantity, o.stop_price) for o in broker.orders]
    campos = [(r.symbol, r.strategy_id, r.status, r.reject_reason, r.intent, r.reasoning_metadata)
              for r in intents]
    return regimen, campos, ordenes, path, narr


def test_llm_que_falla_no_cambia_regimen_intents_ni_ordenes(db_session, mercado, tmp_path):
    as_of, data = mercado
    sano = _correr(as_of, data, db_session, tmp_path / "a", lambda _: "texto")
    db_session.query(TradeIntentRecord).delete()
    db_session.query(Order).delete()
    db_session.commit()
    roto = _correr(as_of, data, db_session, tmp_path / "b", _explota)
    pd.testing.assert_series_equal(pd.Series(sano[0]), pd.Series(roto[0]))
    assert sano[1] == roto[1] and sano[1]
    assert sano[2] == roto[2] and sano[2]
    assert set(pd.read_csv(roto[4]).texto) == {NO_DISPONIBLE}
    assert "texto" in set(pd.read_csv(sano[4]).texto)


def test_narrativa_no_modifica_numeros_guardados(db_session, mercado, tmp_path):
    as_of, data = mercado
    con = _correr(as_of, data, db_session, tmp_path / "a", lambda _: "otro texto 123")
    csv_con = pd.read_csv(con[3])
    db_session.query(TradeIntentRecord).delete()
    db_session.query(Order).delete()
    db_session.commit()
    sin = _correr(as_of, data, db_session, tmp_path / "b", None)
    pd.testing.assert_frame_equal(csv_con, pd.read_csv(sin[3]))
    assert con[1] == sin[1]
