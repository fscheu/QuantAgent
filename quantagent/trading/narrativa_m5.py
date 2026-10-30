"""Narrativas opcionales para los resultados del pipeline M5."""

import logging
import os
from pathlib import Path

import pandas as pd

NO_DISPONIBLE = "no_disponible"
logger = logging.getLogger(__name__)


def _valor(fila, clave):
    return fila.get(clave) if hasattr(fila, "get") else getattr(fila, clave, None)


def nota_semanal(regimen_actual, regimen_anterior, llm):
    try:
        actual = set(filter(None, str(_valor(regimen_actual, "habilitados") or "").split(",")))
        anterior = set(filter(None, str(_valor(regimen_anterior, "habilitados") or "").split(","))) if regimen_anterior is not None else set()
        entraron = sorted(actual - anterior)
        salieron = sorted(anterior - actual)
        drivers = {k: _valor(regimen_actual, k) for k in regimen_actual.keys()
                   if str(k).startswith("z_") and pd.notna(_valor(regimen_actual, k))}
        scores = sorted(((k[len("score_"):], _valor(regimen_actual, k))
                         for k in regimen_actual.keys() if str(k).startswith("score_")
                         and _valor(regimen_actual, k) is not None and pd.notna(_valor(regimen_actual, k))),
                        key=lambda x: x[1], reverse=True)
        hechos = (f"Fecha: {_valor(regimen_actual, 'fecha')}. Estado: {_valor(regimen_actual, 'estado')}. "
                  f"Sectores habilitados: {_valor(regimen_actual, 'habilitados')}. "
                  f"Entraron: {', '.join(entraron) or 'ninguno'}. Salieron: {', '.join(salieron) or 'ninguno'}. "
                  f"Drivers macro (z): {drivers}. Tres mayores scores de los entrados: "
                  f"{[(s, v) for s, v in scores if s in entraron][:3]}.")
        return _llamar(llm, "Redactá una nota semanal breve en español basada solo en estos hechos: " + hechos)
    except Exception:
        logger.warning("No se pudo generar la narrativa semanal", exc_info=True)
        return NO_DISPONIBLE


def explicar_intent(intent, llm):
    try:
        prompt = ("Explicá brevemente en español este intent de trading, sin inventar datos: "
                  f"symbol={intent.symbol}, strategy_id={intent.strategy_id}, entry={intent.entry_price}, "
                  f"stop={intent.stop_loss}, confidence={intent.confidence}, reasoning={intent.reasoning}")
        return _llamar(llm, prompt)
    except Exception:
        logger.warning("No se pudo explicar el intent M5", exc_info=True)
        return NO_DISPONIBLE


def _llamar(llm, prompt):
    if llm is None:
        logger.warning("No hay LLM disponible para narrativa M5")
        return NO_DISPONIBLE
    try:
        return llm(prompt)
    except Exception:
        logger.warning("No se pudo generar narrativa M5", exc_info=True)
        return NO_DISPONIBLE


def llm_por_defecto():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None
    chat = ChatAnthropic(model="claude-haiku-4-5")
    return lambda prompt: chat.invoke(prompt).content


def guardar_narrativa(path, tipo, clave, texto):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fila = pd.DataFrame([[tipo, clave, texto]], columns=["tipo", "clave", "texto"])
    if path.exists():
        actual = pd.read_csv(path, dtype=str)
        actual = actual[~((actual.tipo == tipo) & (actual.clave == clave))]
        fila = pd.concat([actual, fila], ignore_index=True)
    fila.to_csv(path, index=False)
