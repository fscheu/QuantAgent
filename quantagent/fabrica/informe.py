"""Informe HTML por hipótesis (plan M3): veredicto, motivos por etapa y tabla por activo."""
import csv
from html import escape

from quantagent.data.snapshot import snapshot_root
from quantagent.fabrica.candado import abierto
from quantagent.fabrica.estado import ESTADO_CSV

GLOSARIO = (
    ("Sharpe", "retorno sobre volatilidad; en la etapa 1 es del filtro fuera de muestra, en las etapas 2 y 3 es del motor propio."),
    ("V30", "la regla de evaluar: al menos 2 trades, pnl > 0, Sharpe con tasa 0 >= 0,10 y caída máxima <= 35%."),
    ("DSR", "probabilidad de que el Sharpe no sea suerte, dado el total de intentos; pasa con 0,95."),
    ("Intentos", "corridas registradas; más intentos, más riesgo de falso hallazgo."),
)


def _filas(nombre, hipotesis_id):
    path = snapshot_root() / nombre
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [fila for fila in csv.DictReader(f) if fila["hipotesis"] == hipotesis_id]


def datos_informe(hipotesis_id):
    por_etapa = {fila["etapa"]: fila for fila in _filas(ESTADO_CSV, hipotesis_id)}
    if not por_etapa:
        raise ValueError(f"Hipótesis desconocida: {hipotesis_id}")
    intentos = _filas("intentos.csv", hipotesis_id)
    ultima = max(por_etapa)
    return {"id": hipotesis_id, "etapas": [por_etapa[e] for e in sorted(por_etapa)], "final": por_etapa[ultima],
            "activos": {i["activo"]: i for i in intentos if i["etapa"] == ultima},
            "intentos": len(intentos), "prueba_abierta": abierto(hipotesis_id, "prueba")}


def render_informe(d, hipotesis=None):
    f = d["final"]
    li = "".join(f"<li>Etapa {escape(e['etapa'])}: <b>{escape(e['veredicto'])}</b> — {escape(e['motivo'])}</li>"
                 for e in d["etapas"])
    tr = "".join(f"<tr><td>{escape(a)}</td><td>{escape(i['sharpe'])}</td><td>{escape(i['trades'])}</td></tr>"
                 for a, i in d["activos"].items())
    gl = "".join(f"<dt>{k}</dt><dd>{escape(v)}</dd>" for k, v in GLOSARIO)
    extra = f"<p>{escape(hipotesis.justificacion)} ({escape(hipotesis.clase)})</p>" if hipotesis else ""
    return (
        f"<!doctype html><html lang='es'><meta charset='utf-8'><title>Informe {escape(d['id'])}</title>"
        "<style>body{font:16px sans-serif;max-width:50em;margin:2em auto}td,th{padding:.2em .8em}</style>"
        f"<h1>{escape(d['id'])}</h1>{extra}<h2>Veredicto: {escape(f['veredicto'])} en la etapa {escape(f['etapa'])}</h2>"
        f"<ul>{li}</ul><h2>Activos de la etapa {escape(f['etapa'])}</h2>"
        f"<table><tr><th>Activo</th><th>Sharpe</th><th>Trades</th></tr>{tr}</table>"
        f"<p>Intentos registrados: {d['intentos']}. Prueba abierta: {'sí' if d['prueba_abierta'] else 'no'}.</p>"
        f"<h2>Glosario</h2><dl>{gl}</dl></html>"
    )
