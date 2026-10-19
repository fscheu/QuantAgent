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
    ("Sharpe registrado", "En la tabla, la etapa 1 muestra el Sharpe con tasa 0 del filtro y las etapas 2 y 3 el Sharpe del motor con tasa libre de riesgo; V30 usa el de tasa 0, por eso un activo puede pasar V30 con Sharpe registrado negativo."),
    ("Intentos", "corridas registradas; más intentos, más riesgo de falso hallazgo."),
    ("Cartera", "combinación de piernas ponderadas por igual riesgo y evaluadas en conjunto."),
    ("Pierna", "estrategia con sus parámetros y universo de activos dentro de una cartera."),
    ("Igual riesgo", "ponderación inversamente proporcional a la volatilidad pasada, sin optimizar pesos."),
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
    dsr = None
    for fila in reversed(list(por_etapa.values())):
        if "dsr" in fila.get("motivo", "").lower():
            dsr = fila["motivo"]
            break
    from quantagent.cli.sweep import contar_intentos
    return {"id": hipotesis_id, "etapas": [por_etapa[e] for e in sorted(por_etapa)], "final": por_etapa[ultima],
            "activos": {i["activo"]: i for i in intentos if i["etapa"] == ultima},
            "intentos": len(intentos), "n_total": contar_intentos(),
            "dsr": dsr or "no evaluado", "prueba_abierta": abierto(hipotesis_id, "prueba")}


def render_informe(d, hipotesis=None):
    f = d["final"]
    li = "".join(f"<li>Etapa {escape(e['etapa'])}: <b>{escape(e['veredicto'])}</b> — {escape(e['motivo'])}</li>"
                 for e in d["etapas"])
    activos = sorted(d["activos"].items(), key=lambda x: (x[0].lower() != "cartera", x[0]))
    tr = "".join(f"<tr><td>{escape(a)}</td><td>{escape(i['sharpe'])}</td><td>{escape(i['trades'])}</td></tr>"
                 for a, i in activos)
    gl = "".join(f"<dt>{k}</dt><dd>{escape(v)}</dd>" for k, v in GLOSARIO)
    if hipotesis:
        secciones = "".join(
            f"<section><h3>Pierna: {escape(p.clase)}</h3>"
            f"<p><b>Universo:</b> {escape(p.universo)}</p>"
            f"<p><b>Activos:</b> {escape(', '.join(p.activos))}</p></section>"
            for p in hipotesis.piernas
        )
        extra = f"<p>{escape(hipotesis.justificacion)}</p><h2>Piernas</h2>{secciones}"
    else:
        extra = ""
    return (
        f"<!doctype html><html lang='es'><meta charset='utf-8'><title>Informe {escape(d['id'])}</title>"
        "<style>body{font:16px sans-serif;max-width:50em;margin:2em auto}td,th{padding:.2em .8em}</style>"
        f"<h1>{escape(d['id'])}</h1>{extra}<h2>Veredicto: {escape(f['veredicto'])} en la etapa {escape(f['etapa'])}</h2>"
        f"<ul>{li}</ul><h2>Activos de la etapa {escape(f['etapa'])}</h2>"
        f"<table><tr><th>Activo</th><th>Sharpe registrado</th><th>Trades</th></tr>{tr}</table>"
        f"<p>Intentos registrados: {d['intentos']}. Total de intentos (N): {d.get('n_total', d['intentos'])}. DSR: {escape(str(d.get('dsr', 'no evaluado')))}. Prueba abierta: {'sí' if d['prueba_abierta'] else 'no'}.</p>"
        f"<h2>Glosario</h2><dl>{gl}</dl></html>"
    )
