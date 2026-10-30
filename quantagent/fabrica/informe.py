"""Informe HTML por hipótesis (plan M3): veredicto, motivos por etapa y tabla por activo."""
import csv
import json
from html import escape
from pathlib import Path

import pandas as pd

from quantagent.data.snapshot import snapshot_root
from quantagent.fabrica.candado import abierto
from quantagent.fabrica.criterio import recargo_paper
from quantagent.fabrica.estado import ESTADO_CSV


def diagnostico_exceso(hipotesis, carpeta_equity, snapshot, base_dir=None):
    """Calcula Sharpes con tasa libre de riesgo desde curvas guardadas, sin registrar intentos."""
    from quantagent.data import snapshot as snap_mod
    from quantagent.fabrica.cartera import curva_cartera
    from quantagent.fabrica.cartera import pesos as pesos_cartera
    from quantagent.fabrica.criterio import curva_benchmark, sharpes_en_exceso
    from quantagent.fabrica.transversal import leer_seleccion, semanales
    from quantagent.strategy.registry import STRATEGY_REGISTRY, cli_aliases

    retornos = {}
    fechas = None
    for i, pierna in enumerate(hipotesis.piernas):
        for activo in pierna.activos:
            path = Path(carpeta_equity) / f"{pierna.clase}-{activo}.csv"
            if not path.exists():
                path = Path(carpeta_equity) / f"{activo}.csv"
            if not path.exists():
                return {"disponible": False, "motivo": f"curva faltante: {activo}"}
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return {"disponible": False, "motivo": f"curva vacía: {activo}"}
            fecha_col = "date" if "date" in rows[0] else "timestamp"
            equity = pd.Series([float(r["equity"]) for r in rows],
                               index=pd.to_datetime([r[fecha_col] for r in rows]))
            fechas = equity.index if fechas is None else fechas.intersection(equity.index)
            retornos[(i, activo)] = equity
    if fechas is None or len(fechas) < 3:
        return {"disponible": False, "motivo": "curvas sin fechas comunes suficientes"}
    retornos = {k: v.reindex(fechas).pct_change().dropna() for k, v in retornos.items()}
    cierres = pd.DataFrame({a: snap_mod.read_snapshot(snapshot, a, base_dir)["close"].reindex(fechas)
                            for p in hipotesis.piernas for a in p.activos}, index=fechas)
    mapa = {i: list(p.activos) for i, p in enumerate(hipotesis.piernas)}
    try:
        seleccion = leer_seleccion(Path(carpeta_equity).parent, hipotesis.piernas, indice=cierres.index)
    except FileNotFoundError as exc:
        return {"disponible": False, "motivo": f"selección faltante: {exc.filename}"}
    posiciones_objetivo = any(STRATEGY_REGISTRY.get(cli_aliases().get(p.clase, p.clase))
                              and hasattr(STRATEGY_REGISTRY[cli_aliases().get(p.clase, p.clase)]["cls"], "posiciones_objetivo")
                              for p in hipotesis.piernas)
    if posiciones_objetivo:
        retornos_df = pd.DataFrame(retornos).fillna(0.0)
        cartera = 100_000.0 * (1 + retornos_df.T.groupby(level=1).sum().T.sum(axis=1)).cumprod()
    else:
        cartera = curva_cartera(cierres, mapa, retornos, seleccion=seleccion, semanales=semanales(hipotesis.piernas))
    if cartera is None or len(cartera) < 3:
        return {"disponible": False, "motivo": "curvas sin puntos suficientes"}
    w = (pd.DataFrame(1 / len(cierres.columns), index=cierres.index, columns=cierres.columns)
         if posiciones_objetivo else pesos_cartera(cierres, mapa, seleccion=seleccion, semanales=semanales(hipotesis.piernas)).T.groupby(level=1).sum().T)
    benchmark = curva_benchmark(cierres, pesos=w).reindex(cartera.index)
    try:
        sharpe_cartera, sharpe_benchmark = sharpes_en_exceso(cartera, benchmark, base_dir=base_dir)
    except ValueError as exc:
        return {"disponible": False, "motivo": str(exc)}
    return {"disponible": True, "sharpe_cartera": sharpe_cartera,
            "sharpe_benchmark": sharpe_benchmark}

GLOSARIO = (
    ("Sharpe", "retorno sobre volatilidad; en la etapa 1 es del filtro fuera de muestra, en las etapas 2 y 3 es del motor propio."),
    ("V30", "la regla de evaluar: al menos 2 trades, pnl > 0, Sharpe con tasa 0 >= 0,10 y caída máxima <= 35%."),
    ("DSR", "probabilidad de que el Sharpe no sea suerte, dado el total de intentos; no corta el pase pero fija el recargo de paper trading (tamaño y semanas)."),
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

    path_int = snapshot_root() / "intentos.csv"
    todas = list(csv.DictReader(path_int.open(newline="", encoding="utf-8"))) if path_int.exists() else []
    from quantagent.backtesting.verdict import filas_validas, n_efectivo
    n_crudo, n_eff = len(todas), n_efectivo(todas) if todas else 0
    path_epoca = snapshot_root() / "epoca_limpia.json"
    epoca = json.loads(path_epoca.read_text(encoding="utf-8")) if path_epoca.exists() else None

    fila3 = por_etapa.get("3")
    dsr_eff, dsr_crudo, anios_pos, bench, bench_mismo_peso, n_eff_criterio, recargo, pares = None, None, None, None, None, None, None, {}
    if fila3:
        pares = {k.strip(): v.strip() for p in fila3.get("motivo", "").split(";") if "=" in p for k, _, v in [p.partition("=")]}
        def _num(k, t):
            try:
                return t(pares[k]) if pares.get(k) not in (None, "", "-") else None
            except (ValueError, KeyError):
                return None
        dsr_eff, dsr_crudo, bench, bench_mismo_peso, n_eff_criterio = _num("dsr_efectivo", float), _num("dsr_crudo", float), _num("benchmark", float), _num("benchmark_mismo_peso", float), _num("n_efectivo", int)
        anios_pos = pares.get("anios_positivos") if pares.get("anios_positivos") not in (None, "", "-") else None
        rec_p = recargo_paper(dsr_eff) if dsr_eff is not None else None
        recargo = f"{rec_p.tamano_pct}% ({rec_p.semanas} sem)" if rec_p else None

    dsr = None
    for fila in reversed(list(por_etapa.values())):
        if "dsr" in fila.get("motivo", "").lower():
            dsr = fila["motivo"]
            break

    return {
        "id": hipotesis_id, "etapas": [por_etapa[e] for e in sorted(por_etapa)], "final": por_etapa[ultima],
        "activos": {i["activo"]: i for i in intentos if i["etapa"] == ultima},
        "intentos": len(intentos), "n_total": n_crudo,
        "intentos_validos": len(filas_validas(todas)),
        "intentos_invalidados": sum(f.get("estado") == "invalida_bug" for f in todas),
        "epoca_limpia": epoca,
        "n_crudo": n_crudo if fila3 and pares else None,
        "n_efectivo": n_eff_criterio if n_eff_criterio is not None else (n_eff if fila3 and pares else None),
        "dsr": dsr_eff if dsr_eff is not None else (dsr or "no evaluado"),
        "dsr_efectivo": dsr_eff, "dsr_crudo": dsr_crudo, "anios_positivos": anios_pos,
        "benchmark": bench, "benchmark_mismo_peso": bench_mismo_peso,
        "recargo": recargo, "prueba_abierta": abierto(hipotesis_id, "prueba"),
    }


def render_informe(d, hipotesis=None, diagnostico=None, advertencias=()):
    f = d["final"]
    epoca = d["epoca_limpia"]
    texto_epoca = (f"desde {escape(epoca['desde_utc'])}. Motivo: {escape(epoca['motivo'])}"
                   if epoca else "no definida")
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

    avisos = ""
    if advertencias:
        items = "".join(f"<li>{escape(codigo)}</li>" for codigo in advertencias)
        avisos = f"<h2>Advertencias etapa 0 (no bloquean)</h2><ul>{items}</ul>"

    tabla_e3 = ""
    if any(e.get("etapa") == "3" for e in d["etapas"]):
        def _fmt(v):
            return f"{v.tamano_pct}% ({v.semanas} sem)" if hasattr(v, "tamano_pct") else (f"{v:.2f}" if isinstance(v, float) else (str(v) if v not in (None, "") else "-"))
        vals = [_fmt(d.get(k)) for k in ("n_crudo", "n_efectivo", "dsr_efectivo", "dsr_crudo", "anios_positivos", "benchmark", "benchmark_mismo_peso", "recargo")]
        th = "".join(f"<th>{h}</th>" for h in ("N crudo", "N efectivo", "DSR efectivo", "DSR crudo", "Años positivos", "Benchmark 1/N (corte)", "Benchmark con pesos de cartera (diagnóstico)", "Recargo paper"))
        td = "".join(f"<td>{escape(v)}</td>" for v in vals)
        tabla_e3 = f"<h2>Criterio de etapa 3</h2><table><tr>{th}</tr><tr>{td}</tr></table>"

    exceso = ""
    if diagnostico is not None:
        if diagnostico.get("disponible"):
            valores = (f'{diagnostico["sharpe_cartera"]:.2f}', f'{diagnostico["sharpe_benchmark"]:.2f}')
        else:
            valores = ("no disponible", "no disponible")
        exceso = ("<h2>Sharpe en exceso de DGS3MO</h2><table><tr><th></th>"
                  "<th>Cartera</th><th>Benchmark 1/N</th></tr>"
                  f"<tr><td>Sharpe en exceso</td><td>{valores[0]}</td><td>{valores[1]}</td></tr></table>")

    return (
        f"<!doctype html><html lang='es'><meta charset='utf-8'><title>Informe {escape(d['id'])}</title>"
        "<style>body{font:16px sans-serif;max-width:50em;margin:2em auto}td,th{padding:.2em .8em}</style>"
        f"<h1>{escape(d['id'])}</h1>{extra}<h2>Veredicto: {escape(f['veredicto'])} en la etapa {escape(f['etapa'])}</h2>"
        f"<ul>{li}</ul>{avisos}<h2>Activos de la etapa {escape(f['etapa'])}</h2>"
        f"<table><tr><th>Activo</th><th>Sharpe registrado</th><th>Trades</th></tr>{tr}</table>"
        f"{tabla_e3}"
        f"{exceso}"
        f"<p>Época limpia: {texto_epoca}.</p>"
        f"<p>Registro global de intentos: {d['intentos_validos']} válidos; {d['intentos_invalidados']} invalidados por bug.</p>"
        f"<p>Intentos registrados: {d['intentos']}. Total de intentos (N): {d.get('n_total', d['intentos'])}. DSR: {escape(str(d.get('dsr', 'no evaluado')))}. Prueba abierta: {'sí' if d['prueba_abierta'] else 'no'}.</p>"
        f"<h2>Glosario</h2><dl>{gl}</dl></html>"
    )


def render_comparacion(c, id_nucleo, id_capa):
    """Informe HTML de `hipotesis comparar`: las dos carteras, los deltas y las dos cuentas de N."""
    def f(v):
        return "—" if v is None else f"{v:.4f}"
    return (
        f"<!doctype html><html lang='es'><meta charset='utf-8'><title>Comparación {escape(id_nucleo)} y {escape(id_capa)}</title>"
        "<style>body{font:16px sans-serif;max-width:50em;margin:2em auto}td,th{padding:.2em .8em}</style>"
        f"<h1>{escape(id_nucleo)} contra {escape(id_capa)}</h1>"
        f"<h2>Veredicto de la capa: {escape(c['veredicto'])}</h2><p>{escape(c['motivo'])}</p>"
        "<table><tr><th></th><th>Sharpe</th><th>DSR</th></tr>"
        f"<tr><td>Núcleo</td><td>{f(c['sharpe_nucleo'])}</td><td>{f(c['dsr_nucleo'])}</td></tr>"
        f"<tr><td>Núcleo + capa</td><td>{f(c['sharpe_capa'])}</td><td>{f(c['dsr_capa'])}</td></tr>"
        f"<tr><td>Diferencia</td><td>{c['delta_sharpe']:+.4f} ± {c['error_estandar_delta_sharpe']:.4f}</td><td>{f(c['delta_dsr'])}</td></tr></table>"
        f"<p>Años con Sharpe positivo: núcleo {c['anios_positivos_nucleo'].positivos} de {c['anios_positivos_nucleo'].totales}, "
        f"con capa {c['anios_positivos_capa'].positivos} de {c['anios_positivos_capa'].totales}.</p>"
        f"<p>N de intentos: {c['n_con_capa']} contando la capa, {c['n_sin_capa']} sin ella. "
        f"Margen de aceptación: {c['margen']}.</p></html>"
    )
