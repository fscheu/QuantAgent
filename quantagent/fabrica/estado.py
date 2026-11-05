"""Estado por hipótesis (plan M3): hasta qué etapa llegó y si pasó o cortó. Solo agregado."""
import re
from collections import Counter
from datetime import datetime, timezone

from quantagent.backtesting.verdict import filas_validas
COLUMNAS = ("fecha_utc", "hipotesis", "etapa", "veredicto", "motivo")
ETAPAS = ("0", "1", "2", "3")
VEREDICTOS = ("PASA", "CORTA", "ALARMA", "ERROR")
UNIDADES = ("ratio", "pct", "count", "years")
ETIQUETAS_CAUSA = {
    "sharpe_below": "Sharpe por debajo del umbral", "few_trades": "Pocos trades",
    "max_dd": "Drawdown máximo excedido", "history_short": "Historia insuficiente",
    "control_mismatch": "Control no coincide", "data_load": "Error al cargar datos",
    "admission": "No admitida", "runtime_error": "Error de ejecución",
    "filter_optimistic": "Filtro optimista", "v30_fail": "No pasa V30",
    "dsr_below": "DSR por debajo del umbral", "lock_closed": "Candado cerrado",
}


def medida(metric, value, threshold, comparator, unit):
    if comparator not in (">=", "<=", None):
        raise ValueError(f"comparator inválido: {comparator!r}")
    if unit not in (*UNIDADES, None):
        raise ValueError(f"unit inválida: {unit!r}")
    return {"metric": metric, "value": value, "threshold": threshold,
            "comparator": comparator, "unit": unit}


def causas_de_motivo(motivo):
    return {causa for causa in
            (re.sub(r"\b[A-Z][A-Z0-9.-]*:\s*", "", c).strip() for c in motivo.split(";"))
            if causa}


def registrar_estado(hipotesis_id, etapa, veredicto, motivo="", medida=None, causa=None):
    if etapa not in ETAPAS:
        raise ValueError(f"etapa inválida: {etapa!r}; debe ser una de {ETAPAS}")
    if veredicto not in VEREDICTOS:
        raise ValueError(f"veredicto inválido: {veredicto!r}; debe ser uno de {VEREDICTOS}")
    from quantagent.fabrica.registro_db import registrar_evento
    datos = {k: v for k, v in (("measure", medida), ("cause_code", causa)) if v is not None}
    registrar_evento(hipotesis_id, "etapa", etapa, veredicto, motivo, datos=datos)


def causa_de_evento(evento):
    codigo = (evento.get("datos") or {}).get("cause_code")
    if codigo is not None:
        return codigo, ETIQUETAS_CAUSA.get(codigo, codigo)
    texto = evento.get("motivo", "")
    causa = sorted(causas_de_motivo(texto))
    return (causa[0], causa[0]) if causa else (texto, texto)


def ultimo_estado(hipotesis_id):
    from quantagent.fabrica.registro_db import eventos_etapa
    etapas = eventos_etapa(hipotesis_id)
    if etapas:
        evento = etapas[-1]
        return {columna: evento.get(columna, "") for columna in COLUMNAS}
    return None


def resumen_tanda(desde=None):
    """Último estado por hipótesis e intentos dentro del período UTC indicado."""
    from quantagent.fabrica.registro_db import eventos_etapa, intentos
    eventos_etapa_registro = eventos_etapa()
    etapas = [e for e in eventos_etapa_registro
              if desde is None or datetime.fromisoformat(str(e.get("fecha_utc")).replace("Z", "+00:00")) >= desde]
    if etapas:
        ultimos = {}
        for evento in etapas:
            ultimos[evento.get("hipotesis")] = {columna: evento.get(columna, "") for columna in COLUMNAS}
    else:
        ultimos = {}
    conteos = Counter((r["etapa"], r["veredicto"]) for r in ultimos.values())
    causas = Counter(causa for r in ultimos.values() if r["veredicto"] != "PASA"
                     for causa in causas_de_motivo(r["motivo"]))
    desde_utc = desde or datetime.min.replace(tzinfo=timezone.utc)
    filas_intentos = [r for r in intentos() if datetime.fromisoformat(r["fecha_utc"]) >= desde_utc]
    intentos = sum(r.get("hipotesis") in ultimos for r in filas_validas(filas_intentos))
    lineas = ["# Resumen de la tanda", "", "| Etapa | Veredicto | Hipótesis |",
              "| --- | --- | ---: |"]
    lineas += [f"| {e} | {v} | {n} |" for (e, v), n in sorted(conteos.items())]
    lineas += ["", "## Causas más frecuentes", ""]
    lineas += [f"- {causa}: {n}" for causa, n in causas.most_common(5)]
    return "\n".join([*lineas, "", f"Intentos: {intentos}", ""])
