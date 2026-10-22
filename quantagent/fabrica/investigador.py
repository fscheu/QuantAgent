"""Investigador (QuantAgent-d4cq.42): pedido al LLM y escritura en `catalogo/propuestas/`, nunca en `hipotesis/`."""
import csv
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

from quantagent.data.snapshot import VERSIONED_DIR
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.propuestas import advertencias_estructurales, entradas_de_hipotesis, validar_propuesta
from quantagent.fabrica.transversal import METRICAS
from quantagent.strategy.registry import get_strategy_names

CATALOGO, POSTMORTEM_DIR, PROPUESTAS_DIR, HIPOTESIS_DIR = Path("catalogo/primas.yaml"), Path("catalogo/postmortem"), Path("catalogo/propuestas"), Path("hipotesis")
CLAVES_CAT = ("id", "prima", "idea", "fuente", "universo", "clase", "datos", "destino", "veredicto", "motivo")
ID_VALIDO = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _contar_intentos(intentos):
    if intentos is None:
        intentos = Path(os.environ.get("QUANTAGENT_SNAPSHOT_DIR", "local-data")) / "intentos.csv"
    if isinstance(intentos, int):
        return intentos
    if isinstance(intentos, (str, Path)):
        p = Path(intentos)
        return sum(1 for _ in csv.DictReader(p.read_text(encoding="utf-8").splitlines())) if p.exists() else 0
    return len(intentos)


def _manifiesto(nombre):
    raiz = Path(os.environ.get("QUANTAGENT_SNAPSHOT_DIR", "local-data"))
    ruta = raiz / nombre / "manifest.json"
    if not ruta.exists():
        ruta = VERSIONED_DIR / f"{nombre}.json"
    return json.loads(ruta.read_text(encoding="utf-8"))


def armar_pedido(catalogo=CATALOGO, postmortem=POSTMORTEM_DIR, n_intentos=None):
    """Arma el pedido para el LLM con el catálogo, reglas D1 y D2, N de intentos y fichas de I01."""
    datos = yaml.safe_load(Path(catalogo).read_text(encoding="utf-8")) if isinstance(catalogo, (str, Path)) and Path(catalogo).exists() else catalogo
    primas = (datos.get("primas", []) if isinstance(datos, dict) else datos) or []
    cat = [{**{k: e[k] for k in CLAVES_CAT if k in e},
            "advertencias": advertencias_estructurales({**e, "muestra_de_la_cita": e.get("muestra_de_la_cita", e.get("muestra"))})}
           for e in primas if isinstance(e, dict)]
    p_dir = Path(postmortem) if isinstance(postmortem, (str, Path)) else None
    raw = ([yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(p_dir.glob("*.y*ml"))] if p_dir.exists() else []) if p_dir else list(postmortem)
    fichas = [{"id": f["id"], "causas": list(f.get("causas") or [])} for f in raw if isinstance(f, dict) and "id" in f]
    ctx = yaml.safe_dump({"intentos_registrados": _contar_intentos(n_intentos), "catalogo": cat, "postmortem": fichas}, allow_unicode=True, sort_keys=False)
    activos = _manifiesto("etf-1d-2026-10-amplio")["symbols"]
    macro = _manifiesto("macro-2026-10")["series"]
    clases = sorted({*get_strategy_names(), *METRICAS})
    lista_negra = [
        {"id": e["id"], "cita": e.get("cita")}
        for e in [*primas, *(datos.get("descartadas_por_decaimiento", []) if isinstance(datos, dict) else [])]
        if isinstance(e, dict) and e.get("decaimiento")
    ]
    return (
        "Sos investigador de una cartera de trading. Proponé entre 0 y 3 entradas nuevas para el catálogo bajo `propuestas:`. Si ninguna cumple todo, devolvé `propuestas: []`; cada propuesta suma un intento.\n"
        "Cómo se evalúa: cada cartera se compara con comprar y mantener los mismos activos con pesos iguales y rebalanceo mensual, por resultado ajustado al riesgo en exceso de la tasa corta, en 2007-2018 y en una prueba única posterior, y después en paper. Una estrategia que solo decide cuándo estar comprada casi nunca supera esa vara: explicá su fuente de retorno adicional.\n"
        "Qué hay:\n"
        "Las advertencias del catálogo son informativas y no bloquean la admisión.\n"
        f"- Activos y fecha de inicio: {yaml.safe_dump({s: info['start'] for s, info in activos.items()}, allow_unicode=True, sort_keys=True).strip()}\n"
        f"- Series macro: {', '.join(sorted(macro))}. Velas diarias con apertura y cierre; no hay intradía, opciones ni futuros.\n"
        f"- Clases de señal existentes (registry.py y METRICAS): {', '.join(clases)}. Una clase nueva cuesta un ticket; marcala `clase_nueva: true`.\n"
        "- Motor: llenado en la apertura siguiente, 5 pb por lado, solo largo.\n"
        f"Qué no proponer sin evidencia posterior: {yaml.safe_dump(lista_negra, allow_unicode=True, sort_keys=False).strip()}\n"
        "Reglas: 0 parámetros libres; el universo se fija según dónde está definida la prima, no según resultados. Cada DOI tiene que existir y corresponder al artículo nombrado en `cita_texto`. Si la muestra de la cita llega a 2019 o después, no hay prueba ciega en 2019-2022 y va directo a reserva y paper. Descontá la mitad del efecto publicado al estimar `efecto_esperado`.\n\n"
        f"{ctx}"
    )


def escribir_propuestas(respuesta, carpeta=PROPUESTAS_DIR, catalogo=CATALOGO, hipotesis_dir=HIPOTESIS_DIR, crossref=None):
    """Valida el YAML devuelto por el LLM y escribe cada propuesta en `<carpeta>/<id>.yaml`."""
    destino = Path(carpeta)
    if destino.name == "hipotesis":
        raise ValueError("nunca escribir propuestas en hipotesis/")
    texto = re.sub(r"^```\w*\n|\n```$", "", (respuesta or "").strip()).strip()
    if not texto:
        return [], "YAML inválido: respuesta vacía"
    try:
        datos = yaml.safe_load(texto)
    except yaml.YAMLError as exc:
        return [], f"YAML inválido: {exc}"
    lista = datos.get("propuestas") if isinstance(datos, dict) and "propuestas" in datos else ([datos] if isinstance(datos, dict) and "id" in datos else datos)
    if not isinstance(lista, list) or not lista or any(not isinstance(p, dict) or not ID_VALIDO.match(str(p.get("id") or "")) for p in lista) or len({p["id"] for p in lista}) != len(lista):
        return [], "YAML inválido: se esperaba una lista de propuestas con id único y válido"
    cat_p, hip_p = Path(catalogo), Path(hipotesis_dir)
    primas = (yaml.safe_load(cat_p.read_text(encoding="utf-8")) or {}).get("primas") or [] if cat_p.exists() else []
    existentes = [*primas, *([e for f in sorted(hip_p.glob("*.y*ml")) for e in entradas_de_hipotesis(cargar_hipotesis(f))] if hip_p.exists() else [])]
    previas = [d for f in sorted(destino.glob("*.y*ml")) if isinstance(d := yaml.safe_load(f.read_text(encoding="utf-8")), dict)] if destino.exists() else []
    rutas, rechazadas = [], []
    for p in lista:
        opciones = {"previas": previas, "existentes": [*existentes, *(q for q in previas if q.get("id") != p["id"])]}
        if crossref is not None:
            opciones["crossref"] = crossref
        if m := validar_propuesta(p, **opciones):
            rechazadas.append(f"{p['id']}: {m}")
        else:
            destino.mkdir(parents=True, exist_ok=True)
            (ruta := destino / f"{p['id']}.yaml").write_text(yaml.safe_dump(p, allow_unicode=True, sort_keys=False), encoding="utf-8")
            rutas.append(ruta)
            previas.append(p)
    return rutas, ("\n".join(rechazadas) if rechazadas else None)


def proponer_con_llm(carpeta=None, catalogo=CATALOGO, postmortem=POSTMORTEM_DIR, hipotesis_dir=HIPOTESIS_DIR, n_intentos=None):
    """Llama al LLM vía QUANTAGENT_LLM_CMD y valida las propuestas con I04, I02 e I03."""
    if not (cmd := os.environ.get("QUANTAGENT_LLM_CMD", "").strip()):
        raise RuntimeError("QUANTAGENT_LLM_CMD no está definido")
    proc = subprocess.run(cmd, input=armar_pedido(catalogo, postmortem, n_intentos), text=True, shell=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"QUANTAGENT_LLM_CMD falló ({proc.returncode}): {(proc.stderr or proc.stdout or '').strip() or 'sin salida'}")
    rutas, motivo = escribir_propuestas(proc.stdout, PROPUESTAS_DIR if carpeta is None else carpeta, catalogo, hipotesis_dir)
    if not rutas and motivo and motivo.startswith("YAML inválido"):
        raise RuntimeError(motivo)
    return rutas, (motivo.splitlines() if motivo else [])
