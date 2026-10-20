#!/usr/bin/env python3
"""Arma la retrospectiva por lote (QuantAgent-qr1n) sin cifras de rendimiento."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


def buscar_intentos_csv(ruta=None):
    candidatos = [Path(ruta)] if ruta else [Path("local-data/intentos.csv"), Path("intentos.csv")]
    return next((c for c in candidatos if c.exists()), None)


def armar_retrospectiva(lote, postmortem_dir=None, intentos_path=None, intentos_n=None, out=None):
    p_dir = Path(postmortem_dir or "catalogo/postmortem")
    fichas = []
    if p_dir.exists():
        for p in sorted(p_dir.glob("*.y*ml")):
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                fichas.append({"id": d.get("id", p.stem), "causas": d.get("causas", [])})

    conteo, por_causa = Counter(), {}
    for f in fichas:
        for c in f["causas"]:
            conteo[c] += 1
            por_causa.setdefault(c, []).append(f["id"])

    if intentos_n is not None:
        n_intentos = int(intentos_n)
    else:
        csv_f, ids = buscar_intentos_csv(intentos_path), {f["id"] for f in fichas}
        if csv_f and ids:
            with open(csv_f, newline="", encoding="utf-8") as fp:
                n_intentos = sum(1 for r in csv.DictReader(fp) if r.get("hipotesis") in ids)
        else:
            n_intentos = 0

    filas_causas = [f"| {c} | {conteo[c]} | {', '.join(sorted(por_causa[c]))} |" for c in sorted(conteo)] or ["| (ninguna) | 0 | - |"]
    filas_carteras = [f"| {f['id']} | {', '.join(f['causas']) or '(ninguna)'} |" for f in fichas] or ["| (ninguna) | - |"]
    partes = [
        f"# Retrospectiva {lote}", "",
        "## Causas de corte", "", "| Causa | Conteo | Carteras |", "| --- | ---: | --- |", *filas_causas, "",
        "## Detalle por cartera", "", "| Cartera | Causas de corte |", "| --- | --- |", *filas_carteras, "",
        "## Registro", "", f"- Intentos registrados de las carteras con ficha: {n_intentos}", "",
    ]
    dest = Path(out or f"docs/06_implementation/retrospectiva-{lote}.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(partes), encoding="utf-8")
    return "\n".join(partes), dest


def main(argv=None):
    p = argparse.ArgumentParser(description="Retrospectiva por lote.")
    p.add_argument("lote")
    p.add_argument("--postmortem-dir", "-d")
    p.add_argument("--intentos-csv", "-i")
    p.add_argument("--intentos", "-n", type=int)
    p.add_argument("--out", "-o")
    args = p.parse_args(argv)
    _, dest = armar_retrospectiva(args.lote, args.postmortem_dir, args.intentos_csv, args.intentos, args.out)
    print(f"Retrospectiva generada en {dest}")


if __name__ == "__main__":
    main()
