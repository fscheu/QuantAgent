#!/usr/bin/env python3
"""Genera el mapa de lotes (HTML autocontenido) como grafo de commits.

Entradas:
  --issues    .beads/issues.jsonl de un clon de fscheu/QuantAgent. Si está dentro de un clon git,
              además se leen las copias de origin/main y de cada origin/lote/* y por ticket gana la
              de updated_at más nuevo (ninguna rama sola está al día).
  --sessions  sessions.json con sesiones activas y planificadas
  --lotes     lotes.json con la estructura M -> lote -> tickets
  --out       HTML de salida

Estado de cada lote: sale del PR del lote en GitHub (mergeado, abierto o completo esperando R:)
y solo si GitHub no responde se usa el `estado` escrito en lotes.json.

Uso:
  git fetch origin && git checkout -q origin/main
  python3 generar_mapa.py --issues .beads/issues.jsonl --out mapa.html
"""
import argparse, json, re, html, os, datetime, subprocess, urllib.request

REPO_GH = "fscheu/quantagent"

H = 76          # alto de fila
DX = 16         # separación entre carriles
X0 = 14

ESTADOS = ["cerrado", "activo", "desbloqueado", "bloqueado", "propuesto"]
ETIQ = {"cerrado": "cerrado", "activo": "en curso", "desbloqueado": "desbloqueado",
        "bloqueado": "bloqueado", "propuesto": "propuesto"}
LOTE_ETIQ = {"mergeado": "mergeado", "abierto": "abierto", "propuesto": "propuesto",
             "sin_lote": "sin lote", "listo": "completo, esperando R: de Fede"}


def esc(s):
    return html.escape(str(s), quote=True)


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def corto(i):
    return i.replace("QuantAgent-", "")


def titulo_corto(t):
    m = re.match(r"^([^:]{3,32}):\s+(.*)$", t)
    return m.group(2) if m else t


def detalle(d):
    d = d or ""
    m = re.search(r"\*\*Cambio requerido:?\*\*:?\s*(.*?)(\n\*\*|\Z)", d, re.S)
    txt = m.group(1) if m else d
    txt = re.sub(r"\*\*|`", "", txt).strip()
    return txt[:700] + ("…" if len(txt) > 700 else "")


def _leer_jsonl(txt, issues, fuente):
    rango = {"closed": 2, "in_progress": 1}
    for l in txt.splitlines():
        l = l.strip()
        if not l:
            continue
        i = json.loads(l)
        prev = issues.get(i["id"])
        clave = (i.get("updated_at") or "", rango.get(i.get("status"), 0))
        if prev is None or clave > (prev.get("updated_at") or "", rango.get(prev.get("status"), 0)):
            i["_fuente"] = fuente
            issues[i["id"]] = i


def _git(repo, *a, timeout=60):
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, timeout=timeout, env=env)


def cargar_issues(path):
    """Une las copias de .beads/issues.jsonl de origin/main y origin/lote/* del clon del cwd
    (gana el updated_at más nuevo). --issues es una fuente extra opcional."""
    issues = {}
    fuentes = []
    if path and os.path.isfile(path):
        _leer_jsonl(open(path, encoding="utf-8").read(), issues, "archivo")
        fuentes.append("archivo")
    try:
        top = _git(os.getcwd(), "rev-parse", "--show-toplevel").stdout.strip()
        if not top and path:
            top = _git(os.path.dirname(os.path.abspath(path)), "rev-parse", "--show-toplevel").stdout.strip()
        if top:
            f = _git(top, "fetch", "-q", "--prune", "origin", "+refs/heads/main:refs/remotes/origin/main",
                     "+refs/heads/lote/*:refs/remotes/origin/lote/*", timeout=180)
            if f.returncode != 0:
                print("aviso: git fetch falló (%s); uso las refs que ya hay en el clon" % f.stderr.strip()[:200])
            refs = _git(top, "for-each-ref", "--format=%(refname:short)",
                        "refs/remotes/origin/main", "refs/remotes/origin/lote/").stdout.split()
            for r in refs:
                out = _git(top, "show", "%s:.beads/issues.jsonl" % r)
                if out.returncode == 0 and out.stdout.strip():
                    _leer_jsonl(out.stdout, issues, r.replace("origin/", ""))
                    fuentes.append(r.replace("origin/", ""))
    except Exception as e:
        print("aviso: no pude leer las ramas de BEADS:", e)
    if not issues:
        raise SystemExit("error: no encontré ningún .beads/issues.jsonl (ni --issues ni origin/main u origin/lote/*)")
    return issues, fuentes


def estado_pr(n):
    """Estado real del PR en GitHub (REST): dict(state, merged, draft, merged_at) o None."""
    url = "repos/%s/pulls/%d" % (REPO_GH, n)
    try:
        out = subprocess.run(["gh", "api", url], capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return json.loads(out.stdout)
    except Exception:
        pass
    try:
        req = urllib.request.Request("https://api.github.com/" + url, headers={"Accept": "application/vnd.github+json"})
        tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if tok:
            req.add_header("Authorization", "Bearer " + tok)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:
        print("aviso: sin estado de GitHub para PR #%d (%s); uso lotes.json" % (n, e))
    return None


def cargar(args):
    base = os.path.dirname(os.path.abspath(__file__))
    issues, fuentes = cargar_issues(args.issues)
    ses = json.load(open(args.sessions or os.path.join(base, "sessions.json"), encoding="utf-8"))
    cfg = json.load(open(args.lotes or os.path.join(base, "lotes.json"), encoding="utf-8"))
    nuevos = sorted({label[5:] for issue in issues.values() for label in issue.get("labels", [])
                     if label.startswith("lote-") and label[5:] not in cfg["lotes"]}, key=nat)
    for lid in nuevos:
        cfg["lotes"][lid] = {
            "nombre": lid,
            "rama": "",
            "estado": "propuesto",
            "punto": "Lote detectado por label; completar nombre/punto en docs/mapa/lotes.json.",
        }
        cfg["prioridad_lotes"].append(lid)
        for milestone in cfg["ms"]:
            if milestone["id"] == "M3" and "lotes" in milestone:
                milestone["lotes"].append(lid)
    cfg["_fuentes"] = fuentes
    for L in cfg["lotes"].values():
        if L.get("pr"):
            pr = estado_pr(L["pr"])
            if pr:
                L["_pr"] = pr
                if pr.get("merged"):
                    L["estado"] = "mergeado"
                elif pr.get("state") == "open":
                    L["estado"] = "abierto"   # pasa a "listo" en construir() si no le queda nada abierto
    return issues, ses, cfg


def raiz(issues, i):
    """Sube por parent-child hasta el ticket que cuelga del epic d4cq (o sin padre)."""
    seen = set()
    while True:
        padres = [d["depends_on_id"] for d in issues[i].get("dependencies", [])
                  if d["type"] == "parent-child" and d["depends_on_id"] in issues]
        padres = [p for p in padres if issues[p]["issue_type"] != "epic"]
        if not padres or i in seen:
            return i
        seen.add(i)
        i = padres[0]


def estado_base(issues, cfg, i):
    if i in cfg["mergeados_no_cerrados"]:
        return "cerrado"
    return "cerrado" if issues[i]["status"] == "closed" else "abierto"


def subs_abiertas(issues, cfg):
    """Tickets con subtareas todavía abiertas (aunque BEADS cierre al padre)."""
    ab = {}
    for i in issues:
        r = raiz(issues, i)
        if r != i and estado_base(issues, cfg, i) != "cerrado" and issues[i]["status"] != "tombstone":
            ab[r] = ab.get(r, 0) + 1
    return ab


def construir(issues, ses, cfg):
    ocultar = set(cfg["ocultar"])
    prio = cfg["prioridad_lotes"]
    # asignación ticket -> lote por label (gana el lote más nuevo)
    asign = {}
    for i, t in issues.items():
        if i in ocultar or t["status"] == "tombstone":
            continue
        ls = [x[5:] for x in t.get("labels", []) if x.startswith("lote-") and x[5:] in prio]
        if ls:
            asign[i] = max(ls, key=prio.index)
    for lid, L in cfg["lotes"].items():
        for i in L.get("tickets", []):
            asign[i] = lid
    # subtareas -> cuentan en su raíz
    subs = {}
    nodos_ids = {}
    for i in asign:
        r = raiz(issues, i)
        if r != i:
            s = subs.setdefault(r, [0, 0])
            s[1] += 1
            s[0] += estado_base(issues, cfg, i) == "cerrado"
        else:
            nodos_ids[i] = asign[i]
    activas = {}
    for a in ses.get("activas", []):
        activas.setdefault(a["ticket"], []).append(a)
    planif = {}
    fantasmas = {}
    for p in ses.get("planificadas", []):
        if p["ticket"] in issues:
            planif.setdefault(p["ticket"], []).append(p)
        else:
            fantasmas.setdefault(p.get("lote", ""), []).append(p)

    sab = subs_abiertas(issues, cfg)

    def cerrado(i):
        return estado_base(issues, cfg, i) == "cerrado" and not sab.get(i)

    nodos = {}
    for i, lid in nodos_ids.items():
        t = issues[i]
        mc = re.search(r"Código en el plan: \*\*([A-Za-z0-9.]+)\*\*", t.get("description") or "")
        deps = [d["depends_on_id"] for d in t.get("dependencies", []) if d["type"] == "blocks"
                and d["depends_on_id"] in issues]
        deps += [d for d in cfg.get("deps_extra", {}).get(i, []) if d in issues and d not in deps]
        abiertos = [d for d in deps if not cerrado(d)]
        est_lote = cfg["lotes"][lid]["estado"]
        if cerrado(i):
            est = "cerrado"
        elif i in activas or t["status"] == "in_progress":
            est = "activo"
        elif est_lote in ("propuesto",):
            est = "propuesto"
        elif abiertos:
            est = "bloqueado"
        else:
            est = "desbloqueado"
        nodos[i] = dict(id=i, lote=lid, estado=est, titulo=titulo_corto(t["title"]),
                        desc=detalle(t.get("description")), deps=deps, espera=abiertos,
                        activas=activas.get(i, []) if est != "cerrado" else [],
                        planif=planif.get(i, []) if est != "cerrado" else [],
                        en_curso_beads=(t["status"] == "in_progress" and i not in activas and est == "activo"),
                        desactualizado=(i in cfg["mergeados_no_cerrados"] and t["status"] != "closed" and cerrado(i)),
                        pr=cfg["mergeados_no_cerrados"].get(i), subs=subs.get(i), code=mc.group(1) if mc else "",
                        prio=t.get("priority"), fantasma=False)
    for lid, lst in fantasmas.items():
        for k, p in enumerate(lst):
            gid = "sin-ticket-%s-%d" % (lid, k)
            nodos[gid] = dict(id=gid, lote=lid, estado="propuesto", titulo=p["ticket"],
                              desc="Sin ticket en BEADS todavía. %s: %s." % (p.get("maquina", ""), p.get("cuando", "")),
                              deps=[], espera=[], activas=[], planif=[p], desactualizado=False, en_curso_beads=False,
                              pr=None, subs=None, prio=None, fantasma=True, code="")
    # lote con PR abierto y nada pendiente: completo, esperando R: de Fede
    for lid, L in cfg["lotes"].items():
        if L.get("_pr") and L["estado"] == "abierto":
            ests = [n["estado"] for n in nodos.values() if n["lote"] == lid]
            if ests and all(e == "cerrado" for e in ests):
                L["estado"] = "listo"
    return nodos


def orden_topologico(ids, nodos):
    dentro = set(ids)
    prof = {}

    def p(i, vis=()):
        if i in prof:
            return prof[i]
        ds = [d for d in nodos[i]["deps"] if d in dentro and d not in vis]
        prof[i] = 0 if not ds else 1 + max(p(d, vis + (i,)) for d in ds)
        return prof[i]
    for i in ids:
        p(i)
    return sorted(ids, key=lambda i: (prof[i], nat(i)))


def carriles(orden, nodos):
    """Asigna carril por fila al estilo `git log --graph`. Devuelve filas y cantidad de carriles."""
    dentro = set(orden)
    hijos = {i: [] for i in orden}
    for i in orden:
        for d in nodos[i]["deps"]:
            if d in dentro:
                hijos[d].append(i)
    restan = {i: len(hijos[i]) for i in orden}
    lanes = []
    filas = []
    maxl = 1
    for t in orden:
        D = [d for d in nodos[t]["deps"] if d in dentro]
        pos = {d: lanes.index(d) for d in D if d in lanes}
        cand = [pos[d] for d in D if d in pos and restan[d] == 1]
        if cand:
            ln = min(cand)
        else:
            libres = [k for k, o in enumerate(lanes) if o is None]
            ln = libres[0] if libres else len(lanes)
            if ln == len(lanes):
                lanes.append(None)
        segs = []   # (tipo, lane_a, lane_b)
        for k, o in enumerate(lanes):
            if o is not None and o not in D:
                segs.append(("v", k, k))
        for d in D:
            if d not in pos:
                continue
            lp = pos[d]
            restan[d] -= 1
            segs.append(("v_top", lp, ln) if lp == ln else ("curva", lp, ln))
            if restan[d] > 0 and lp != ln:
                segs.append(("v", lp, lp))
            if restan[d] == 0 and lp != ln:
                lanes[lp] = None
        lanes[ln] = t if restan[t] > 0 else None
        if restan[t] > 0:
            segs.append(("v_bot", ln, ln))
        filas.append(dict(id=t, lane=ln, segs=segs))
        maxl = max(maxl, len(lanes))
    return filas, maxl


def svg_fila(f, nlanes, estado, fantasma):
    w = X0 * 2 + (nlanes - 1) * DX
    cy = H // 2
    out = ['<svg class="g" width="%d" height="%d" viewBox="0 0 %d %d" aria-hidden="true">' % (w, H, w, H)]
    x = lambda k: X0 + k * DX
    for tipo, a, b in f["segs"]:
        c = "var(--l%d)" % (a % 5)
        if tipo == "v":
            out.append('<line x1="%d" y1="0" x2="%d" y2="%d" stroke="%s"/>' % (x(a), x(a), H, c))
        elif tipo == "v_top":
            out.append('<line x1="%d" y1="0" x2="%d" y2="%d" stroke="%s"/>' % (x(a), x(a), cy, c))
        elif tipo == "v_bot":
            out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s"/>' % (x(a), cy, x(a), H, c))
        else:
            c2 = "var(--l%d)" % (b % 5)
            out.append('<path d="M%d 0 C%d %d %d %d %d %d" fill="none" stroke="%s"/>' % (
                x(a), x(a), cy * 0.7, x(b), cy * 0.5, x(b), cy, c2))
    cx = x(f["lane"])
    out.append('<circle class="dot d-%s" cx="%d" cy="%d" r="7"/>' % (estado, cx, cy))
    if estado == "activo":
        out.append('<circle class="halo" cx="%d" cy="%d" r="11"/>' % (cx, cy))
    out.append("</svg>")
    return "".join(out)


def fila_html(f, nlanes, n):
    e = n["estado"]
    metas = []
    for a in n["activas"]:
        metas.append('<span class="b b-ses">%s · %s</span>' % (esc(a["sesion"]), esc(a["maquina"])))
    if n.get("en_curso_beads"):
        metas.append('<span class="b b-ses">en curso según BEADS</span>')
    for p in n["planif"]:
        metas.append('<span class="b b-plan">▶ %s · %s</span>' % (esc(p.get("maquina", "")), esc(p["cuando"])))
    if n["desactualizado"]:
        metas.append('<span class="b b-des">BEADS desactualizado · mergeado%s</span>' % esc(" #%s" % n["pr"] if n["pr"] else ""))
    if n["espera"] and e in ("bloqueado",):
        metas.append('<span class="b b-esp">espera %s</span>' % esc(", ".join(corto(x) for x in n["espera"])))
    if n["subs"]:
        metas.append('<span class="b">%d/%d subtareas</span>' % (n["subs"][0], n["subs"][1]))
    idtxt = "sin ticket" if n["fantasma"] else (n["code"] + " · " + corto(n["id"]) if n["code"] else corto(n["id"]))
    ses = "; ".join("%s (%s): %s" % (a["sesion"], a["maquina"], a["rol"]) for a in n["activas"])
    plan = "; ".join("%s: %s" % (p.get("maquina", ""), p["cuando"]) for p in n["planif"])
    return (
        '<li class="tk st-%s" tabindex="0" data-id="%s" data-est="%s" data-titulo="%s" data-desc="%s" '
        'data-deps="%s" data-ses="%s" data-plan="%s">%s<div class="tx"><div class="t1">'
        '<span class="id">%s</span><span class="pill p-%s">%s</span></div>'
        '<div class="t2">%s</div><div class="t3">%s</div></div></li>'
    ) % (e, esc(idtxt), esc(ETIQ[e]), esc(n["titulo"]), esc(n["desc"]),
         esc(", ".join(corto(d) for d in n["deps"])), esc(ses), esc(plan),
         svg_fila(f, nlanes, e, n["fantasma"]), esc(idtxt), e, ETIQ[e], esc(n["titulo"]), "".join(metas))


def lote_html(lid, L, nodos, abierto):
    ids = [i for i, n in nodos.items() if n["lote"] == lid]
    orden = orden_topologico(ids, nodos)
    filas, nl = carriles(orden, nodos)
    cuenta = {e: 0 for e in ESTADOS}
    for i in ids:
        cuenta[nodos[i]["estado"]] += 1
    strip = "".join('<i class="mini d-%s" title="%s"></i>' % (nodos[i]["estado"], esc(corto(i))) for i in orden)
    resumen = " · ".join("%d %s" % (cuenta[e], ETIQ[e]) for e in ESTADOS if cuenta[e])
    est = LOTE_ETIQ.get(L["estado"], L["estado"])
    if L["estado"] == "listo" and L.get("pr"):
        est, pr = est + " en #%s" % L["pr"], ""
    else:
        pr = ' · PR #%s' % L["pr"] if L.get("pr") else ""
    rows = "".join(fila_html(f, nl, nodos[f["id"]]) for f in filas)
    lista = '<ol class="log">%s</ol>' % rows if rows else '<p class="vacio">Sin tickets en BEADS todavía.</p>'
    merge = '<div class="merge-fin"></div>' if L["estado"] == "mergeado" else ""
    return (
        '<details class="lote l-%s" data-k="lote-%s"%s><summary><span class="rama"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">'
        '<circle cx="3.5" cy="2.5" r="1.8"/><circle cx="3.5" cy="11.5" r="1.8"/><circle cx="10.5" cy="5.5" r="1.8"/>'
        '<path d="M3.5 4.3v5.4M10.5 7.3c0 2.2-3 2-7 3.2" fill="none"/></svg>%s</span>'
        '<span class="lnom">%s</span><span class="lbadge lb-%s">%s%s</span>'
        '<span class="strip">%s</span><span class="lres">%s</span></summary>'
        '<div class="lbody"><p class="punto"><b>Punto de control.</b> %s</p>%s</div>%s</details>'
    ) % (L["estado"], lid, " open" if abierto else "", esc(L["rama"]), esc(L["nombre"]),
         L["estado"], esc(est), esc(pr), strip, esc(resumen),
         esc(L["punto"]), lista, merge)


CSS = """
:root{
  /* Layout: tronco vertical a la izquierda (main), cada lote es una rama con sus tickets como commits */
  --bg:#f4f6f8; --panel:#ffffff; --fg:#17202b; --mut:#5d6b7a; --line:#d5dce4; --trunk:#9aa7b5;
  --ok:#2f8f6b; --act:#d9822b; --des:#2f6fd6; --blo:#8b97a5; --pro:#7a5bc0; --warn:#b4531f;
  --l0:#2f6fd6; --l1:#c0609a; --l2:#2f8f6b; --l3:#d9822b; --l4:#7a5bc0;
  --sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f141a; --panel:#171e26; --fg:#e4e9ef; --mut:#93a1b1; --line:#2a3541; --trunk:#4a5a6b;
  --ok:#4fc08f; --act:#f0a24f; --des:#6c9cf0; --blo:#6f7d8c; --pro:#a68ae8; --warn:#f08a54;
  --l0:#6c9cf0; --l1:#e287be; --l2:#4fc08f; --l3:#f0a24f; --l4:#a68ae8; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0f141a; --panel:#171e26; --fg:#e4e9ef; --mut:#93a1b1; --line:#2a3541; --trunk:#4a5a6b;
  --ok:#4fc08f; --act:#f0a24f; --des:#6c9cf0; --blo:#6f7d8c; --pro:#a68ae8; --warn:#f08a54;
  --l0:#6c9cf0; --l1:#e287be; --l2:#4fc08f; --l3:#f0a24f; --l4:#a68ae8; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.45 var(--sans);margin:0;
  padding-block:20px 120px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:26px;line-height:1.15;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--mut);margin:0 0 16px;font-size:13px}
.mono,.id,.rama{font-family:var(--mono)}
.bar{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:14px}
.leg{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12.5px;color:var(--mut)}
.leg span{display:inline-flex;align-items:center;gap:6px}
.btns{margin-left:auto;display:flex;gap:6px}
button{font:inherit;font-size:12.5px;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button:hover{border-color:var(--trunk)}
:focus-visible{outline:2px solid var(--des);outline-offset:2px}
.dot,.mini{display:inline-block}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block;border:2px solid transparent}
.d-cerrado,.sw.d-cerrado{fill:var(--ok);background:var(--ok);stroke:var(--ok)}
.d-activo,.sw.d-activo{fill:var(--act);background:var(--act);stroke:var(--act)}
.d-desbloqueado,.sw.d-desbloqueado{fill:var(--bg);stroke:var(--des);stroke-width:3;background:var(--panel);border-color:var(--des)}
.d-bloqueado,.sw.d-bloqueado{fill:var(--bg);stroke:var(--blo);stroke-width:2;stroke-dasharray:3 2;background:transparent;border:2px dashed var(--blo)}
.d-propuesto,.sw.d-propuesto{fill:var(--bg);stroke:var(--pro);stroke-width:2;stroke-dasharray:2 2;background:transparent;border:2px dashed var(--pro)}
.halo{fill:none;stroke:var(--act);stroke-width:1.5;opacity:.5}
@media (prefers-reduced-motion:no-preference){.halo{animation:pulso 2s ease-out infinite;transform-origin:center;transform-box:fill-box}
@keyframes pulso{from{opacity:.7;transform:scale(.8)}to{opacity:0;transform:scale(1.5)}}}
.mini{width:9px;height:9px;border-radius:50%;border:2px solid transparent;margin-right:3px}
.mini.d-cerrado{background:var(--ok)}.mini.d-activo{background:var(--act)}
.mini.d-desbloqueado{border-color:var(--des);background:var(--panel)}
.mini.d-bloqueado{border:2px dashed var(--blo);background:transparent}
.mini.d-propuesto{border:2px dashed var(--pro);background:transparent}
.ses{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:22px;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px 24px}
.ses h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin:0 0 6px;font-weight:600}
.ses ul{margin:0;padding:0;list-style:none;display:grid;gap:6px}
.ses li{font-size:13px;min-width:0}
.ses .k{font-family:var(--mono);font-size:12px;color:var(--mut)}
/* M */
.m{margin-bottom:26px}
.m>summary{list-style:none;cursor:pointer;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;padding:6px 0}
.m>summary::-webkit-details-marker{display:none}
.mtag{font-family:var(--mono);font-weight:600;font-size:12px;background:var(--fg);color:var(--bg);border-radius:4px;padding:2px 8px}
.mtit{font-size:18px;font-weight:600}
.mres{color:var(--mut);font-size:12.5px}
.m>summary::after{content:"▾";margin-left:auto;color:var(--mut)}
.m:not([open])>summary::after{content:"▸"}
.mbody{position:relative;padding:8px 0 4px}
.mbody::before{content:"";position:absolute;left:5px;top:0;bottom:0;width:2px;background:var(--trunk)}
.mlinea{display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;padding:10px 0;border-top:1px solid var(--line);margin-bottom:6px}
.mlinea p{margin:0;flex:1 1 360px;min-width:0;color:var(--mut);font-size:13px}
/* lote */
.lote{position:relative;margin:0 0 14px 34px}
.lote::before{content:"";position:absolute;left:-30px;top:0;width:32px;height:22px;border-left:2px solid var(--trunk);border-bottom:2px solid var(--trunk);border-bottom-left-radius:12px}
.lote::after{content:"";position:absolute;left:2px;top:22px;bottom:0;width:2px;background:var(--lc,var(--l0))}
.l-listo{--lc:var(--des)}.l-abierto{--lc:var(--act)}.l-mergeado{--lc:var(--ok)}.l-propuesto{--lc:var(--pro)}.l-sin_lote{--lc:var(--blo)}
.l-propuesto::after{background:repeating-linear-gradient(var(--pro) 0 5px,transparent 5px 9px)}
.lote>summary{list-style:none;cursor:pointer;display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;padding:0 0 6px 18px;min-height:44px;position:relative;z-index:1}
.lote>summary::-webkit-details-marker{display:none}
.rama{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:500;background:var(--panel);border:1px solid var(--lc);border-radius:5px;padding:2px 8px;color:var(--fg);min-width:0;overflow-wrap:anywhere}
.rama svg{stroke:var(--lc);stroke-width:1.5;fill:var(--panel);flex:none}
.lnom{font-weight:600}
.lbadge{font-size:11.5px;border-radius:99px;padding:1px 9px;border:1px solid var(--lc);color:var(--fg)}
.strip{display:inline-flex;flex-wrap:wrap;align-items:center}
.lres{color:var(--mut);font-size:12px}
.lbody{padding-left:18px}
.punto{margin:0 0 6px;color:var(--mut);font-size:13px;max-width:70ch}
.punto b{color:var(--fg)}
.log{list-style:none;margin:0;padding:0}
.tk{display:flex;gap:6px;min-height:__H__px;cursor:pointer;border-radius:6px;margin-left:-6px;padding-right:8px}
.tk:hover,.tk:focus-visible{background:color-mix(in srgb,var(--panel) 80%,var(--line))}
.tk svg.g{flex:none;display:block}
.tk svg.g line,.tk svg.g path{stroke-width:2}
.tx{padding:8px 0;min-width:0;flex:1}
.t1{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.id{font-size:12px;font-weight:600}
.pill{font-size:11px;border-radius:99px;padding:0 8px;border:1px solid currentColor;line-height:17px}
.p-cerrado{color:var(--ok)}.p-activo{color:var(--act)}.p-desbloqueado{color:var(--des)}.p-bloqueado{color:var(--blo)}.p-propuesto{color:var(--pro)}
.t2{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;font-size:13.5px}
.st-cerrado .t2{color:var(--mut)}
.t3{display:flex;gap:6px;flex-wrap:wrap;margin-top:2px;max-height:19px;overflow:hidden}
.b{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:0 6px;line-height:17px;white-space:nowrap;max-width:100%;overflow:hidden;text-overflow:ellipsis}
.b-ses{color:var(--act);border-color:var(--act)}
.b-plan{color:var(--des);border-color:var(--des)}
.b-des{color:var(--warn);border-color:var(--warn)}
.vacio{color:var(--mut);margin:4px 0}
.merge-fin{position:relative;height:22px;margin-left:-32px}
.merge-fin::before{content:"";position:absolute;left:2px;width:32px;top:-1px;height:23px;border-right:2px solid var(--ok);border-bottom:2px solid var(--ok);border-bottom-right-radius:12px;transform:translateX(-2px)}
.merge-fin::after{display:none}
.lote:not([open])>.merge-fin{display:none}
/* hoja de detalle */
#det{position:fixed;left:0;right:0;bottom:0;z-index:20;background:var(--panel);border-top:2px solid var(--trunk);
  padding:12px max(16px,env(safe-area-inset-right,0px)) calc(12px + env(safe-area-inset-bottom,0px)) max(16px,env(safe-area-inset-left,0px));
  box-shadow:0 -6px 24px rgba(0,0,0,.18);max-height:50vh;overflow:auto}
#det .in{max-width:980px;margin:0 auto;position:relative;padding-right:80px}
#det h3{margin:0 0 4px;font-size:15px}
#det p{margin:4px 0;font-size:13px;max-width:80ch;white-space:pre-wrap}
#det .x{position:absolute;right:0;top:0}
.avis{color:var(--warn);font-weight:500}
.pie{color:var(--mut);font-size:12px;margin-top:24px}
@media (max-width:560px){.lote{margin-left:28px}.lote::before{left:-24px;width:26px}.btns{margin-left:0}}
""".replace("__H__", str(H))

JS = """
(function(){
  var K='mapa-lotes:'; function g(k){try{return localStorage.getItem(K+k)}catch(e){return null}}
  function s(k,v){try{localStorage.setItem(K+k,v)}catch(e){}}
  var ds=document.querySelectorAll('details[data-k]');
  ds.forEach(function(d){var v=g(d.dataset.k); if(v==='1')d.open=true; else if(v==='0')d.open=false;
    d.addEventListener('toggle',function(){s(d.dataset.k,d.open?'1':'0')});});
  function all(sel,open){document.querySelectorAll(sel).forEach(function(d){d.open=open})}
  document.getElementById('b-exp').onclick=function(){all('details.m',true);all('details.lote',true)};
  document.getElementById('b-com').onclick=function(){all('details.lote',false)};
  document.getElementById('b-def').onclick=function(){
    try{Object.keys(localStorage).forEach(function(k){if(k.indexOf(K)===0)localStorage.removeItem(k)})}catch(e){}
    document.querySelectorAll('details[data-k]').forEach(function(d){d.open=d.hasAttribute('data-def')})};
  var det=document.getElementById('det');
  function show(li){
    var d=li.dataset, h='<h3><span class="mono">'+esc(d.id)+'</span> · '+esc(d.est)+'</h3><p><b>'+esc(d.titulo)+'</b></p>';
    if(d.desc) h+='<p>'+esc(d.desc)+'</p>';
    if(d.deps) h+='<p><b>Depende de:</b> '+esc(d.deps)+'</p>';
    if(d.ses) h+='<p><b>Sesión activa:</b> '+esc(d.ses)+'</p>';
    if(d.plan) h+='<p><b>Planificado:</b> '+esc(d.plan)+'</p>';
    det.querySelector('.body').innerHTML=h; det.hidden=false;}
  function esc(t){var e=document.createElement('div');e.textContent=t;return e.innerHTML}
  document.querySelectorAll('.tk').forEach(function(li){
    li.addEventListener('click',function(){show(li)});
    li.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();show(li)}});});
  det.querySelector('.x').onclick=function(){det.hidden=true};
})();
"""


def main():
    print("NOTA PARA LA RUTINA: la rama lote/cartera ya no existe (mergeada a main en el PR #253, 2026-10-18). "
          "Si el paso «git fetch origin lote/cartera» falló, es esperado: seguí. Este script baja solo "
          "origin/main y origin/lote/* y lee BEADS de ahí, sin importar qué esté checkouteado.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", default=".beads/issues.jsonl")
    ap.add_argument("--sessions")
    ap.add_argument("--lotes")
    ap.add_argument("--out", default="mapa.html")
    args = ap.parse_args()
    issues, ses, cfg = cargar(args)
    nodos = construir(issues, ses, cfg)

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.getcwd()).stdout.strip()
    except Exception:
        sha = ""
    fuentes_txt = " + ".join(f for f in cfg["_fuentes"] if f != "archivo") or "archivo dado"
    fuentes_txt += (" @ " + sha) if sha else ""
    ahora = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%MZ")

    totales = {e: 0 for e in ESTADOS}
    for n in nodos.values():
        totales[n["estado"]] += 1

    cerr_label = {}
    for i, t in issues.items():
        if t["status"] == "closed":
            for l in t.get("labels", []):
                cerr_label[l] = cerr_label.get(l, 0) + 1

    ms_html = []
    for m in cfg["ms"]:
        if not m["detalle"]:
            n = cerr_label.get(m["label"], 0)
            ms_html.append(
                '<div class="m"><div class="mlinea"><span class="mtag">%s</span><span class="mtit">%s</span>'
                '<p>%s <span class="mono">%d tickets cerrados en BEADS.</span></p></div></div>'
                % (esc(m["id"]), esc(m["titulo"]), esc(m["linea"]), n))
            continue
        lotes = []
        cnt = {e: 0 for e in ESTADOS}
        for lid in m["lotes"]:
            L = cfg["lotes"][lid]
            for n in nodos.values():
                if n["lote"] == lid:
                    cnt[n["estado"]] += 1
            abierto = L["estado"] == "abierto"
            h = lote_html(lid, L, nodos, abierto)
            if abierto:
                h = h.replace('<details class="lote', '<details data-def="1" class="lote', 1)
            lotes.append(h)
        res = " · ".join("%d %s" % (cnt[e], ETIQ[e]) for e in ESTADOS if cnt[e])
        ms_html.append(
            '<details class="m" data-k="m-%s" data-def="1" open><summary><span class="mtag">%s</span><span class="mtit">%s</span>'
            '<span class="mres">%s</span></summary><div class="mbody">%s</div></details>'
            % (esc(m["id"]), esc(m["id"]), esc(m["titulo"]), esc(res), "".join(lotes)))

    def vigente(tk):
        n = nodos.get(tk)
        return not (n and n["estado"] == "cerrado") and not (tk in issues and issues[tk]["status"] == "closed"
                                                              and tk not in nodos)
    act = "".join(
        '<li><b>%s</b> <span class="k">%s</span><br>%s</li>' % (
            esc(a["sesion"]), esc(a["maquina"] + " · " + corto(a["ticket"])), esc(a["rol"]))
        for a in ses.get("activas", []) if vigente(a["ticket"]))
    act += "".join(
        '<li><b>%s</b> <span class="k">%s</span></li>' % (esc(x["maquina"]), esc(x["estado"]))
        for x in ses.get("inactivas", []))
    act = act or "<li>Ninguna</li>"
    aviso = ""
    try:
        t0 = datetime.datetime.strptime(ses.get("actualizado", ""), "%Y-%m-%dT%H:%MZ").replace(tzinfo=datetime.timezone.utc)
        horas = (datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds() / 3600
        if horas > 24:
            aviso = ' <b class="avis">sessions.json sin tocar hace %d h: las sesiones pueden estar desactualizadas.</b>' % horas
    except ValueError:
        aviso = ' <b class="avis">sessions.json sin fecha de actualización.</b>'
    ses = dict(ses, planificadas=[p for p in ses.get("planificadas", []) if vigente(p["ticket"])])
    pl = "".join(
        '<li><span class="k">%s</span> %s<br><span class="k">%s</span></li>' % (
            esc(corto(p["ticket"]) if p["ticket"].startswith("QuantAgent-") else "sin ticket"),
            esc(p["cuando"] if p["ticket"].startswith("QuantAgent-") else p["ticket"] + " · " + p["cuando"]),
            esc(p.get("maquina", ""))) for p in ses.get("planificadas", []))
    pr = "".join("<li>%s</li>" % esc(x) for x in ses.get("propuestos", [])) or "<li>Ninguno</li>"

    leg = "".join('<span><i class="sw d-%s"></i>%s <b>%d</b></span>' % (e, ETIQ[e], totales[e]) for e in ESTADOS)

    doc = """<title>Mapa de lotes</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>%s</style>
<div class="wrap">
<h1>Mapa de lotes de QuantAgent</h1>
<p class="sub">Datos fijos. BEADS: %s, leído %s. Estado de lotes: PRs en GitHub. Sesiones: %s.%s Se regenera una vez por día o a demanda. Tocá un ticket para ver el detalle.</p>
<div class="bar"><div class="leg">%s</div>
<div class="btns"><button id="b-exp" type="button">Desplegar todo</button><button id="b-com" type="button">Compactar lotes</button><button id="b-def" type="button">Vista inicial</button></div></div>
<section class="ses" aria-label="Sesiones">
<div><h2>Sesiones activas</h2><ul>%s</ul></div>
<div><h2>Planificadas</h2><ul>%s</ul></div>
<div><h2>Lotes propuestos</h2><ul>%s</ul></div>
</section>
%s
<p class="pie">Un círculo es un ticket y la línea de cada rama une cada ticket con los que dependen de él. Los tickets cerrados que BEADS todavía no cerró se marcan «BEADS desactualizado». Los dependientes de otros lotes no se dibujan.</p>
</div>
<div id="det" hidden><div class="in"><div class="body"></div><button class="x" type="button">Cerrar</button></div></div>
<script>%s</script>
""" % (CSS, esc(fuentes_txt), ahora, esc(ses.get("actualizado", "?")), aviso, leg, act, pl, pr, "".join(ms_html), JS)
    open(args.out, "w", encoding="utf-8").write(doc)
    print("ok", args.out, len(doc), "bytes;", len(nodos), "nodos;", totales)


if __name__ == "__main__":
    main()
