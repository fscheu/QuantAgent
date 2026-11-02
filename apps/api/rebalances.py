"""Read-only strategy rebalance dates for the calendar endpoint."""

from datetime import timedelta


def proximos_rebalanceos(cuentas, hoy, dias, feriados, cargar):
    """Return (account, date) pairs where an active strategy rebalances in the window."""
    from quantagent.fabrica.transversal import semanales

    feriados = set(feriados)
    fin = hoy + timedelta(days=dias)
    primer_dia_mes = hoy.replace(day=1)
    inicio_calendario = primer_dia_mes - timedelta(days=primer_dia_mes.weekday())
    fin_calendario = (fin.replace(day=1) + timedelta(days=32)).replace(day=1)
    sesiones = []
    dia = inicio_calendario
    while dia < fin_calendario:
        if dia.weekday() < 5 and dia not in feriados:
            sesiones.append(dia)
        dia += timedelta(days=1)

    primeras_mes, primeras_semana, ultimas_mes = set(), set(), set()
    meses = {}
    semanas = {}
    for dia in sesiones:
        meses.setdefault((dia.year, dia.month), []).append(dia)
        semanas.setdefault(dia.isocalendar()[:2], []).append(dia)
    primeras_mes.update(dias_mes[0] for dias_mes in meses.values())
    primeras_semana.update(dias_semana[0] for dias_semana in semanas.values())
    for dias_mes in meses.values():
        ultimas_mes.update(dias_mes[-5:])

    resultado = []
    for cuenta in cuentas:
        if cuenta.status != "active":
            continue
        try:
            h = cargar(cuenta.strategy)
        except Exception:
            continue
        semanal = bool(semanales(h.piernas))
        fin_mes = any(p.clase == "rebalanceo-fin-de-mes" for p in h.piernas)
        for dia in sesiones:
            if hoy <= dia < fin and (dia in primeras_mes or (semanal and dia in primeras_semana)
                                     or (fin_mes and dia in ultimas_mes)):
                resultado.append((cuenta, dia))
    return resultado
