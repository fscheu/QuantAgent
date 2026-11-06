"""Macro series exposed by the market endpoint."""


_SERIES = (
    ("T10Y2Y", "Curva del Tesoro", "daily", "pp"),
    ("CPIAUCSL", "Inflación", "monthly", "index"),
    ("UNRATE", "Desempleo", "monthly", "pct"),
    ("PAYEMS", "Empleo no agrícola", "monthly", "index"),
    ("INDPRO", "Producción industrial", "monthly", "index"),
    ("BAMLH0A0HYM2", "High yield OAS", "daily", "pct"),
    ("VIXCLS", "VIX", "daily", "index"),
    ("T5YIE", "Inflación esperada a 5 años", "daily", "pct"),
    ("DFII10", "Tasa real a 10 años", "daily", "pct"),
    ("DTWEXBGS", "Dólar amplio", "daily", "index"),
)


def series_macro():
    return [
        {
            "code": code,
            "title": title,
            "frequency": frequency,
            "status": "active" if code == "T10Y2Y" else "coming_soon",
            "unit": unit,
            "last": None,
            "change_1m": None,
            "stale": False,
            "expected_release": None,
            "points": [],
        }
        for code, title, frequency, unit in _SERIES
    ]
