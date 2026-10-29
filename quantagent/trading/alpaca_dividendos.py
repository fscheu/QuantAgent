"""Read Alpaca dividend activities for a trading day."""

from decimal import Decimal, InvalidOperation


def leer_dividendos_alpaca(client, day):
    activities = client.get("/account/activities/DIV", data={"date": day.isoformat()})
    dividends = []
    for item in activities:
        if not isinstance(item, dict) or not isinstance(item.get("symbol"), str) or not item["symbol"].strip():
            raise ValueError("Actividad DIV malformada: falta symbol válido")
        try:
            amount = Decimal(str(item.get("net_amount")))
            if not amount.is_finite():
                raise InvalidOperation
        except InvalidOperation as exc:
            raise ValueError("Actividad DIV malformada: net_amount debe ser numérico y finito") from exc
        dividends.append((item["symbol"], item["net_amount"], "USD", "efectivo en cuenta"))
    return dividends
