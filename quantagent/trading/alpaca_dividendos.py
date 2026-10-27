"""Read Alpaca dividend activities for a trading day."""


def leer_dividendos_alpaca(client, day):
    activities = client.get("/account/activities/DIV", data={"date": day.isoformat()})
    return [
        (item.symbol, item.net_amount, "USD", "efectivo en cuenta")
        for item in activities
    ]
