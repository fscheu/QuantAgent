# Cuenta chica para carteras

`quantagent backtest cuenta-chica --hipotesis <archivo.yaml> --snapshot <nombre>` simula una cartera de hipótesis con pesos operables y rebalanceo entero, junto con tarjetas de capital y benchmark de compra y mantenimiento. Las hipótesis con una pierna `curva-pendiente` requieren `--macro-snapshot <nombre>`, que carga T10Y2Y en modo punto-en-el-tiempo; la opción solo se admite con `--hipotesis`. El rango por defecto termina en 2018; `--to` desde 2019 se rechaza.

La retención de dividendos usa el valor de las posiciones al cierre previo.

La lógica de simulación queda fuera del CLI en `cartera_cuenta_chica.py`. La ruta de estrategia existente se conserva para `--strategy`.
