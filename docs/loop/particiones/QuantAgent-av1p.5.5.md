# Partición QuantAgent-av1p.5.5: digest de carteras

Estimación total: 275 líneas con pruebas, superior al objetivo de 100 (§3.4).
La consulta depende de ActivePosition y Trade; rebalancear escribe Position y Fill.
Además, Position conserva precios de entrada y la DB no persiste caja de la cuenta.
No se cambia código en esta entrega; el bug sigue abierto hasta integrar los hijos.

1. **QuantAgent-av1p.5.5.1 — consulta local (90 líneas).** Leer Position de la DB de cartera seleccionada, fills PAPER del día por fecha de ejecución y órdenes pendientes de cualquier fecha. Binario: SQLite sin ActivePosition/Trade devuelve posiciones, fills del día y pendientes previos, excluye LIVE y otras fechas; estrategias conservan sus tests. Revisión: probar.
2. **QuantAgent-av1p.5.5.2 — valuación actual (90 líneas).** Reutilizar AlpacaBroker.get_account/get_positions para equity, caja y valuación por símbolo local, sin persistir cambios. Binario: broker simulado devuelve equity 1000, caja 50 y posición 950 con P&L distinto del almacenado; API fallida o símbolo sin valuación produce error explícito. Revisión: probar.
3. **QuantAgent-av1p.5.5.3 — integración y presentación (95 líneas).** Depende de ambos anteriores. Activar las lecturas con --portfolio explícito, mostrar valor de mercado, P&L no realizado, equity, caja %, reserva objetivo 5 % y pendientes. Binario: test CLI con SQLite sin ActivePosition/Trade lista posición, fill diario, pendiente previo y caja 5 %; estrategias sin --portfolio conservan el digest; error API da aviso y exit 1. Revisión: probar.

Los dos primeros hijos preparan lecturas probadas sin activar el cambio en el CLI.
El tercero conecta esas lecturas y cubre los criterios del padre de extremo a extremo.
La valuación se identifica como foto actual de Alpaca, no saldo histórico de --date.
Un fill es actividad ejecutada, no un round trip ni P&L realizado por sí mismo.
No se tocan trading, credenciales, procesos, Hermes ni la reserva real de efectivo.
Tests dirigidos de cada hijo: en Beads. Smoke requerido en cada entrega.
