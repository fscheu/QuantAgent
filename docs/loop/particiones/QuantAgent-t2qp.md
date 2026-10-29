# Partición de QuantAgent-t2qp

El ticket cruza tres límites del embudo: señal por activo con dato macro de vintage, uso de la señal corta en el filtro de cartera, y configuración del YAML/CLI. Cada entrega queda acotada a 100 líneas y tiene aceptación verificable.

1. `QuantAgent-t2qp.1` — estrategia `tsmom-largo-corto` y señal mensual, con prueba sintética LONG/SHORT frente a DGS3MO. Revisión: leer.
2. `QuantAgent-t2qp.2` — integración de señal con filtro y cartera, pesos por 1/sigma y preservación long-only. Depende de .1. Revisión: leer.
3. `QuantAgent-t2qp.3` — habilitar el YAML oficial en el CLI hasta etapa 1, sin moverlo a `hipotesis/`. Depende de .2. Revisión: probar.

Los tickets hijos conservan los criterios binarios en Beads. El padre queda como agrupador y no se implementa directamente.
