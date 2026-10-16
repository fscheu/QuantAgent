# QuantAgent-d4cq.23: partición de E08

Estimación previa: 180–250 líneas entre validación, CLI y tests reales.
Supera el objetivo de 100; esta entrega no modifica código ni abre datos de prueba.
El padre queda in_progress hasta integrar los tres hijos, en orden en L10.
E09 sigue dependiendo del padre; no se adelanta el punto de control.

| Hijo | Objetivo | Aceptación binaria | Estimación | Revisión |
|---|---|---|---|---|
| QuantAgent-d4cq.23.1 | Validación reutilizable de etapa 3 | Motor real 2019–2022 e intentos etapa 3; V30 PASA con DSR 0,95 pasa, 0,94 corta, V30 NO PASA con 0,99 corta; basta un activo | 80–100 líneas | leer |
| QuantAgent-d4cq.23.2 | CLI, candado y estado | Abre prueba antes del validador; candado usado registra CORTA sin invocarlo; tabla V30/DSR, cuarto estado y exit code coherentes | 80–100 líneas | leer |
| QuantAgent-d4cq.23.3 | Integración real y reapertura | Snapshot 2007–2022 deja cuatro estados y una apertura; segunda corrida corta sin compare de prueba; centinelas desde 2023 no se usan | 60–90 líneas | leer |

Dependencias: .23.2 espera .23.1; .23.3 espera .23.2.
Cada hijo tiene sus archivos, comando dirigido y criterio ≤100 líneas en Beads.
La integración usa importorskip("backtesting") cuando llega al control real.
Los tests del CLI ajustan la expectativa anterior de tres estados donde corresponda.
Los artefactos de etapa 3 van en una subcarpeta para preservar etapa 2.
No se cambia el candado, la reserva ni los umbrales. Si un hijo excede su estimación,
se vuelve a partir antes de implementar; no se eliminan criterios del padre.
