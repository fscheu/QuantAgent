# QuantAgent-L21: plan del lote de paper trading con Alpaca (M4)

**Estado:** aprobado por Fede (2026-10-22); lote activo, PR #412 abierto. La puesta en marcha en la VM durante dos semanas es posterior al merge y requiere autorización operativa explícita.
**Rama:** `lote/paper-alpaca`, desde `origin/main` (no depende de S07 ni de S09 de L19).
**Marco (Fede, DC R01):** paper es prueba del sistema, no evaluación de estrategias. Paper liviano: pesos iguales a la sombra salvo redondeo, ninguna orden rechazada sin explicar, dividendos bien tratados; si algo no coincide, se frena y se investiga. Ningún resultado de paper promueve una estrategia a dinero real.

## Estado de implementación (reconciliado 2026-10-23; PR #412)

- En la rama del lote ya están la interfaz `Broker`, `PaperBroker` y `AlpacaBroker` para paper; los tickets `13a`, `qr6`, `yme` y `y62` figuran cerrados en Beads.
- Cerrados y en main: `say`/`ev7d` (la salida solo cierra), `oc0h` (reversión: plana al 100 %), `dfa4` (salida intravela compartida), `7nle` (diseño de velas para operar), `kw84` (gate por lote).
- Claves de Alpaca paper en `~/secrets/alpaca.env`.
- **Estado actualizado:** D1–D3 (`av1p.1`–`.3`) ya entregan pesos, rebalanceador y sombra local. Falta integrar esa pieza en el ciclo del scheduler (`av1p.4`); hasta entonces no está completo el recorrido de operación automática de las dos carteras.
- Nota: núcleo+curva dio CORTA el criterio de paper (IM de `ro3w.1`: Sharpe 0,38 contra 0,61 de 1/N en 2019–2022). Entra a M4 igual, por el marco de arriba: es prueba de sistema.

## Decisiones tomadas por Fede

1. `0gh7.1`–`.4` entran en L21; `.5` queda en L20.
2. Las carteras no tienen stop propio; stop ±2 % solo para estrategias sueltas.

## Orden de tickets

Cada bloque es una sola cola; un ticket por vez. Los tickets nuevos se registran como hijos de `QuantAgent-av1p`; el trabajo pendiente de integración es `av1p.4`.

**A. Salida y riesgo sin broker**
1. `QuantAgent-beo` queda como decisión de Fede sobre el diagnóstico ya entregado en #83.
2. `QuantAgent-brwj` (X01): guard "vela posterior a la entrada"; cerrado e integrado en L21.

**B. Sizing por riesgo (decisión tomada; tickets .1–.4 integrados en L21)**
3. `0gh7.1` stop por defecto → `.2` regla `riesgo` → `.3` stop hasta `execute_decision` → `.4` configuración y flags. Los cuatro están cerrados e integrados en L21; `.5` (crossval) quedó en L20 y también está cerrado.

**C. Broker (tickets cerrados e integrados)**
4. `QuantAgent-13a`: contrato Broker, cerrado e integrado.
5. `QuantAgent-8659` y `.1`–`.2`: medición y reserva de efectivo, cerrados e integrados.
6. `QuantAgent-qr6`: selector de broker y bloqueo de live, cerrado e integrado.
7. `QuantAgent-yme`: `AlpacaBroker` paper (alpaca-py), cerrado e integrado.
8. `QuantAgent-y62`: reconciliación de posiciones DB local contra Alpaca, cerrada e integrada.

**D. Carteras en paper**
9. **D1** (`av1p.1`) pesos objetivo por fecha para las dos carteras. Cerrado.
10. **D2** (`av1p.2`) rebalanceador de pesos y posiciones actuales a órdenes. Cerrado; la prueba actual invoca la función localmente.
11. **D3** (`av1p.3`) sombra de backtest por fecha. Cerrado. La integración D4 aún debe conectarlos al scheduler.

**E. Integración, chequeos y puesta en marcha**
12. `QuantAgent-av1p.4` (D4): integrar D2/D3 en el ciclo del scheduler para ambas carteras, con `AlpacaBroker`; Beads lo hace depender de `.2`, `.3` y `yme.3`.
13. `QuantAgent-e4jc.13.1`–`.13.3` (P-1): chequeos en tres partes; `.13.3` ya está integrado al lote. La ruta del scheduler debe quedar conectada antes del piloto.
14. `QuantAgent-b41`: digest diario; depende de la reconciliación y D4 (Beads lo bloquea por `av1p.4`). Se elimina la referencia W14, inexistente.
15. `QuantAgent-55x` (P3, opcional): evaluar con la integración si hace falta corregir el reloj por velas.
16. Puesta en marcha en la VM: dos carteras durante 2 semanas sin intervención. Actividad posterior al merge; requiere autorización operativa explícita de Fede.

## Lo que no entra

Dinero real, cuenta margin y cortos (X11 queda en L20), escalera de incubación, DBC/USO, retención del 30 % de dividendos (todo se decide antes de dinero real, DC R01).

## Gate y paralelismo con la notebook

- Distinto lote que L19 (`lote/medicion-y-generador`, PR #360): con `LOOP_LOTE` el gate evalúa solo la entrega previa del lote activo (K01, en main), así que el PR loop pendiente de L19 no frena este lote. Si algún día la notebook abre un PR loop en `lote/paper-alpaca`, la VM no lanza hasta mergearlo.
- Archivos distintos (L19: informe y catálogo de hipótesis; L21: `quantagent/trading/`). El único roce probable es `PLAN-CONTINUACION.md` (§4 y la tabla de lotes): lo resuelve el PM al mergear.
- Agente: Codex con gpt-6-astra, un ticket por vez con `hermes_launch.sh --run --lote lote/paper-alpaca`.

## Cierre

PR del lote a main marcado listo y pedido de `R: lote-paper-alpaca vi: ... decido: merge`. La puesta en marcha (ítem 15) va después del merge.
