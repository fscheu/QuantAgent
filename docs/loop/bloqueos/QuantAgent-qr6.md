# Bloqueo: QuantAgent-qr6

El ticket pide que `StrategyAssembler` seleccione `paper_local` o `alpaca_paper`, y que construir un broker
rechace una URL live. En el lote actual solo existe `PaperBroker`; no hay adapter Alpaca ni campo de URL
configurado. Un selector que acepte `alpaca_paper` no puede construir un broker funcional sin ese adapter.

`QuantAgent-yme` pide implementar `AlpacaBroker` y depende de este ticket, así que implementar el adapter acá
invierte el alcance y deja sin una interfaz acordada qué constructor debe seleccionar el assembler. La
implementación de `settings.py`, `ConfigManager` y perfiles no resuelve ese bloqueo por sí sola: el criterio
de aceptación exige que el selector funcione a través del assembler.

Decisión para destrabar: definir si qr6 entrega solo el plumbing del selector y guardrail contra una futura
interfaz `AlpacaBroker`, o si yme pasa a ser quien implementa esa interfaz y conecta el selector. No se
modificó código ni se agregaron credenciales.
