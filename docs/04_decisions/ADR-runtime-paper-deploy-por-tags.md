# ADR: desarrollo y deploy del paper trading en la VM (runtime separado, deploy por tag)

**Estado:** propuesto, aprobado en esquema por Fede (2026-10-22). Se implementa en el ítem 15 de `QuantAgent-L21`.

## Decisión

1. **Checkout de runtime aparte** (`~/runtime/QuantAgent`). Solo se actualiza con `git checkout <tag>`. Nadie edita ahí. Los worktrees del loop, Codex y dev quedan fuera.
2. **Deploy manual por tag, no por merge a main.** Hay cambios que entran a main de forma automática (loop); un deploy atado al merge no tendría freno. El tag (`paper-AAAA-MM-DD`) lo crea el PM o Fede tras revisar. Rollback = volver al tag anterior.
3. **Reinicio**: servicio systemd de usuario; `systemctl --user restart` fuera de horario de mercado, sin órdenes pendientes. Al arrancar reconcilia posiciones contra Alpaca (`QuantAgent-y62`).
4. **DB propia del runtime**, separada de dev/tests. Backup antes de cada deploy. Migraciones alembic manuales en el deploy y compatibles hacia atrás.
5. **Claves**: `~/secrets/alpaca.env` solo lo carga el servicio de runtime. Dev y loop no tienen claves de Alpaca.
6. **Salvaguardas**: flag `PAPER_ENABLED`; bloqueo del endpoint live (`QuantAgent-qr6`).

## Descartado

Deploy con GitHub Actions al merge a main: no deja frenar un deploy productivo cuando main cambia solo.
